import contextlib
import queue
import threading
import time
import traceback
from typing import List, Literal, Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import torch
import PyNvVideoCodec as nvc

from ..generator import ImageMatGenerator
from ..ImageMat import ColorType

logger = print


class _LiveBitstreamFeeder:
    """
    Feeds compressed OAK H264/H265 bytes into PyNvVideoCodec CreateDemuxer(callback).

    One feeder is used per encoded stream: RGB, left mono, right mono.
    """

    def __init__(self, bitstream_queue, stop_event):
        self.q = bitstream_queue
        self.stop_event = stop_event
        self.pending = bytearray()
        self.eof = False
        self.total_bytes_fed = 0

    def feed_chunk(self, demuxer_buffer):
        if self.stop_event.is_set():
            self.pending.clear()
            self.eof = True
            return 0

        capacity = len(demuxer_buffer)

        while len(self.pending) == 0 and not self.eof:
            if self.stop_event.is_set():
                self.eof = True
                break

            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.eof = True
                break

            self.pending.extend(item)

        if len(self.pending) == 0 and self.eof:
            return 0

        n = min(capacity, len(self.pending))
        demuxer_buffer[:n] = self.pending[:n]
        del self.pending[:n]
        self.total_bytes_fed += n
        return n


class _EncodedStreamRuntime:
    """Host-side state for one encoded DepthAI stream."""

    def __init__(self, name: str, bitstream_queue_size: int, stop_event: threading.Event):
        self.name = name
        self.depthai_q = None
        self.bitstream_q = queue.Queue(maxsize=bitstream_queue_size)
        self.feeder = _LiveBitstreamFeeder(self.bitstream_q, stop_event)
        self.demuxer = None
        self.decoder = None
        self.packet_iter = None
        self.producer_thread = None
        self.decode_thread = None
        self.pending_decoded_frames = []
        self.packet_count = 0
        self.byte_count = 0
        self.demux_packet_count = 0
        self.decoded_frame_count = 0
        self.latest_tensor = None
        self.latest_frame_index = 0
        self.latest_at = 0.0
        self.decoded_frame_refs = []

    def push_eof(self):
        try:
            self.bitstream_q.put_nowait(None)
            return
        except queue.Full:
            try:
                self.bitstream_q.get_nowait()
            except Exception:
                pass
            try:
                self.bitstream_q.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass


class _DepthAIPoeRGBStereoH26xBottomTorchTensorCapture:
    """
    RGB + stereo capture using H26x on all three streams.

    Device side:
        CAM_A RGB  -> NV12 -> VideoEncoder H264/H265
        CAM_B left -> NV12/YUV400p -> VideoEncoder H264/H265
        CAM_C right-> NV12/YUV400p -> VideoEncoder H264/H265

    Host side:
        RGB, left and right encoded streams are demuxed/decoded by PyNvVideoCodec.
        The latest decoded left/right grayscale frames are packed into the bottom
        rows of the RGB tensor. The returned tensor keeps the RGB-only shape:

            [1, 3, rgb_height, rgb_width]

        The bottom packed_stereo_rows rows are overwritten with:
            left.flatten(), right.flatten()
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.stop_event = threading.Event()

        self.rgb = _EncodedStreamRuntime("rgb", owner.rgb_bitstream_queue_size, self.stop_event)
        self.left = _EncodedStreamRuntime("left", owner.stereo_bitstream_queue_size, self.stop_event)
        self.right = _EncodedStreamRuntime("right", owner.stereo_bitstream_queue_size, self.stop_event)

        self.combined_frame_count = 0
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_combined_count = 0

        self._latest_stereo_lock = threading.Lock()
        self._stereo_ready = threading.Event()

        self._released = False
        self._exit_stack = contextlib.ExitStack()

        self._start()

    def _open_device(self):
        src = str(self.source).strip()
        if src.startswith("depthai://"):
            src = src.replace("depthai://", "", 1).strip()
        if src in ("", "auto", "default", "none", "None"):
            return dai.Device()
        return dai.Device(dai.DeviceInfo(src))

    @staticmethod
    def _camera_socket(socket_name: str):
        if hasattr(dai.CameraBoardSocket, socket_name):
            return getattr(dai.CameraBoardSocket, socket_name)
        aliases = {"RGB": "CAM_A", "LEFT": "CAM_B", "RIGHT": "CAM_C"}
        alias = aliases.get(socket_name)
        if alias and hasattr(dai.CameraBoardSocket, alias):
            return getattr(dai.CameraBoardSocket, alias)
        raise ValueError(f"Unsupported camera socket: {socket_name}")

    @staticmethod
    def _depthai_profile(codec: str):
        if codec == "h265":
            return dai.VideoEncoderProperties.Profile.H265_MAIN
        if codec == "h264":
            return dai.VideoEncoderProperties.Profile.H264_MAIN
        raise ValueError(f"Unsupported codec: {codec}")

    @staticmethod
    def _output_color_type(name: str):
        if name == "rgbp":
            return nvc.OutputColorType.RGBP
        if name == "rgb":
            return nvc.OutputColorType.RGB
        if name == "native":
            return nvc.OutputColorType.NATIVE
        raise ValueError(f"Unsupported decoder_output_color: {name}")

    @staticmethod
    def _img_frame_type(type_name: str):
        if hasattr(dai.ImgFrame.Type, type_name):
            return getattr(dai.ImgFrame.Type, type_name)

        # Older/newer DepthAI builds expose 8-bit mono using slightly different
        # enum names. Prefer YUV400p for VideoEncoder, because the encoder
        # warning explicitly says it accepts NV12 or YUV400p.
        if type_name == "GRAY8":
            for fallback in ("YUV400p", "YUV400P", "RAW8"):
                if hasattr(dai.ImgFrame.Type, fallback):
                    return getattr(dai.ImgFrame.Type, fallback)

        if type_name == "YUV400p":
            for fallback in ("YUV400P", "GRAY8", "RAW8"):
                if hasattr(dai.ImgFrame.Type, fallback):
                    return getattr(dai.ImgFrame.Type, fallback)

        raise ValueError(f"Unsupported DepthAI ImgFrame.Type: {type_name}")

    @staticmethod
    def _resize_mode(mode_name: str):
        if hasattr(dai.ImgResizeMode, mode_name):
            return getattr(dai.ImgResizeMode, mode_name)
        raise ValueError(f"Unsupported DepthAI ImgResizeMode: {mode_name}")

    @staticmethod
    def _packet_to_bytes(packet):
        data = packet.getData()
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        arr = np.asarray(data, dtype=np.uint8)
        return arr.tobytes()

    @staticmethod
    def _get_low_latency_enum():
        if hasattr(nvc, "DisplayDecodeLatencyType"):
            if hasattr(nvc.DisplayDecodeLatencyType, "LOW"):
                return nvc.DisplayDecodeLatencyType.LOW
        if hasattr(nvc, "DisplayDecodeLatency"):
            enum = nvc.DisplayDecodeLatency
            for name in ("DISPLAYDECODELATENCY_LOW", "LOW"):
                if hasattr(enum, name):
                    return getattr(enum, name)
        return None

    @staticmethod
    def _set_end_of_picture(packet):
        if not hasattr(nvc, "VideoPacketFlag"):
            return
        flag = None
        for name in ("ENDOFPICTURE", "END_OF_PICTURE"):
            if hasattr(nvc.VideoPacketFlag, name):
                flag = getattr(nvc.VideoPacketFlag, name)
                break
        if flag is None:
            return
        for attr in ("decode_flag", "flags"):
            try:
                setattr(packet, attr, flag)
                return
            except Exception:
                pass

    def _create_depthai_pipeline(self):
        owner = self.owner

        raw_pipeline = dai.Pipeline(self.device)
        if hasattr(raw_pipeline, "__enter__"):
            pipeline = self._exit_stack.enter_context(raw_pipeline)
        else:
            pipeline = raw_pipeline

        rgb_socket = self._camera_socket(owner.rgb_camera_socket)
        left_socket = self._camera_socket(owner.left_camera_socket)
        right_socket = self._camera_socket(owner.right_camera_socket)

        rgb_cam = pipeline.create(dai.node.Camera).build(rgb_socket)
        left_cam = pipeline.create(dai.node.Camera).build(left_socket)
        right_cam = pipeline.create(dai.node.Camera).build(right_socket)

        rgb_nv12 = rgb_cam.requestOutput(
            (owner.rgb_width, owner.rgb_height),
            dai.ImgFrame.Type.NV12,
            self._resize_mode(owner.rgb_resize_mode),
            owner.capture_fps,
        )

        mono_type = self._img_frame_type(owner.stereo_encoder_input_type)
        stereo_resize_mode = self._resize_mode(owner.stereo_resize_mode)
        left_gray = left_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            mono_type,
            stereo_resize_mode,
            owner.capture_fps,
        )
        right_gray = right_cam.requestOutput(
            (owner.stereo_width, owner.stereo_height),
            mono_type,
            stereo_resize_mode,
            owner.capture_fps,
        )

        rgb_encoder = pipeline.create(dai.node.VideoEncoder).build(
            rgb_nv12,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.rgb_codec),
        )
        left_encoder = pipeline.create(dai.node.VideoEncoder).build(
            left_gray,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.stereo_codec),
        )
        right_encoder = pipeline.create(dai.node.VideoEncoder).build(
            right_gray,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.stereo_codec),
        )

        for enc_name, enc, bitrate in (
            ("RGB", rgb_encoder, owner.rgb_bitrate_kbps),
            ("left", left_encoder, owner.stereo_bitrate_kbps),
            ("right", right_encoder, owner.stereo_bitrate_kbps),
        ):
            try:
                enc.setBitrateKbps(int(bitrate))
            except Exception as e:
                logger(f"Warning: could not set OAK {enc_name} encoder bitrate: {e}")
            try:
                enc.setKeyframeFrequency(int(owner.capture_fps))
            except Exception:
                pass

        self.rgb.depthai_q = rgb_encoder.out.createOutputQueue(
            maxSize=owner.rgb_depthai_queue_size,
            blocking=True,
        )
        self.left.depthai_q = left_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=True,
        )
        self.right.depthai_q = right_encoder.out.createOutputQueue(
            maxSize=owner.stereo_depthai_queue_size,
            blocking=True,
        )

        pipeline.start()
        return pipeline

    def _start(self):
        owner = self.owner

        for label, width in (("rgb_width", owner.rgb_width), ("stereo_width", owner.stereo_width)):
            if width % 32 != 0:
                raise ValueError(
                    f"DepthAI H264/H265 encoder requires {label} to be a multiple of 32; got {width}."
                )

        stereo_values = 2 * int(owner.stereo_width) * int(owner.stereo_height)
        values_per_row = 3 * int(owner.rgb_width)
        payload_rows = (stereo_values + values_per_row - 1) // values_per_row
        payload_start = int(owner.rgb_height) - payload_rows
        if payload_start < 0:
            raise ValueError(
                "Stereo payload does not fit inside RGB bottom rows: "
                f"payload_rows={payload_rows}, rgb_height={owner.rgb_height}."
            )

        logger("Opening DepthAI device...")
        self.device = self._open_device()

        logger("Connected DepthAI device:")
        logger(f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        logger(f"  Cameras: {self.device.getConnectedCameras()}")
        logger("")
        logger("Starting DepthAI RGB + H26x stereo bottom-pack tensor pipeline v4:")
        logger(f"  Source: {self.source}")
        logger(f"  RGB socket: {owner.rgb_camera_socket}")
        logger(f"  Left socket: {owner.left_camera_socket}")
        logger(f"  Right socket: {owner.right_camera_socket}")
        logger(f"  RGB size: {owner.rgb_width}x{owner.rgb_height}")
        logger(f"  Stereo size: {owner.stereo_width}x{owner.stereo_height}")
        logger(f"  Capture FPS: {owner.capture_fps}")
        logger(f"  OAK RGB encoder: {owner.rgb_codec.upper()} @ {owner.rgb_bitrate_kbps} kbps")
        logger(f"  OAK stereo encoders: {owner.stereo_codec.upper()} @ {owner.stereo_bitrate_kbps} kbps each")
        logger(f"  RGB decoder output: {owner.decoder_output_color}")
        logger(f"  Stereo decoder output: {owner.stereo_decoder_output_color}")
        logger(f"  Stereo encoder input type: {owner.stereo_encoder_input_type}")
        logger(f"  Output: [1, 3, {owner.rgb_height}, {owner.rgb_width}]")
        logger(f"  Stereo payload rows inside RGB bottom: {payload_rows}")
        logger(f"  Stereo payload start row: {payload_start}")
        logger("  Note: bottom RGB rows are overwritten by stereo payload")
        logger(f"  normalize_rgb: {owner.normalize_rgb}")
        logger(f"  normalize_stereo: {owner.normalize_stereo}")
        logger(f"  GPU ID: {owner.gpu_id}")
        logger("")

        self.pipeline = self._create_depthai_pipeline()

        for stream in (self.rgb, self.left, self.right):
            stream.producer_thread = threading.Thread(
                target=self._encoded_producer_loop,
                args=(stream,),
                daemon=True,
            )
            stream.producer_thread.start()

        # Create demuxers/decoders after producer threads start, because
        # CreateDemuxer(callback) may block until enough header bytes arrive.
        logger("Creating PyNvVideoCodec demuxer/decoder for RGB stream...")
        self._create_stream_decoder(
            self.rgb,
            max_width=owner.rgb_width,
            max_height=owner.rgb_height,
            output_color=owner.decoder_output_color,
        )

        logger("Creating PyNvVideoCodec demuxer/decoder for left stereo stream...")
        self._create_stream_decoder(
            self.left,
            max_width=owner.stereo_width,
            max_height=owner.stereo_height,
            output_color=owner.stereo_decoder_output_color,
        )

        logger("Creating PyNvVideoCodec demuxer/decoder for right stereo stream...")
        self._create_stream_decoder(
            self.right,
            max_width=owner.stereo_width,
            max_height=owner.stereo_height,
            output_color=owner.stereo_decoder_output_color,
        )

        for stream in (self.left, self.right):
            stream.decode_thread = threading.Thread(
                target=self._stereo_decode_loop,
                args=(stream,),
                name=f"stereo-{stream.name}-decode",
                daemon=True,
            )
            stream.decode_thread.start()

        logger("DepthAI RGB + H26x stereo bottom-pack tensor pipeline v4 ready.")

    def _create_stream_decoder(self, stream: _EncodedStreamRuntime, max_width: int, max_height: int, output_color: str):
        owner = self.owner
        stream.demuxer = nvc.CreateDemuxer(stream.feeder.feed_chunk)
        kwargs = {
            "gpuid": owner.gpu_id,
            "codec": stream.demuxer.GetNvCodecId(),
            "usedevicememory": True,
            "maxwidth": int(max_width),
            "maxheight": int(max_height),
            "outputColorType": self._output_color_type(output_color),
        }
        if owner.low_latency:
            latency = self._get_low_latency_enum()
            if latency is not None:
                kwargs["latency"] = latency
            else:
                logger("Warning: PyNvVideoCodec low-latency enum not found.")
        stream.decoder = nvc.CreateDecoder(**kwargs)
        stream.packet_iter = iter(stream.demuxer)

    def _encoded_producer_loop(self, stream: _EncodedStreamRuntime):
        try:
            while not self.stop_event.is_set():
                try:
                    if hasattr(stream.depthai_q, "tryGet"):
                        pkt = stream.depthai_q.tryGet()
                        if pkt is None:
                            time.sleep(0.001)
                            continue
                    else:
                        pkt = stream.depthai_q.get()
                except Exception:
                    if self.stop_event.is_set():
                        break
                    raise

                data = self._packet_to_bytes(pkt)
                stream.packet_count += 1
                stream.byte_count += len(data)

                # Do not drop compressed packets during normal operation. Dropping
                # H26x bytes corrupts the stream until a later keyframe.
                while not self.stop_event.is_set():
                    try:
                        stream.bitstream_q.put(data, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        except Exception:
            if not self.stop_event.is_set() and not self._released:
                logger(f"DepthAI {stream.name} producer thread failed:")
                traceback.print_exc()
        finally:
            stream.push_eof()

    def _retain_decoded_frame_ref(self, stream: _EncodedStreamRuntime, frame):
        stream.decoded_frame_refs.append(frame)
        max_refs = max(1, int(getattr(self.owner, "retain_decoded_frame_refs", 16)))
        if len(stream.decoded_frame_refs) > max_refs:
            stream.decoded_frame_refs = stream.decoded_frame_refs[-max_refs:]

    def _decode_next_frame_tensor(self, stream: _EncodedStreamRuntime) -> torch.Tensor:
        while not self.stop_event.is_set():
            if stream.pending_decoded_frames:
                frame = stream.pending_decoded_frames.pop(0)
                tensor = torch.from_dlpack(frame)
                self._retain_decoded_frame_ref(stream, frame)
                stream.decoded_frame_count += 1
                return tensor

            if stream.packet_iter is None or stream.decoder is None:
                raise StopIteration

            try:
                packet = next(stream.packet_iter)
            except StopIteration:
                raise
            except Exception:
                if self.stop_event.is_set() or self._released:
                    raise StopIteration
                raise

            stream.demux_packet_count += 1

            if self.owner.low_latency:
                self._set_end_of_picture(packet)

            try:
                frames = stream.decoder.Decode(packet)
            except Exception:
                if self.stop_event.is_set() or self._released:
                    raise StopIteration
                raise

            for frame in frames:
                stream.pending_decoded_frames.append(frame)

        raise StopIteration

    def _stereo_decode_loop(self, stream: _EncodedStreamRuntime):
        try:
            while not self.stop_event.is_set():
                tensor = self._decode_next_frame_tensor(stream)
                gray = self._stereo_decoded_tensor_to_gray(tensor)

                with self._latest_stereo_lock:
                    stream.latest_tensor = gray
                    stream.latest_frame_index = stream.decoded_frame_count
                    stream.latest_at = time.monotonic()
                    if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                        self._stereo_ready.set()

        except StopIteration:
            pass
        except Exception:
            if not self.stop_event.is_set() and not self._released:
                logger(f"DepthAI {stream.name} stereo decode thread failed:")
                traceback.print_exc()
        finally:
            self._stereo_ready.set()

    def _stereo_decoded_tensor_to_gray(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a decoded stereo video frame to one clean [H, W] GPU tensor.

        v3 still could show horizontal stripes because this function treated every
        3-D tensor as CHW. On some PyNvVideoCodec builds, RGB/RGBP output can
        arrive as HWC. If HWC is read as CHW, tensor[0] is the first image row,
        not the first color channel, and the later resize/pack step produces
        striped stereo previews.

        v4 detects all common layouts explicitly:
            [3, H, W]       -> CHW
            [H, W, 3]       -> HWC
            [1, 3, H, W]    -> BCHW
            [1, H, W, 3]    -> BHWC
            [H, W]          -> gray/native luma
        """

        owner = self.owner
        stereo_h = int(owner.stereo_height)
        stereo_w = int(owner.stereo_width)
        output_color = str(getattr(owner, "stereo_decoder_output_color", "rgbp"))

        if getattr(owner, "debug_stereo_decoded_shape", False):
            # Print once per stream, not once total, so left/right can be compared.
            printed_attr = f"_debug_printed_{getattr(self, 'idx', 0)}_{output_color}_{id(tensor)}"
            stream_name = "unknown"
            # This function is called from the stereo decode thread. Use the
            # current thread name if available to make debugging less ambiguous.
            try:
                stream_name = threading.current_thread().name
            except Exception:
                pass
            if not getattr(self, f"_debug_printed_shape_{stream_name}", False):
                logger(
                    f"Decoded stereo tensor stream={stream_name}, "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                    f"stride={tuple(tensor.stride())}, color={output_color}"
                )
                setattr(self, f"_debug_printed_shape_{stream_name}", True)

        shape = tuple(tensor.shape)

        # Native/luma or already gray.
        if tensor.ndim == 2:
            gray = tensor

        elif tensor.ndim == 3:
            c_first = tensor.shape[0]
            c_last = tensor.shape[-1]

            # CHW, e.g. [3, H, W] or [1, H, W].
            if c_first in (1, 3, 4) and tensor.shape[-2:] == (stereo_h, stereo_w):
                gray = tensor[0]

            # HWC, e.g. [H, W, 3] or [H, W, 1]. This is the important v4 fix.
            elif c_last in (1, 3, 4) and tensor.shape[:2] == (stereo_h, stereo_w):
                gray = tensor[..., 0]

            # Native NV12-like fallback: take the visible luma rectangle.
            elif tensor.shape[-2] >= stereo_h and tensor.shape[-1] >= stereo_w:
                gray = tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])[0, :stereo_h, :stereo_w]

            elif tensor.shape[0] >= stereo_h and tensor.shape[1] >= stereo_w:
                gray = tensor[:stereo_h, :stereo_w, 0] if c_last in (1, 3, 4) else tensor[:stereo_h, :stereo_w]

            else:
                raise ValueError(
                    f"Cannot infer stereo decoded layout from shape={shape}; "
                    f"expected CHW/HWC/native for {(stereo_h, stereo_w)}."
                )

        elif tensor.ndim == 4:
            # BCHW, e.g. [1, 3, H, W].
            if tensor.shape[0] == 1 and tensor.shape[1] in (1, 3, 4) and tensor.shape[-2:] == (stereo_h, stereo_w):
                gray = tensor[0, 0]

            # BHWC, e.g. [1, H, W, 3].
            elif tensor.shape[0] == 1 and tensor.shape[-1] in (1, 3, 4) and tensor.shape[1:3] == (stereo_h, stereo_w):
                gray = tensor[0, ..., 0]

            # Generic fallback for unexpected leading dimensions.
            else:
                flat = tensor.reshape(-1, *tensor.shape[-2:])
                if flat.shape[-2] >= stereo_h and flat.shape[-1] >= stereo_w:
                    gray = flat[0, :stereo_h, :stereo_w]
                else:
                    raise ValueError(
                        f"Cannot infer stereo decoded layout from shape={shape}; "
                        f"expected BCHW/BHWC/native for {(stereo_h, stereo_w)}."
                    )
        else:
            raise ValueError(f"Cannot extract gray stereo frame from tensor shape {shape}")

        if tuple(gray.shape[-2:]) != (stereo_h, stereo_w):
            if owner.strict_stereo_shape:
                raise ValueError(
                    f"Decoded stereo frame shape {tuple(gray.shape)} from tensor shape {shape} "
                    f"does not match ({owner.stereo_height}, {owner.stereo_width}). "
                    "Set debug_stereo_decoded_shape=True to print the actual decoder layout."
                )
            gray = torch.nn.functional.interpolate(
                gray.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
                size=(stereo_h, stereo_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if not owner.normalize_stereo:
                gray = gray.round().clamp(0, 255).to(dtype=torch.uint8)

        return gray.contiguous()

    def _get_latest_stereo_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
            return self.left.latest_tensor, self.right.latest_tensor

        timeout = float(getattr(self.owner, "stereo_startup_timeout_sec", 2.0))
        if timeout > 0:
            self._stereo_ready.wait(timeout=timeout)

        with self._latest_stereo_lock:
            if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                return self.left.latest_tensor, self.right.latest_tensor

        if getattr(self.owner, "allow_missing_stereo", False):
            device = self._torch_device()
            zero = torch.zeros(
                (int(self.owner.stereo_height), int(self.owner.stereo_width)),
                dtype=torch.uint8,
                device=device,
            )
            return zero, zero

        while not self.stop_event.is_set() and not self._released:
            self._stereo_ready.wait(timeout=0.01)
            with self._latest_stereo_lock:
                if self.left.latest_tensor is not None and self.right.latest_tensor is not None:
                    return self.left.latest_tensor, self.right.latest_tensor

        raise StopIteration

    def _torch_device(self):
        owner = self.owner
        if owner.torch_device:
            return torch.device(owner.torch_device)
        if torch.cuda.is_available() and owner.gpu_id is not None and owner.gpu_id >= 0:
            return torch.device(f"cuda:{owner.gpu_id}")
        return torch.device("cpu")

    def _decoded_rgb_frame_to_tensor(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        self.owner.on_rgb_tensor(frame_tensor, self.rgb.decoded_frame_count)

        if self.owner.show_rgb_preview:
            self._show_small_rgb_preview(frame_tensor)

        tensor = frame_tensor.unsqueeze(0)
        if self.owner.normalize_rgb:
            tensor = tensor / 255.0
        return tensor

    def _decode_next_rgb_tensor(self):
        tensor = self._decode_next_frame_tensor(self.rgb)
        return self._decoded_rgb_frame_to_tensor(tensor)

    def _show_small_rgb_preview(self, tensor: torch.Tensor):
        owner = self.owner
        stride = max(1, int(owner.preview_stride))
        if tensor.ndim != 3:
            logger(f"Cannot preview RGB tensor shape: {tuple(tensor.shape)}")
            return
        if tensor.shape[0] == 3:
            small_hwc = tensor[:, ::stride, ::stride].permute(1, 2, 0).contiguous()
        elif tensor.shape[-1] == 3:
            small_hwc = tensor[::stride, ::stride, :].contiguous()
        else:
            logger(f"Cannot preview RGB tensor shape: {tuple(tensor.shape)}")
            return
        small_rgb = small_hwc.detach().cpu().numpy()
        if small_rgb.dtype != np.uint8:
            small_rgb = np.clip(small_rgb, 0, 255).astype(np.uint8)
        small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(owner.rgb_window_name, small_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _show_small_stereo_preview(self, left_gray: torch.Tensor, right_gray: torch.Tensor):
        owner = self.owner
        stride = max(1, int(owner.preview_stride))
        left = left_gray[::stride, ::stride]
        right = right_gray[::stride, ::stride]
        preview = torch.cat((left, right), dim=1).detach()
        if owner.normalize_stereo and preview.dtype.is_floating_point:
            preview = preview.mul(255.0)
        if preview.dtype != torch.uint8:
            preview = preview.clamp(0, 255).to(dtype=torch.uint8)
        cv2.imshow(owner.stereo_window_name, preview.cpu().numpy())
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _rgb_tensor_to_bchw3(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        if rgb_tensor.ndim != 4:
            raise ValueError(f"Expected RGB tensor with 4 dims, got {tuple(rgb_tensor.shape)}")
        if rgb_tensor.shape[1] == 3:
            return rgb_tensor.contiguous()
        if rgb_tensor.shape[-1] == 3:
            return rgb_tensor.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "Bottom packed RGB+stereo output needs 3 RGB channels. "
            f"Got RGB tensor shape {tuple(rgb_tensor.shape)}. "
            "Use decoder_output_color='rgbp' or 'rgb'."
        )

    def _prepare_stereo_flat_for_rgb(self, gray: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        flat = gray.reshape(-1)
        if flat.device != rgb.device or flat.dtype != rgb.dtype:
            flat = flat.to(device=rgb.device, dtype=rgb.dtype, non_blocking=self.owner.non_blocking_gpu_copy)
        if self.owner.normalize_stereo and flat.dtype.is_floating_point:
            # Stereo H26x decode produces 0..255 values. Match normalized RGB by
            # writing 0..1 payload values when requested.
            flat = flat / 255.0
        return flat

    def _pack_encoded_stereo_bottom(self, rgb_tensor: torch.Tensor, left_gray: torch.Tensor, right_gray: torch.Tensor):
        owner = self.owner
        rgb = self._rgb_tensor_to_bchw3(rgb_tensor)
        b, _, rgb_h, rgb_w = rgb.shape
        if b != 1:
            raise ValueError(f"Expected batch size 1, got {b}.")

        stereo_h = int(owner.stereo_height)
        stereo_w = int(owner.stereo_width)
        if tuple(left_gray.shape[-2:]) != (stereo_h, stereo_w):
            raise ValueError(f"Left stereo shape {tuple(left_gray.shape)} does not match {(stereo_h, stereo_w)}")
        if tuple(right_gray.shape[-2:]) != (stereo_h, stereo_w):
            raise ValueError(f"Right stereo shape {tuple(right_gray.shape)} does not match {(stereo_h, stereo_w)}")

        stereo_one_values = stereo_h * stereo_w
        stereo_flat_values = 2 * stereo_one_values
        values_per_bottom_row = 3 * int(rgb_w)
        payload_rows = (stereo_flat_values + values_per_bottom_row - 1) // values_per_bottom_row
        payload_start = int(rgb_h) - int(payload_rows)
        if payload_start < 0:
            raise ValueError(
                "Stereo payload does not fit inside RGB tensor bottom rows: "
                f"payload_rows={payload_rows}, rgb_height={rgb_h}."
            )

        payload = rgb[:, :, payload_start:, :]
        payload_flat = payload.reshape(b, -1)
        dst = payload_flat[0, :stereo_flat_values]

        left_flat = self._prepare_stereo_flat_for_rgb(left_gray, rgb)
        right_flat = self._prepare_stereo_flat_for_rgb(right_gray, rgb)

        dst[:stereo_one_values].copy_(left_flat[:stereo_one_values], non_blocking=True)
        dst[stereo_one_values:stereo_flat_values].copy_(right_flat[:stereo_one_values], non_blocking=True)

        if getattr(owner, "clear_unused_payload_tail", False):
            tail = payload_flat[0, stereo_flat_values:]
            if tail.numel() > 0:
                tail.fill_(float(owner.packed_stereo_pad_value))

        return rgb

    def next_frame(self):
        if self._released:
            raise StopIteration

        try:
            rgb_tensor = self._decode_next_rgb_tensor()
            left_gray, right_gray = self._get_latest_stereo_tensors()

            if self.owner.show_stereo_preview:
                self._show_small_stereo_preview(left_gray, right_gray)

            self.combined_frame_count += 1
            packed_tensor = self._pack_encoded_stereo_bottom(rgb_tensor, left_gray, right_gray)
            self.owner.on_rgb_stereo_tensor(packed_tensor, self.combined_frame_count)

            now = time.monotonic()
            if self.owner.log_fps and now - self.last_log_at >= 1.0:
                dt = max(now - self.last_log_at, 1e-6)
                combined_fps = (self.combined_frame_count - self.last_combined_count) / dt
                elapsed = max(now - self.started_at, 1e-6)
                rgb_mbps = self.rgb.byte_count * 8.0 / elapsed / 1_000_000
                left_mbps = self.left.byte_count * 8.0 / elapsed / 1_000_000
                right_mbps = self.right.byte_count * 8.0 / elapsed / 1_000_000
                logger(
                    f"combined={self.combined_frame_count}, "
                    f"fps={combined_fps:.2f}, "
                    f"rgb_dec={self.rgb.decoded_frame_count}, "
                    f"left_dec={self.left.decoded_frame_count}, "
                    f"right_dec={self.right.decoded_frame_count}, "
                    f"mbps rgb/left/right={rgb_mbps:.1f}/{left_mbps:.1f}/{right_mbps:.1f}, "
                    f"pack_rows={self.owner.packed_stereo_rows}"
                )
                self.last_log_at = now
                self.last_combined_count = self.combined_frame_count

            return packed_tensor

        except StopIteration:
            self.release()
            raise
        except Exception:
            if self._released or self.stop_event.is_set():
                raise StopIteration
            raise

    def release(self):
        if self._released:
            return

        self._released = True
        self.stop_event.set()
        self._stereo_ready.set()

        for stream in (self.rgb, self.left, self.right):
            stream.push_eof()
            try:
                if stream.depthai_q is not None and hasattr(stream.depthai_q, "close"):
                    stream.depthai_q.close()
            except Exception:
                pass

        for stream in (self.rgb, self.left, self.right):
            try:
                if stream.producer_thread is not None:
                    stream.producer_thread.join(timeout=2.0)
            except Exception as e:
                logger(f"Error joining {stream.name} producer thread: {e}")

        for stream in (self.left, self.right):
            try:
                if stream.decode_thread is not None:
                    stream.decode_thread.join(timeout=2.0)
            except Exception as e:
                logger(f"Error joining {stream.name} decode thread: {e}")

        try:
            if self.pipeline is not None:
                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self.pipeline.stop()
        except Exception as e:
            logger(f"Warning: DepthAI pipeline stop during release: {e}")

        try:
            self._exit_stack.close()
        except Exception:
            pass

        for stream in (self.rgb, self.left, self.right):
            stream.pending_decoded_frames.clear()
            stream.decoded_frame_refs.clear()
            stream.latest_tensor = None
            stream.packet_iter = None
            stream.decoder = None
            stream.demuxer = None
            stream.feeder = None
            stream.depthai_q = None

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self.device.close()
        except Exception:
            pass

        self.pipeline = None
        self.device = None

        for window_name in (self.owner.rgb_window_name, self.owner.stereo_window_name):
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass


class DepthAIPoeRGBStereoH26xBottomV4TorchGenerator(ImageMatGenerator):
    """
    ImageMatGenerator-style DepthAI PoE RGB + stereo generator.

    This v4 version encodes RGB, left mono, and right mono on the OAK device using
    H264/H265 before sending them over PoE. It is intended for the case where
    RGB-only H26x reaches about 15 FPS, but adding raw stereo throttles the
    pipeline.

    Output per source:
        one torch.Tensor with shape [1, 3, rgb_height, rgb_width]

    Layout:
        tensor[:, :, :stereo_payload_start_row, :] is normal RGB.
        tensor[:, :, stereo_payload_start_row:, :] contains flattened stereo:
            left.flatten(), then right.flatten().

    Important:
        The bottom packed_stereo_rows RGB rows are overwritten by stereo payload.
    """

    color_types: List['ColorType'] = []

    capture_fps: float = 15.0

    rgb_width: int = 4032
    rgb_height: int = 3040
    stereo_width: int = 1280
    stereo_height: int = 800

    rgb_camera_socket: Literal["CAM_A", "RGB"] = "CAM_A"
    left_camera_socket: Literal["CAM_B", "LEFT"] = "CAM_B"
    right_camera_socket: Literal["CAM_C", "RIGHT"] = "CAM_C"

    rgb_codec: Literal["h265", "h264"] = "h265"
    stereo_codec: Literal["h265", "h264"] = "h265"

    # Kept as compatibility aliases for earlier generated files.
    codec: Literal["h265", "h264"] = "h265"
    bitrate_kbps: int = 60000

    rgb_bitrate_kbps: int = 60000
    stereo_bitrate_kbps: int = 6000

    decoder_output_color: Literal["rgbp", "rgb", "native"] = "rgbp"
    stereo_decoder_output_color: Literal["rgbp", "rgb", "native"] = "rgbp"
    # VideoEncoder accepts NV12 or 8-bit gray. On some DepthAI v3 builds,
    # Camera.requestOutput(GRAY8) can arrive as a different internal type and
    # trigger: "Arrived frame type (...) is not either NV12 or YUV400p".
    # NV12 is the safest default; with stereo_decoder_output_color="rgbp"
    # v4 defaults to stereo_decoder_output_color="rgbp" to avoid native NV12 pitch/plane interpretation issues; payload is still grayscale because channel 0 is used.
    stereo_encoder_input_type: Literal["NV12", "YUV400p", "GRAY8", "RAW8"] = "NV12"

    rgb_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"
    stereo_resize_mode: Literal["CROP", "LETTERBOX", "STRETCH"] = "CROP"

    gpu_id: int = 0
    torch_device: Optional[str] = None
    non_blocking_gpu_copy: bool = True

    # Match your original RGB-only float32 output by default. Set both False if
    # you want raw uint8 output instead.
    normalize_rgb: bool = True
    normalize_stereo: bool = True
    strict_stereo_shape: bool = True
    debug_stereo_decoded_shape: bool = False

    packed_stereo_pad_value: float = 0.0
    clear_unused_payload_tail: bool = False

    rgb_depthai_queue_size: int = 8
    stereo_depthai_queue_size: int = 8
    rgb_bitstream_queue_size: int = 64
    stereo_bitstream_queue_size: int = 64

    stereo_startup_timeout_sec: float = 2.0
    allow_missing_stereo: bool = False

    low_latency: bool = False
    log_fps: bool = True
    retain_decoded_frame_refs: int = 16

    show_rgb_preview: bool = False
    show_stereo_preview: bool = False
    preview_stride: int = 10
    rgb_window_name: str = "DepthAI RGB small preview"
    stereo_window_name: str = "DepthAI encoded stereo small preview - left | right"

    # Let the camera/encoder control FPS.
    fps: int = 0

    def __init__(self, *args, **kwargs):
        # Compatibility: previous files used codec/bitrate_kbps. If the caller
        # supplies those but not the new explicit RGB fields, mirror them.
        if "codec" in kwargs and "rgb_codec" not in kwargs:
            kwargs["rgb_codec"] = kwargs["codec"]
        if "codec" in kwargs and "stereo_codec" not in kwargs:
            kwargs["stereo_codec"] = kwargs["codec"]
        if "bitrate_kbps" in kwargs and "rgb_bitrate_kbps" not in kwargs:
            kwargs["rgb_bitrate_kbps"] = kwargs["bitrate_kbps"]
        super().__init__(*args, **kwargs)

    @property
    def packed_stereo_rows(self) -> int:
        stereo_values = 2 * int(self.stereo_height) * int(self.stereo_width)
        values_per_row = 3 * int(self.rgb_width)
        return (stereo_values + values_per_row - 1) // values_per_row

    @property
    def stereo_payload_start_row(self) -> int:
        return int(self.rgb_height) - int(self.packed_stereo_rows)

    @property
    def rgb_valid_height(self) -> int:
        return int(self.stereo_payload_start_row)

    @property
    def packed_height(self) -> int:
        return int(self.rgb_height)

    def unpack_packed_tensor(self, packed: torch.Tensor):
        """
        Returns (rgb_with_payload, stereo, left, right).

        rgb_with_payload is the returned tensor itself; its bottom rows contain
        stereo payload rather than original RGB pixels.
        """

        if packed.ndim != 4 or packed.shape[1] != 3:
            raise ValueError(f"Expected tensor [B, 3, H, W], got {tuple(packed.shape)}.")

        start = int(self.stereo_payload_start_row)
        payload = packed[:, :, start:, :].reshape(packed.shape[0], -1)
        stereo_values = 2 * int(self.stereo_height) * int(self.stereo_width)
        stereo = payload[:, :stereo_values].reshape(
            packed.shape[0], 2, int(self.stereo_height), int(self.stereo_width)
        )
        left = stereo[:, 0]
        right = stereo[:, 1]
        return packed, stereo, left, right

    def _tensor_color_type(self):
        for name in ("RGBP", "RGB_CHW", "RGB", "BGR"):
            if hasattr(ColorType, name):
                return getattr(ColorType, name)
        return ColorType.BGR

    def on_rgb_tensor(self, tensor: torch.Tensor, frame_index: int):
        pass

    def on_rgb_stereo_tensor(self, tensor: torch.Tensor, frame_index: int):
        pass

    def create_frame_generator(self, idx, source):
        tensor_color_type = self._tensor_color_type()
        if idx >= len(self.color_types):
            self.color_types.append(tensor_color_type)
        else:
            self.color_types[idx] = tensor_color_type

        capture = self.register_resource(
            _DepthAIPoeRGBStereoH26xBottomTorchTensorCapture(
                owner=self,
                source=source,
                idx=idx,
            )
        )

        def gen(capture=capture):
            while True:
                try:
                    yield capture.next_frame()
                except StopIteration:
                    return
                except Exception:
                    if capture.stop_event.is_set() or capture._released:
                        return
                    logger("DepthAI RGB + H26x stereo bottom generator failed:")
                    traceback.print_exc()
                    raise

        return gen()


def test_rgb_stereo():
    """Small shape/unpack smoke test. This prints every frame and is not a benchmark."""

    gen = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_codec="h265",
        stereo_codec="h265",
        rgb_bitrate_kbps=60000,
        stereo_bitrate_kbps=6000,
        decoder_output_color="rgbp",
        stereo_decoder_output_color="rgbp",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        normalize_rgb=True,
        normalize_stereo=True,
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )

    try:
        for mats in gen:
            packed = mats[0].data()
            rgb, stereo, left, right = gen.unpack_packed_tensor(packed)
            print(
                "packed", tuple(packed.shape), packed.device, packed.dtype,
                "rgb", tuple(rgb.shape),
                "stereo", tuple(stereo.shape),
                "left", tuple(left.shape),
                "right", tuple(right.shape),
            )
    finally:
        gen.release()
        cv2.destroyAllWindows()

def to_small_cv(mat, s=10, rgb_to_bgr=True):
    """
    Accepts torch tensor in:
      CHW RGB: [3, H, W]
      HWC RGB: [H, W, 3]
      Gray:    [H, W]

    Returns uint8 CPU numpy image for cv2.imshow().
    """
    x = mat.detach()

    # CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)

    x = x[::s, ::s]

    if x.dtype.is_floating_point:
        # Works for normalized 0-1 tensors.
        # Also safe if values are already 0-255-ish.
        if float(x.max()) <= 1.5:
            x = x * 255.0
        x = x.clamp(0, 255).to(torch.uint8)
    else:
        x = x.to(torch.uint8)

    arr = x.cpu().numpy()

    # RGB -> BGR only for 3-channel images.
    if rgb_to_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, ::-1].copy()

    # If shape is [H, W, 1], make it [H, W]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return arr

def test_rgb_stereo():
    """Small shape/unpack smoke test. This prints every frame and is not a benchmark."""

    gen = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_codec="h265",
        stereo_codec="h265",
        rgb_bitrate_kbps=60000,
        stereo_bitrate_kbps=6000,
        decoder_output_color="rgbp",
        stereo_decoder_output_color="rgbp",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        normalize_rgb=True,
        normalize_stereo=True,
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )

    try:
        for mats in gen:
            packed = mats[0].data()
            rgb, stereo, left, right = gen.unpack_packed_tensor(packed)

            # Important for bottom-inplace packing:
            # bottom rows of rgb are overwritten by stereo payload.
            rgb_valid = rgb[:, :, :gen.stereo_payload_start_row, :]

            small_rgb = to_small_cv(rgb_valid[0], s=10, rgb_to_bgr=True)
            small_left = to_small_cv(left[0], s=4, rgb_to_bgr=False)
            small_right = to_small_cv(right[0], s=4, rgb_to_bgr=False)

            cv2.imshow("small_rgb", small_rgb)
            cv2.imshow("small_left", small_left)
            cv2.imshow("small_right", small_right)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # For FPS testing, do not print every frame.
            # Print every N frames instead.
            # print(
            #     "packed", tuple(packed.shape), packed.device, packed.dtype,
            #     "rgb", tuple(rgb.shape),
            #     "stereo", tuple(stereo.shape),
            #     "left", tuple(left.shape),
            #     "right", tuple(right.shape),
            # )

    finally:
        gen.release()
        cv2.destroyAllWindows()

def benchmark_rgb_stereo(duration_sec: float = 10.0, warmup_sec: float = 2.0):
    """No per-frame printing/unpacking benchmark."""

    gen = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_codec="h265",
        stereo_codec="h265",
        rgb_bitrate_kbps=60000,
        stereo_bitrate_kbps=6000,
        decoder_output_color="rgbp",
        stereo_decoder_output_color="rgbp",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        normalize_rgb=True,
        normalize_stereo=True,
        show_rgb_preview=False,
        show_stereo_preview=False,
        log_fps=False,
        fps=0,
    )

    total = 0
    measured = 0
    first_shape = None
    first_dtype = None
    first_device = None
    t0 = time.monotonic()
    measure_t0 = None

    try:
        for mats in gen:
            packed = mats[0].data()
            total += 1
            if first_shape is None:
                first_shape = tuple(packed.shape)
                first_dtype = packed.dtype
                first_device = packed.device
            now = time.monotonic()
            if measure_t0 is None and now - t0 >= warmup_sec:
                measure_t0 = now
                measured = 0
            if measure_t0 is not None:
                measured += 1
                if now - measure_t0 >= duration_sec:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = max(time.monotonic() - (measure_t0 or t0), 1e-6)
        fps = measured / elapsed if measure_t0 is not None else total / max(time.monotonic() - t0, 1e-6)
        print(
            f"benchmark frames={measured if measure_t0 is not None else total}, "
            f"fps={fps:.2f}, shape={first_shape}, dtype={first_dtype}, device={first_device}"
        )
    finally:
        gen.release()
        cv2.destroyAllWindows()


# Compatibility aliases.
DepthAIPoeRGBStereoH26xBottomV3TorchGenerator = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator
DepthAIPoeRGBStereoH26xBottomTorchGenerator = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator
DepthAIPoeRGBStereoEncodedBottomTorchGenerator = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator
DepthAIPoeRGBStereoH265BottomTorchGenerator = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator
DepthAIPoeRGBStereoPackedGenerator = DepthAIPoeRGBStereoH26xBottomV4TorchGenerator


if __name__ == "__main__":
    benchmark_rgb_stereo()
