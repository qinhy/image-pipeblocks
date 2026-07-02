import contextlib
import queue
import threading
import time
import traceback
from typing import List, Literal

import cv2
import depthai as dai
import numpy as np
import torch
import PyNvVideoCodec as nvc

from ..generator import ImageMatGenerator
from ..ImageMat import ColorType
from ..logger import logger

class _LiveBitstreamFeeder:
    """
    Feeds compressed OAK H264/H265 bytes into PyNvVideoCodec CreateDemuxer(callback).

    This is safer than manually creating nvc.PacketData.
    """

    def __init__(self, bitstream_queue, stop_event):
        self.q = bitstream_queue
        self.stop_event = stop_event
        self.pending = bytearray()
        self.eof = False
        self.total_bytes_fed = 0

    def feed_chunk(self, demuxer_buffer):
        # During shutdown, do not feed buffered partial GOP data to the demuxer.
        # Return EOF immediately so PyNvVideoCodec can unwind cleanly.
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


class _DepthAIPoeTorchTensorCapture:
    """
    Internal capture/decoder resource.

    Pipeline:
        DepthAI RGB camera
        -> OAK on-device H265/H264 encoder
        -> compressed packets over PoE
        -> PyNvVideoCodec demuxer
        -> PyNvVideoCodec NVDEC
        -> torch.from_dlpack(decoded_frame)
        -> full CUDA torch.Tensor
    """

    def __init__(self, owner, source: str, idx: int):
        self.owner = owner
        self.source = source
        self.idx = idx

        self.device = None
        self.pipeline = None
        self.depthai_q = None

        self.stop_event = threading.Event()
        self.bitstream_q = queue.Queue(maxsize=owner.bitstream_queue_size)
        self.producer_thread = None

        self.feeder = None
        self.demuxer = None
        self.decoder = None
        self.packet_iter = None

        self.pending_decoded_frames = []

        self.oak_packet_count = 0
        self.oak_byte_count = 0
        self.demux_packet_count = 0
        self.decoded_frame_count = 0

        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self.last_decoded_count = 0

        self._decoded_frame_refs = []
        self._released = False

        # Lets us use DepthAI v3's preferred Pipeline context-manager cleanup
        # while still keeping this object-oriented lifecycle.
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

        socket = getattr(dai.CameraBoardSocket, owner.camera_socket)
        cam = pipeline.create(dai.node.Camera).build(socket)

        rgb_nv12 = cam.requestOutput(
            (owner.width, owner.height),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            owner.capture_fps,
        )

        encoder = pipeline.create(dai.node.VideoEncoder).build(
            rgb_nv12,
            frameRate=owner.capture_fps,
            profile=self._depthai_profile(owner.codec),
        )

        try:
            encoder.setBitrateKbps(owner.bitrate_kbps)
        except Exception as e:
            logger(f"Warning: could not set OAK encoder bitrate: {e}")

        try:
            encoder.setKeyframeFrequency(int(owner.capture_fps))
        except Exception:
            pass

        q = encoder.out.createOutputQueue(
            maxSize=owner.depthai_queue_size,
            blocking=True,
        )

        pipeline.start()
        return pipeline, q

    def _producer_loop(self):
        """
        Reads compressed packets from DepthAI and pushes them to PyNvVideoCodec.

        Important:
            Do not drop H265/H264 packets if possible. Dropping compressed packets
            corrupts the stream until the next keyframe.
        """

        try:
            while not self.stop_event.is_set():
                try:
                    if hasattr(self.depthai_q, "tryGet"):
                        pkt = self.depthai_q.tryGet()
                        if pkt is None:
                            time.sleep(0.001)
                            continue
                    else:
                        pkt = self.depthai_q.get()
                except Exception:
                    if self.stop_event.is_set():
                        break
                    raise

                data = self._packet_to_bytes(pkt)

                self.oak_packet_count += 1
                self.oak_byte_count += len(data)

                while not self.stop_event.is_set():
                    try:
                        self.bitstream_q.put(data, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        except Exception:
            # Queue/pipeline methods may throw while we are intentionally closing.
            # Treat those as normal shutdown, not as producer failures.
            if not self.stop_event.is_set() and not self._released:
                logger("DepthAI producer thread failed:")
                traceback.print_exc()

        finally:
            self.stop_event.set()
            self._push_bitstream_eof()

    def _create_decoder(self):
        owner = self.owner

        kwargs = {
            "gpuid": owner.gpu_id,
            "codec": self.demuxer.GetNvCodecId(),
            "usedevicememory": True,
            "maxwidth": owner.width,
            "maxheight": owner.height,
            "outputColorType": self._output_color_type(owner.decoder_output_color),
        }

        if owner.low_latency:
            latency = self._get_low_latency_enum()
            if latency is not None:
                kwargs["latency"] = latency
            else:
                logger("Warning: PyNvVideoCodec low-latency enum not found.")

        return nvc.CreateDecoder(**kwargs)

    def _start(self):
        owner = self.owner

        if owner.codec in ("h264", "h265") and owner.width % 32 != 0:
            raise ValueError(
                "DepthAI H264/H265 encoder requires width multiple of 32. "
                "Use 4032 instead of 4056."
            )

        logger("Opening DepthAI device...")
        self.device = self._open_device()

        logger("Connected DepthAI device:")
        logger(f"  Device ID: {self.device.getDeviceInfo().getDeviceId()}")
        logger(f"  Cameras: {self.device.getConnectedCameras()}")
        logger("")
        logger("Starting DepthAI PoE torch tensor pipeline:")
        logger(f"  Source: {self.source}")
        logger(f"  Camera socket: {owner.camera_socket}")
        logger(f"  Size: {owner.width}x{owner.height}")
        logger(f"  OAK encoder: {owner.codec.upper()} @ {owner.capture_fps} FPS")
        logger(f"  Bitrate: {owner.bitrate_kbps} kbps")
        logger(f"  Decoder output: {owner.decoder_output_color}")
        logger(f"  GPU ID: {owner.gpu_id}")
        logger("")

        self.pipeline, self.depthai_q = self._create_depthai_pipeline()

        self.producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
        )
        self.producer_thread.start()

        self.feeder = _LiveBitstreamFeeder(self.bitstream_q, self.stop_event)

        logger("Creating PyNvVideoCodec demuxer...")
        self.demuxer = nvc.CreateDemuxer(self.feeder.feed_chunk)

        logger("Creating PyNvVideoCodec decoder...")
        self.decoder = self._create_decoder()
        self.packet_iter = iter(self.demuxer)

        logger("DepthAI tensor pipeline ready.")

    def _retain_decoded_frame_ref(self, frame):
        """
        Keep a few PyNvVideoCodec decoded frame objects alive.

        This helps avoid lifetime issues with DLPack/CUDA memory.
        """

        self._decoded_frame_refs.append(frame)

        max_refs = max(1, int(getattr(self.owner, "retain_decoded_frame_refs", 16)))
        if len(self._decoded_frame_refs) > max_refs:
            self._decoded_frame_refs = self._decoded_frame_refs[-max_refs:]

    def _show_small_preview_from_tensor(self, tensor: torch.Tensor):
        owner = self.owner

        stride = max(1, int(owner.preview_stride))

        if tensor.ndim != 3:
            logger(f"Cannot preview tensor shape: {tuple(tensor.shape)}")
            return

        # RGBP / CHW
        if tensor.shape[0] == 3:
            small = tensor[:, ::stride, ::stride]
            small_hwc = small.permute(1, 2, 0).contiguous()

        # RGB / HWC
        elif tensor.shape[-1] == 3:
            small_hwc = tensor[::stride, ::stride, :].contiguous()

        else:
            logger(f"Cannot preview tensor shape: {tuple(tensor.shape)}")
            return

        small_rgb = small_hwc.detach().cpu().numpy()
        small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(owner.window_name, small_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.release()
            raise StopIteration

    def _decoded_frame_to_tensor(self, frame):
        """
        Convert PyNvVideoCodec decoded frame to full GPU torch.Tensor.

        For decoder_output_color='rgbp':
            expected shape: [3, H, W]

        For decoder_output_color='rgb':
            expected shape: [H, W, 3]

        For 4032x3040 RGBP:
            [1, 3, 3040, 4032]
        """

        tensor = torch.from_dlpack(frame)

        self._retain_decoded_frame_ref(frame)
        self.decoded_frame_count += 1

        self.owner.on_decoded_tensor(tensor, self.decoded_frame_count)

        if self.owner.show_small_preview:
            self._show_small_preview_from_tensor(tensor)

        return tensor.unsqueeze(0)/255.0

    def next_frame(self):
        """
        Return one full decoded GPU torch.Tensor.
        """

        while not self.stop_event.is_set():
            if self.pending_decoded_frames:
                frame = self.pending_decoded_frames.pop(0)
                return self._decoded_frame_to_tensor(frame)

            if self.packet_iter is None or self.decoder is None:
                raise StopIteration

            try:
                packet = next(self.packet_iter)
            except StopIteration:
                self.release()
                raise
            except Exception:
                if self.stop_event.is_set() or self._released:
                    raise StopIteration
                raise

            self.demux_packet_count += 1

            if self.owner.low_latency:
                self._set_end_of_picture(packet)

            try:
                frames = self.decoder.Decode(packet)
            except Exception:
                if self.stop_event.is_set() or self._released:
                    raise StopIteration
                raise

            for frame in frames:
                self.pending_decoded_frames.append(frame)

            now = time.monotonic()
            if self.owner.log_fps and now - self.last_log_at >= 1.0:
                dt = max(now - self.last_log_at, 1e-6)
                dec_fps = (self.decoded_frame_count - self.last_decoded_count) / dt

                avg_mbps = (
                    self.oak_byte_count
                    * 8.0
                    / max(now - self.started_at, 1e-6)
                    / 1_000_000
                )

                logger(
                    f"decoded={self.decoded_frame_count}, "
                    f"dec_fps={dec_fps:.2f}, "
                    f"oak_packets={self.oak_packet_count}, "
                    f"oak_mbps={avg_mbps:.1f}, "
                    f"fed_mb={self.feeder.total_bytes_fed / 1_000_000:.1f}"
                )

                self.last_log_at = now
                self.last_decoded_count = self.decoded_frame_count

        raise StopIteration

    def _push_bitstream_eof(self):
        """Best-effort unblock for the demuxer callback."""
        try:
            self.bitstream_q.put_nowait(None)
            return
        except queue.Full:
            # Make one slot for EOF. Dropping a compressed packet is OK here because
            # we are shutting down and the feeder will report EOF.
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

    def release(self):
        if self._released:
            return

        self._released = True
        self.stop_event.set()
        self._push_bitstream_eof()

        # Stop host-side queue use before stopping the DepthAI pipeline/device.
        # This avoids producer-thread messages from tryGet()/queue internals during exit.
        try:
            if self.depthai_q is not None and hasattr(self.depthai_q, "close"):
                self.depthai_q.close()
        except Exception:
            pass

        try:
            if self.producer_thread is not None:
                self.producer_thread.join(timeout=2.0)
        except Exception as e:
            logger(f"Error joining producer thread: {e}")

        try:
            if self.pipeline is not None:
                if not hasattr(self.pipeline, "isRunning") or self.pipeline.isRunning():
                    self.pipeline.stop()
        except Exception as e:
            # By this point shutdown has already been requested, so do not make
            # benign DepthAI teardown races look like runtime failures.
            logger(f"Warning: DepthAI pipeline stop during release: {e}")

        try:
            self._exit_stack.close()
        except Exception:
            pass

        # Do not Flush() on forced release: the H264/H265 stream is intentionally
        # truncated, and draining can produce noisy decoder messages. Let objects
        # destruct after references are dropped.
        self.pending_decoded_frames.clear()
        self._decoded_frame_refs.clear()
        self.packet_iter = None
        self.decoder = None
        self.demuxer = None
        self.feeder = None

        try:
            if self.device is not None and hasattr(self.device, "close"):
                self.device.close()
        except Exception:
            pass

        self.depthai_q = None
        self.pipeline = None
        self.device = None

        if self.owner.show_small_preview:
            try:
                cv2.destroyWindow(self.owner.window_name)
            except Exception:
                pass


class DepthAIPoeRGBTorchGenerator(ImageMatGenerator):
    """
    ImageMatGenerator-style DepthAI PoE realtime tensor generator.

    Output:
        List[ImageMat]

    Each ImageMat receives:
        full GPU torch.Tensor

    Pipeline:
        OAK RGB CAM_A
        -> OAK H265/H264 encoder
        -> PyNvVideoCodec NVDEC
        -> torch.from_dlpack(decoded_frame)
        -> ImageMat.unsafe_update_mat(torch.Tensor)

    Recommended:
        width=4032
        height=3040
        capture_fps=15
        codec='h265'
        decoder_output_color='rgbp'
        fps=0
    """

    color_types: List['ColorType'] = []

    width: int = 4032
    height: int = 3040
    capture_fps: float = 15.0

    camera_socket: Literal["CAM_A", "CAM_B", "CAM_C"] = "CAM_A"

    codec: Literal["h265", "h264"] = "h265"
    bitrate_kbps: int = 60000

    gpu_id: int = 0

    # Use rgbp for torch processing.
    # Usually returns tensor shape [3, H, W].
    decoder_output_color: Literal["rgbp", "rgb", "native"] = "rgbp"

    depthai_queue_size: int = 8
    bitstream_queue_size: int = 64

    low_latency: bool = False
    log_fps: bool = True

    retain_decoded_frame_refs: int = 16

    # Optional small cv2 preview from the big GPU tensor.
    show_small_preview: bool = False
    preview_stride: int = 10
    window_name: str = "DepthAI full GPU tensor small preview"

    # Important:
    # Let the camera/encoder control FPS.
    # Do not add ImageMatGenerator sleep on top.
    fps: int = 0

    def _tensor_color_type(self):
        """
        Choose the best ColorType available in your enum.

        For rgbp, the tensor is usually CHW:
            [3, H, W]
        """

        if self.decoder_output_color == "rgbp":
            for name in ("RGBP", "RGB_CHW", "RGB"):
                if hasattr(ColorType, name):
                    return getattr(ColorType, name)

        if self.decoder_output_color == "rgb":
            if hasattr(ColorType, "RGB"):
                return ColorType.RGB

        if self.decoder_output_color == "native":
            for name in ("NV12", "YUV", "RGB"):
                if hasattr(ColorType, name):
                    return getattr(ColorType, name)

        return ColorType.BGR

    def on_decoded_tensor(self, tensor: torch.Tensor, frame_index: int):
        """
        Override this in a subclass for your real GPU processing.

        For decoder_output_color='rgbp':
            tensor shape is usually [3, 3040, 4032]

        For decoder_output_color='rgb':
            tensor shape is usually [3040, 4032, 3]
        """

        pass

    def create_frame_generator(self, idx, source):
        tensor_color_type = self._tensor_color_type()

        if idx >= len(self.color_types):
            self.color_types.append(tensor_color_type)
        else:
            self.color_types[idx] = tensor_color_type

        capture = self.register_resource(
            _DepthAIPoeTorchTensorCapture(
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
                    raise

        return gen()


def test1():
    gen = DepthAIPoeRGBTorchGenerator(
        sources=["169.254.1.222"],
        color_types=[],
        width=4032,
        height=3040,
        capture_fps=15,
        codec="h265",
        bitrate_kbps=60000,
        decoder_output_color="rgbp",
        show_small_preview=False,   # optional
        preview_stride=10,
        fps=0,
    )

    try:
        for mats in gen:
            mat = mats[0]
            tensor = mat.data()
            # print(tensor.shape, tensor.device, tensor.dtype)

            # Example GPU-side small view:
            small_gpu = (tensor[0].permute(1, 2, 0)[::10, ::10]*255.0
                         ).clone().detach().to(dtype=torch.uint8).cpu().numpy()[:,:,::-1]
            # print(small_gpu.shape, small_gpu.device, small_gpu.dtype)

            cv2.imshow("test", small_gpu)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        gen.release()
        cv2.destroyAllWindows()

