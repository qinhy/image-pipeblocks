import json
import multiprocessing
import os
import sys
import glob
import time
import uuid
import platform
from enum import IntEnum
from typing import Iterator, List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .ImageMat import ColorType, ImageMat

logger = print

class ImageMatGenerator(BaseModel):
    sources: List[str]
    color_types: List['ColorType']
    uuid: str = ''
    shmIO_mode: Literal[False,'writer','reader'] = False
    fps:int = -1
    _min_frame_time:float = 0.0

    _resources: list = []
    _frame_generators: list = []
    ouput_mats:List[ImageMat] = []

    def model_post_init(self, context):
        self._min_frame_time = 1.0 / self.fps if self.fps != 0 else 0
        self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'        
        if len(self.sources)==0:raise ValueError('empty sources.')
        self._frame_generators = [self.create_frame_generator(i,src) for i,src in enumerate(self.sources)]

        if len(self._frame_generators)==0:raise ValueError('empty frame_generators.')
        if len(self.color_types)==0:raise ValueError('empty color_types.')
        if len(self.ouput_mats)==0:
            self.ouput_mats = [ImageMat(color_type=color_type).build(next(gen))
                        for gen,color_type in zip(self._frame_generators, self.color_types)]
            
        for mat in self.ouput_mats:
            mat.shmIO_mode=self.shmIO_mode
            if mat.shmIO_writer:
                mat.shmIO_writer.build_buffer()
            elif mat.shmIO_mode=='writer':
                mat.build_shmIO_writer()
        return super().model_post_init(context)        

    def register_resource(self, resource):
        self._resources.append(resource)
        return resource

    @staticmethod
    def has_func(obj, name):
        return callable(getattr(obj, name, None))

    def release_resources(self):
        cleanup_methods = [
            "exit", "end", "teardown",
            "stop", "shutdown", "terminate",
            "join", "cleanup", "deactivate",
            "release", "close", "disconnect",
            "destroy",
        ]

        for res in self._resources:
            for method in cleanup_methods:
                if self.has_func(res, method):
                    try:
                        getattr(res, method)()
                    except Exception as e:
                        logger(f"Error during {method} on {res}: {e}")

        self._resources.clear()

    def create_frame_generator(self, idx,source):
        raise NotImplementedError("Subclasses must implement `create_frame_generator`")

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        try:
            frames = [next(frame_gen) for frame_gen in self._frame_generators]
            if not frames or any(f is None for f in frames):
                raise StopIteration
            for frame, mat in zip(frames, self.ouput_mats):
                mat.unsafe_update_mat(frame)

            if self.fps:
                # Enforce FPS limit
                elapsed = time.time() - start_time
                sleep_time = self._min_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            return self.ouput_mats
        except StopIteration:
            raise StopIteration
    
    def reset_generators(self):
        self.release_resources()
        self._frame_generators = [self.create_frame_generator(i,src) for i,src in enumerate(self.sources)]

    def release(self):
        for i in self.ouput_mats:i.release()
        self.release_resources()

    def __del__(self):
        self.release()

    def __len__(self):
        return None

class CvVideoFrameGenerator(ImageMatGenerator):    
    color_types: List['ColorType'] = []
    
    def create_frame_generator(self, idx,source):
        if idx>=len(self.color_types):
            self.color_types.append(ColorType.BGR)
        else:
            self.color_types[0] = ColorType.BGR
        cap = self.register_resource(cv2.VideoCapture(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        def gen(cap=cap):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                yield np.ascontiguousarray(frame)
        return gen()
    
class XVSdkRGBDGenerator(ImageMatGenerator):
    class RGBResolution(IntEnum):
        RGB_1920x1080 = 0
        RGB_1280x720 = 1
        RGB_640x480 = 2
        RGB_320x240 = 3
        RGB_2560x1920 = 4
        RGB_3840x2160 = 5
    """
    Generator that produces synchronized [depth, rgb] ImageMat frames from an XVSDK RGB-D camera.
    """
    sources:List[str]=[]
    color_types:List[str]=["bgr", "bgr"]
    max_frames: Optional[int] = None

    def model_post_init(self, context):
        
        self.color_resolution = XVSdkRGBDGenerator.RGBResolution.RGB_1280x720
        self._frame_idx = 0
        self._running = False
        self.rgb_image: Optional[ImageMat] = None
        self.depth_image: Optional[ImageMat] = None

        import xvsdk  # Imported late to avoid runtime issues
        self.xvsdk = xvsdk
        self.register_resource(self.xvsdk)  # Use base class mechanism

        self._start_device()

        return super().model_post_init(context)

    def _start_device(self):
        """Starts RGB and TOF streams."""
        self.xvsdk.init()
        self.xvsdk.rgb_start()
        self.xvsdk.tof_start()
        self.xvsdk.xvisio_set_rgb_camera_resolution(self.color_resolution)
        self._running = True

    def release(self):
        """Stop the SDK streams."""
        if self._running:
            try:
                self.xvsdk.slam_stop()
                self.xvsdk.stop()
            except Exception:
                pass
            self._running = False

    def __iter__(self) -> Iterator[List[ImageMat]]:
        return self

    def __next__(self) -> List[ImageMat]:
        if not self._running or (self.max_frames is not None and self._frame_idx >= self.max_frames):
            self.release()
            raise StopIteration

        while True:
            rgb = self._get_rgb()
            depth = self._get_depth_as_uint8()
            if rgb is not None and depth is not None:
                break

        self._frame_idx += 1

        if self.rgb_image and self.depth_image:
            self.rgb_image.unsafe_update_mat(rgb)
            self.depth_image.unsafe_update_mat(depth)
        else:
            self.rgb_image = ImageMat(rgb, color_type=ColorType.BGR)
            self.depth_image = ImageMat(depth, color_type=ColorType.BGR)

        return [self.depth_image, self.rgb_image]

    def _get_rgb(self) -> Optional[np.ndarray]:
        """Retrieve and decode RGB frame."""
        try:
            w, h, _, ts, _, data, size = self.xvsdk.xvisio_get_rgb()
            if size.value > 0:
                yuv = np.frombuffer(data, dtype=np.uint8)
                height = int(h.value * 3 / 2)
                yuv = yuv[:w.value * height].reshape(height, w.value)
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_IYUV)
                return rgb[::-1, ::-1, :]
        except Exception as e:
            logger("RGB capture error:", e)
        return None

    def _get_depth_as_float32(self) -> Optional[np.ndarray]:
        """Retrieve raw float32 depth frame."""
        try:
            w, h, _, ts, data, size, _ = self.xvsdk.xvisio_get_tof()
            if size.value > 0:
                return np.frombuffer(data, dtype=np.float32).reshape(h.value, w.value)
        except Exception as e:
            logger("TOF capture error:", e)
        return None

    def _get_depth_as_uint8(self, depth_min: float = 0, depth_max: float = 10) -> Optional[np.ndarray]:
        """Convert raw depth to normalized uint8 colormap with values in green channel."""
        depth = self._get_depth_as_float32()
        if depth is None:
            return None
        depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).clip(0, 255).astype(np.uint8)
        colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        output = np.zeros_like(colormap)
        output[..., 0] = colormap[..., 0]  # Blue
        output[..., 1] = depth_norm        # Green = depth
        output[..., 2] = colormap[..., 2]  # Red
        return output

    def decode_depth_from_uint8(self, depth_colored: np.ndarray, depth_min: float = 0, depth_max: float = 10) -> np.ndarray:
        """Recover float32 depth from green channel of colorized depth image."""
        norm = depth_colored[..., 1].astype(np.float32)
        return norm / 255.0 * (depth_max - depth_min) + depth_min

class BitFlowFrameGenerator(ImageMatGenerator):
    class BitFlowCamera:
        """
        BitFlow Camera class for interfacing with a BitFlow frame grabber.
        """

        def __init__(self, video_src='bitflow-0', num_buffers=10):
            """
            Initializes the BitFlow Camera.

            Args:
                video_src (str): Video source identifier (e.g., 'bitflow-0').
                num_buffers (int): Number of image buffers to allocate.
            """
            self.video_src = video_src
            self.num_buffers = num_buffers
            self.CirAq, self.BufArray = None, None
            self.channel = int(self.video_src.split('-')[1]) if '-' in self.video_src else 0

            self._initialize_dlls()
            self._initialize_camera_params()
            self._initialize_camera()

        def _initialize_dlls(self):
            """Handles DLL setup for Windows, auto-searching for SDK paths."""
            if platform.system() != 'Windows' or sys.version_info < (3, 8):
                return

            sdk_paths = [r"C:\BitFlow SDK*", r"C:\Program Files\BitFlow SDK*"]
            serial_paths = [r"C:\Program Files\CameraLink\Serial", r"C:\Program Files (x86)\CameraLink\Serial"]

            sdk_path = next((os.path.join(path, "Bin64") for path in sorted(glob.glob(sdk_paths[0]), reverse=True) if os.path.exists(path)), None)
            serial_path = next((path for path in serial_paths if os.path.exists(path)), None)

            if sdk_path:
                os.add_dll_directory(sdk_path)
            else:
                logger(f"Warning: BitFlow SDK not found.")

            if serial_path:
                os.add_dll_directory(serial_path)
            else:
                logger(f"Warning: CameraLink Serial directory not found.")

        def _initialize_camera_params(self):
            from BFModule import BFGTLUtils as BFGTL

            """Initializes camera parameters."""
            dev = BFGTL.BFGTLDevice()
            try:
                dev.Open(self.channel)
                self.fps = self.get_camera_param(dev, 'AcquisitionFrameRate')
                self.width = self.get_camera_param(dev, 'Width')
                self.height = self.get_camera_param(dev, 'Height')
            except Exception as e:
                logger(f"Error reading parameter: {e}")
            finally:
                dev.Close()

        def _initialize_camera(self):
            from BFModule import BufferAcquisition as Buf

            """Initializes the BitFlow frame grabber and sets up acquisition."""
            try:
                self.CirAq = Buf.clsCircularAcquisition(Buf.ErrorMode.ErIgnore)
                self.CirAq.Open(self.channel)

                self.BufArray = self.CirAq.BufferSetup(self.num_buffers)
                self.CirAq.AqSetup(Buf.SetupOptions.setupDefault)
                self.CirAq.AqControl(Buf.AcqCommands.Start, Buf.AcqControlOptions.Wait)
            except Exception as e:
                logger(f"Error initializing BitFlow camera: {e}")
                self.close()
                raise


        def get_camera_param(self, dev, param_name):
            from BFModule import BFGTLUtils as BFGTL
            """
            Return the value of the requested parameter node.
            (Assumes 'dev' is already open and accessible.)
            """

            # Attempt to retrieve the node by name
            node = dev.getNode(param_name)
            
            # Check access permission
            if node.NodeAccess not in (BFGTL.BFGTLAccess.RO, BFGTL.BFGTLAccess.RW):
                raise RuntimeError(f"Node '{param_name}' is not readable.")

            # Check node type and return the appropriate value
            node_type = node.NodeType
            if node_type == BFGTL.BFGTLNodeType.Float:
                return BFGTL.FloatNode(node).FloatValue
            elif node_type == BFGTL.BFGTLNodeType.Integer:
                return BFGTL.IntegerNode(node).IntValue
            elif node_type == BFGTL.BFGTLNodeType.Boolean:
                return BFGTL.BooleanNode(node).BooleanValue
            elif node_type == BFGTL.BFGTLNodeType.String:
                return BFGTL.StringNode(node).StrValue
            elif node_type == BFGTL.BFGTLNodeType.Enumeration:
                return BFGTL.EnumerationNode(node).EntrySymbolic
            elif node_type == BFGTL.BFGTLNodeType.EnumEntry:
                enum_entry = BFGTL.EnumEntryNode(node)
                return (enum_entry.getValue, enum_entry.getSymbolic)
            else:
                raise TypeError(
                    f"Node '{param_name}' has an unhandled node type: {node_type}")
                    
        def read(self):
            """Reads the next frame from the BitFlow buffer."""
            if self.CirAq and self.CirAq.GetAcqStatus().Start:
                curBuf = self.CirAq.WaitForFrame(1000)
                return self.BufArray[curBuf.BufferNumber]
            return None

        def close(self):
            """Cleans up and closes the camera connection."""
            if self.CirAq:
                self.CirAq.AqCleanup()
                self.CirAq.BufferCleanup()
                self.CirAq.Close()
                self.CirAq = None

    """
    Frame generator for multiple BitFlow cameras.
    """
    sources:List[str] = ['bitflow-0']
    _resources:List['BitFlowFrameGenerator.BitFlowCamera'] = []
    _frame_generators: list = []    
    _mats:List[ImageMat] = []

    def create_frame_generator(self, idx,source):
        try:
            self.color_types = [ColorType.BAYER for _ in range(len(self.sources))]
            return self.register_resource(BitFlowFrameGenerator.BitFlowCamera(source))
        except Exception as e:
            self.release()
            raise e

class NumpyRawFrameFileGenerator(ImageMatGenerator):
    color_types: List['ColorType']
    def create_frame_generator(self, idx,source):
        arr = np.load(source)
        def gen(arr=arr):
            cnt=-1
            while True:
                # idx = np.random.choice(len(arr))
                cnt += 1
                yield np.ascontiguousarray(arr[cnt%len(arr)])
        return gen()    
          
class ImageMatGenerators(BaseModel):
    
    @staticmethod
    def dumps(gen:ImageMatGenerator):
        return json.dumps(gen.model_dump())
    
    @staticmethod
    def loads(gen_json:str)->ImageMatGenerator:
        gen = {
            'CvVideoFrameGenerator':CvVideoFrameGenerator,
            'XVSdkRGBDGenerator':XVSdkRGBDGenerator,
            'BitFlowFrameGenerator':BitFlowFrameGenerator,
        }
        g = json.loads(gen_json)
        return gen[f'{g["uuid"].split(":")[0]}'](**g) 

    @staticmethod
    def worker(gen_serialized):
        gen = ImageMatGenerators.loads(gen_serialized)
        for imgs in gen: pass
        
    @staticmethod
    def run_async(gen: 'ImageMatGenerator | str'):
        if isinstance(gen, str):
            gen_serialized = gen
        else:
            gen_serialized = gen.model_dump_json()

        p = multiprocessing.Process(target=ImageMatGenerators.worker, args=(gen_serialized,))
        p.start()
        return p