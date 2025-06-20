import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from enum import IntEnum
import uuid

import cv2
import numpy as np
from pydantic import BaseModel

from ImageMat import ColorType, ImageMat, ImageMatGenerator
from shmIO import NumpyUInt8SharedMemoryStreamIO

class ImageMatGenerator(BaseModel):
    sources: list[str] = str
    color_types: list[ColorType] = []
    uuid:str = ''
    _resources = []  # General-purpose resource registry
    _source_generators = []  # General-purpose resource registry

    def model_post_init(self, context):
        self._source_generators = [self.create_source_generator(src) for src in self.sources]        
        self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'
        return super().model_post_init(context)

    def register_resource(self, resource):
        self._resources.append(resource)

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
                        print(f"Error during {method} on {res}: {e}")

        self._resources.clear()


    def create_source_generator(self, source):
        raise NotImplementedError("Subclasses must implement `create_source_generator`")

    def iterate_raw_images(self):
        for generator in self._source_generators:
            yield from generator

    def __iter__(self):
        return self

    def __next__(self):
        for raw_image, color_type in zip(self.iterate_raw_images(), self.color_types):
            yield ImageMat(color_type=color_type).build(raw_image)

    def reset_generators(self):
        pass

    def release(self):
        self.release_resources()

    def __del__(self):
        self.release()

    def __len__(self):
        return None

class VideoFrameGenerator(ImageMatGenerator):
    """
    Generator that uses cv2.VideoCapture to yield frames from video sources.
    """
    def create_source_generator(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.register_resource(cap)
        return cap  # Use the capture object directly

    def __next__(self):
        results = []
        resources:list[cv2.VideoCapture] = self._resources
        for cap, color_type in zip(resources, self.color_types):
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if not ret:
                continue
            results.append(ImageMat(color_type=color_type).build(frame))
        if not results:
            raise StopIteration
        return results

    def reset_generators(self):
        self.release_resources()
        self.source_generators = [self.create_source_generator(src
                                        ) for src in self.sources]

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
    sources:list[str]=[]
    color_types:list[str]=["bgr", "bgr"]
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

    def __iter__(self) -> Iterator[list[ImageMat]]:
        return self

    def __next__(self) -> list[ImageMat]:
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
            print("RGB capture error:", e)
        return None

    def _get_depth_as_float32(self) -> Optional[np.ndarray]:
        """Retrieve raw float32 depth frame."""
        try:
            w, h, _, ts, data, size, _ = self.xvsdk.xvisio_get_tof()
            if size.value > 0:
                return np.frombuffer(data, dtype=np.float32).reshape(h.value, w.value)
        except Exception as e:
            print("TOF capture error:", e)
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

class NumpyUInt8SharedMemoryReader(ImageMatGenerator):
    def __init__(self, stream_key_prefix: str, color_type, array_shapes=[]):
        super().__init__(color_type=color_type)
        self.readers:list[NumpyUInt8SharedMemoryStreamIO.StreamReader] = []
        self.stream_key_prefix = stream_key_prefix

    def validate_img(self, img_idx, img: ImageMat):
        img.require_ndarray()
        img.require_np_uint()
        stream_key = f"{self.stream_key_prefix}:{img_idx}"
        rd = NumpyUInt8SharedMemoryStreamIO.reader(stream_key, img.data().shape)
        rd.build_buffer()
        self.readers.append(rd)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        return [rd.read() for rd in self.readers]

class ImageMatGenerators(BaseModel):
    
    @staticmethod    
    def dumps(gen:ImageMatGenerator):
        return json.dumps(gen.model_dump())
    
    @staticmethod
    def loads(gen_json:str)->ImageMatGenerator:
        gen = {
            'VideoFrameGenerator':VideoFrameGenerator,
            'XVSdkRGBDGenerator':XVSdkRGBDGenerator,
            'NumpyUInt8SharedMemoryReader':NumpyUInt8SharedMemoryReader,
        }
        g = json.loads(gen_json)
        return gen[f'{g["uuid"].split(":")[0]}'](**g) 
    


    

