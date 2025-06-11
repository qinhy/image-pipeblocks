from typing import Iterator, Optional, Dict, Any, List, Union
from typing import Optional, Iterator, Tuple, Any
import numpy as np
import cv2
import cv2
from ImageMat import *
from shmIO import NumpyUInt8SharedMemoryStreamIO

import cv2
import numpy as np
from enum import IntEnum
from typing import Optional, Iterator

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
        for cap, color_mode in zip(resources, self.color_modes):
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if not ret:
                continue
            results.append(ImageMat(frame, color_mode))
        if not results:
            raise StopIteration
        return results

    def reset_generators(self):
        self.release_resources()
        self.source_generators = [self.create_source_generator(src
                                        ) for src in self.sources]


class XVSdkRGBDGenerator:
    class RGBresolution:
        RGB_1920x1080: int = 0
        RGB_1280x720: int = 1
        RGB_640x480: int = 2
        RGB_320x240: int = 3
        RGB_2560x1920: int = 4
        RGB_3840x2160: int = 5

    """
    Generator for synchronized RGB and Depth images from xvsdk camera.
    """

    def __init__(
        self, 
        color_resolution: int = 1,  # e.g., RGBresolution.RGB_1280x720
        max_frames: Optional[int] = None
    ) -> None:
        self.color_resolution: int = color_resolution
        self.max_frames: Optional[int] = max_frames
        self._running: bool = False
        self._frame_idx: int = 0

        import xvsdk
        self.xvsdk = xvsdk
        self.xvsdk.init()
        self.xvsdk.rgb_start()
        self.xvsdk.tof_start()
        self.xvsdk.xvisio_set_rgb_camera_resolution(self.color_resolution)
        self._running = True
        self._frame_idx = 0
        self.rgb_image:ImageMat = None
        self.depth_image:ImageMat = None

    def __del__(self):
        self.close()

    def __iter__(self) -> Iterator[list[ImageMat]]:
        return self

    def __next__(self) -> list[ImageMat]:
        if not self._running:
            raise StopIteration

        if self.max_frames is not None and self._frame_idx >= self.max_frames:
            self.close()
            raise StopIteration

        while True:
            rgb_image = self.get_rgb()
            depth_image = self.get_depth_as_uint8()
            if rgb_image is None or depth_image is None:
                continue
            break
            
        self._frame_idx += 1
        
        if self.rgb_image and self.depth_image:
            self.rgb_image.unsafe_update_mat(rgb_image)
            self.depth_image.unsafe_update_mat(depth_image)
        else:
            self.rgb_image = ImageMat(rgb_image,color_type=ColorType.BGR)
            self.depth_image = ImageMat(depth_image,color_type=ColorType.BGR)
        return [self.depth_image, self.rgb_image]

    def close(self) -> None:
        if self._running:
            try:
                self.xvsdk.slam_stop()
                self.xvsdk.stop()
            except Exception:
                pass
            self._running = False

    def get_rgb(self) -> Optional[np.ndarray]:
        """Retrieve and decode the latest RGB frame from the camera."""
        rgb_width, rgb_height, _, rgb_hostTimestamp, _, rgb_data, rgb_dataSize = self.xvsdk.xvisio_get_rgb()
        if rgb_dataSize.value > 0:
            rgb_raw = np.frombuffer(rgb_data, dtype=np.uint8)
            nheight = int(rgb_height.value * 3 / 2)
            rgb_raw = rgb_raw[:rgb_width.value * nheight].reshape(
                                            nheight, rgb_width.value)
            rgb_image = cv2.cvtColor(rgb_raw, cv2.COLOR_YUV2BGR_IYUV)
            rgb_image = rgb_image[::-1, ::-1, :]
            return rgb_image
        return None

    def get_depth_as_float32(self) -> Optional[np.ndarray]:
        """Retrieve the latest depth frame from the TOF sensor."""
        try:
            tof_width, tof_height, _, tof_hostTimestamp, tof_data, tof_dataSize, _ = self.xvsdk.xvisio_get_tof()
            if tof_dataSize.value > 0:
                depth_image = np.frombuffer(tof_data, dtype=np.float32
                                    ).reshape(tof_height.value, tof_width.value)
                return depth_image
        except Exception as e:
            print("TOF read error:", e)
        return None
    
    def get_depth_as_uint8(
        self, 
        depth_min: float = 0, 
        depth_max: float = 10
    ) -> Optional[np.ndarray]:
        """
        Normalize the depth image to 0-255 and apply a colormap,
        store normalized depth in the **green** channel.
        """
        depth_image = self.get_depth_as_float32()
        if depth_image is None:
            return None
        depth_norm = ((depth_image - depth_min) / (depth_max - depth_min) * 255
                      ).clip(0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        output = np.zeros_like(depth_colormap)
        output[..., 0] = depth_colormap[..., 0]  # Blue
        output[..., 1] = depth_norm              # Green channel for depth value
        output[..., 2] = depth_colormap[..., 2]  # Red
        return output

    def get_depth_from_uint8(
        self, 
        depth_color_image: np.ndarray, 
        depth_min: float = 0, 
        depth_max: float = 10
    ) -> np.ndarray:
        """
        Recover the depth (float32) from the depth colormap output by get_depth_as_uint8.
        Now assumes the **green** channel stores the normalized depth.
        """
        # Extract the green channel (depth values were stored here as uint8)
        depth_norm = depth_color_image[..., 1].astype(np.float32)
        # De-normalize back to [depth_min, depth_max]
        depth = depth_norm / 255.0 * (depth_max - depth_min) + depth_min
        return depth

class RGBResolution(IntEnum):
    RGB_1920x1080 = 0
    RGB_1280x720 = 1
    RGB_640x480 = 2
    RGB_320x240 = 3
    RGB_2560x1920 = 4
    RGB_3840x2160 = 5

class XVSdkRGBDGenerator(ImageMatGenerator):
    """
    Generator that produces synchronized [depth, rgb] ImageMat frames from an XVSDK RGB-D camera.
    """

    def __init__(self, color_resolution: int = RGBResolution.RGB_1280x720, max_frames: Optional[int] = None):
        self.color_resolution = color_resolution
        self.max_frames = max_frames
        self._frame_idx = 0
        self._running = False
        self.rgb_image: Optional[ImageMat] = None
        self.depth_image: Optional[ImageMat] = None

        import xvsdk  # Imported late to avoid runtime issues
        self.xvsdk = xvsdk
        self.register_resource(self.xvsdk)  # Use base class mechanism

        self._start_device()

        super().__init__(sources=[], color_modes=["bgr", "bgr"])

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
        stream_key = f'{self.stream_key_prefix}:{img_idx}'
        rd = NumpyUInt8SharedMemoryStreamIO.reader(stream_key, img.data().shape)
        rd.build_buffer()
        self.readers.append(rd)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        return [rd.read() for rd in self.readers]



