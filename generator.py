from typing import Iterator, Optional, Dict, Any, List, Union
from typing import Optional, Iterator, Tuple, Any
import numpy as np
import cv2
import cv2
from ImageMat import *
from shmIO import NumpyUInt8SharedMemoryStreamIO

class CvMultiVideoMatGenerator(ImageMatGenerator):
    """
    At each iteration, yields a list of ImageMat, one from each video.
    Loops videos when end is reached. Stops only at max_frames (if specified).
    """
    def __init__(self, 
        video_paths: List[str],
        scale: Optional[float] = None,
        step: int = 1,
        max_frames: Optional[int] = None,
    ):
        super().__init__()
        self.video_paths = list(video_paths)
        self.scale = scale
        self.step = step
        self.max_frames = max_frames
        self._caps = None
        self._frame_idx = 0

    def __iter__(self):
        # Open all video captures
        self._caps = [cv2.VideoCapture(p) for p in self.video_paths]
        for i, cap in enumerate(self._caps):
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_paths[i]}")
        self._frame_idx = 0
        return self

    def __next__(self) -> List['ImageMat']:
        if self.max_frames is not None and self._frame_idx >= self.max_frames:
            self._release()
            raise StopIteration
        frames = []
        for i, cap in enumerate(self._caps):
            frame = None
            for _ in range(self.step):
                ret, frame = cap.read()
                if not ret:
                    # Rewind to the beginning and try again
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        # If still cannot read, stop iteration
                        self._release()
                        raise StopIteration
            if self.scale is not None and frame is not None:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            frames.append(ImageMat(frame, color_type=ColorType.BGR))
        self._frame_idx += 1
        return frames

    def _release(self):
        if self._caps is not None:
            for cap in self._caps:
                try:
                    cap.release()
                except Exception:
                    pass
            self._caps = None

    def reset(self):
        self._release()
        self._caps = [cv2.VideoCapture(p) for p in self.video_paths]
        self._frame_idx = 0

    def __del__(self):
        self._release()


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
        
        rgb_image = ImageMat(rgb_image,color_type=ColorType.BGR)        
        depth_image = ImageMat(depth_image,color_type=ColorType.BGR)
        return [depth_image, rgb_image]

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


class NumpyUInt8SharedMemoryReader(ImageMatGenerator):
    def __init__(self, stream_key_prefix: str, color_type, array_shapes=[]):
        super().__init__(color_type=color_type)
        self.readers:list[NumpyUInt8SharedMemoryStreamIO.StreamReader] = []
        self.stream_key_prefix = stream_key_prefix

    def validate_img(self, img_idx, img: ImageMat):
        img.require_ndarray()
        img.require_np_uint()
        stream_key = f'{self.stream_key_prefix}:{i}'
        rd = NumpyUInt8SharedMemoryStreamIO.reader(stream_key, img.data().shape)
        rd.build_buffer()
        self.readers.append(rd)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        return [rd.read() for rd in self.readers]



