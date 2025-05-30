from typing import Iterator, Optional, Dict, Any, List, Union
import cv2
from ImageMat import *
from shmIO import NumpyUInt8SharedMemoryStreamIO

class CvMultiVideoMatGenerator(ImageMatGenerator):
    """
    At each iteration, yields a list of ImageMat, one from each video.
    Loops videos when end is reached. Stops only at max_frames (if specified).
    """
    def __init__(
        self, 
        video_paths: List[str],
        color_type: Union[str, ColorType] = ColorType.BGR,
        scale: Optional[float] = None,
        step: int = 1,
        max_frames: Optional[int] = None,
    ):
        super().__init__(color_type=color_type)
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
            frames.append(ImageMat(frame, color_type=self.color_type))
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



