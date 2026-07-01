import json
import multiprocessing
import os
import sys
import glob
import time
from types import SimpleNamespace
import uuid
import platform
from enum import IntEnum
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..ImageMat import ColorType, ImageMat
from ..generator import ImageMatGenerator, ImageMatGenerators
from .utils import DepthAICamera

logger = print




class DaiCameraFrameGenerator(ImageMatGenerator):
    """DepthAI camera generator using DepthAICamera from dai_camera.py.

    Each source is interpreted as a DepthAI device_id. Use an empty string
    or "default" to open the default connected device. The generator yields
    one BGR preview/composite frame per source.
    """
    sources: List[str] = ['default'] # ["169.254.1.222"]
    color_types: List['ColorType'] = []

    # Required by DepthAICamera's config contract.
    width: int = 4000
    height: int = 3000
    queue_max_size: int = 1
    frame_poll_sleep_s: float = 0.001
    pipeline_name: str = "make_full_rgb_h264_pipeline"

    # Optional pipeline/config knobs consumed by dai_camera.py via getattr().
    rgb_bitrate_kbps: int = 40000
    mono_bitrate_kbps: int = 4000
    preview_width: Optional[int] = 960
    preview_height: Optional[int] = 720
    preview_stereo_height: Optional[int] = None
    rgb_encoder_pool_frames: int = 1
    mono_encoder_pool_frames: int = 1
    rgb_full_width: Optional[int] = None
    rgb_full_height: Optional[int] = None

    # read_frame() options.
    read_timeout_s: float = 1.0
    frame_index: int = 0
    stop_on_timeout: bool = False

    def _make_config(self, source: str):
        device_id = '' if source is None else str(source)
        return SimpleNamespace(
            width=self.width,
            height=self.height,
            fps=self.fps if self.fps and self.fps > 0 else 15,
            queue_max_size=self.queue_max_size,
            frame_poll_sleep_s=self.frame_poll_sleep_s,
            device_id=device_id,
            rgb_bitrate_kbps=self.rgb_bitrate_kbps,
            mono_bitrate_kbps=self.mono_bitrate_kbps,
            preview_width=self.preview_width,
            preview_height=self.preview_height,
            preview_stereo_height=self.preview_stereo_height,
            rgb_encoder_pool_frames=self.rgb_encoder_pool_frames,
            mono_encoder_pool_frames=self.mono_encoder_pool_frames,
            rgb_full_width=self.rgb_full_width,
            rgb_full_height=self.rgb_full_height,
        )

    def create_frame_generator(self, idx, source):
        if idx >= len(self.color_types):
            self.color_types.append(ColorType.BGR)
        else:
            self.color_types[idx] = ColorType.BGR

        try:

            cam = self.register_resource(DepthAICamera(self._make_config(source),
                                            pipeline_name=self.pipeline_name))
            cam.open()

            def gen(cam=cam):
                while cam.is_open:
                    frames = cam.read_frame(timeout_s=self.read_timeout_s)
                    if not frames:
                        if self.stop_on_timeout:
                            raise StopIteration('No DepthAI frame available before timeout.')
                        continue

                    # The realtime DepthAICamera path returns a single composed preview
                    # frame. Legacy paths may return multiple frames, so frame_index lets
                    # callers choose which one to expose as this generator's output.
                    try:
                        frame = frames[self.frame_index]
                    except IndexError:
                        frame = frames[0]

                    if frame is None:
                        continue
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    yield np.ascontiguousarray(frame)

            return gen()
        except Exception:
            self.release()
            raise



if __name__=="__main__":
    gen = DaiCameraFrameGenerator(
        sources=["169.254.1.222"],
        fps=15,
        width=4000,
        height=3000,
        pipeline_name="make_full_rgb_stereo_h264_plus_preview_pipeline",

        preview_width=960,
        preview_height=720,
        queue_max_size=1,
        frame_poll_sleep_s=0.001,
    )

    for mats in gen:
        for i, mat in enumerate(mats):
            img = mat.data()

            cv2.imshow(f"dai cam {i}", img)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()