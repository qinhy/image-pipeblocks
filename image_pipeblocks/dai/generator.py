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
import torch

from ..ImageMat import ColorType, ImageMat
from ..generator import ImageMatGenerator, ImageMatGenerators
from .utils import DepthAICamera, DepthAINvH264Decoder

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
    decoder_for_full_resolution: Literal["pyav","nvdec",None] = None

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
            def gen(cam=cam, dec:Literal["pyav","nvdec",None]=self.decoder_for_full_resolution):
                if dec=="nvdec":
                    rgb_nvdec = DepthAINvH264Decoder(
                        gpuid=0,
                        use_device_memory=True,
                        output_color_type="RGB",
                        low_latency=True,
                        max_width=4096,
                        max_height=3072,
                    )
                    rgb_nvdec_func = rgb_nvdec.decode
                else:
                    rgb_nvdec_func = None

                while cam.is_open:
                    if rgb_nvdec_func is not None:
                        # Check full-res encoded packets.
                        encoded = cam.read_encoded_packets(max_packets=8)
                        for p in encoded["rgb"]:
                            frames = rgb_nvdec_func(p)
                            for f in frames:
                                f = torch.from_dlpack(f)
                                print("NVDEC RGB frame shape:", f.shape)
                                print("format:", getattr(f,"format",None))
                                print("dtype:", f.dtype)
                                print("device:", getattr(f,"device",None))
                                f = f.detach().cpu().numpy()
                                cv2.imwrite("test.jpg",f)

                    # for name, stream_packets in encoded.items():
                    #     if stream_packets:
                    #         p = stream_packets[-1]
                    #         print(
                    #             name,
                    #             "packets:", len(stream_packets),
                    #             "bytes:", len(p.getData()),
                    #             "seq:", p.getSequenceNum() if hasattr(p, "getSequenceNum") else None,
                    #         )

                    # Existing preview frame path.
                    frames = cam.read_frame(timeout_s=self.read_timeout_s)
                    if not frames:
                        if self.stop_on_timeout:
                            raise StopIteration("No DepthAI frame available before timeout.")
                        continue

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

def test1():
    gen = DaiCameraFrameGenerator(
        sources=["169.254.1.222"],
        fps=15,
        width=4000,
        height=3000,
        pipeline_name="make_full_rgb_stereo_h264_plus_preview_pipeline",
        decoder_for_full_resolution="nvdec",

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

    gen.release()

if __name__=="__main__":
    pass