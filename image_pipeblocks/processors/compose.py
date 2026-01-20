from __future__ import annotations

import math
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from pydantic import BaseModel

from ..ImageMat import ColorType, ImageMat, ImageMatInfo, ImageMatProcessor


class TileNumpyImages(ImageMatProcessor):
    class Layout(BaseModel):
        tile_width: int
        tile_height: int

        col_widths: List[int]
        row_heights: List[int]

        total_width: int
        total_height: int

        channels: int
        _canvas: Any

    title: str = 'tile_numpy_images'
    tile_width: int
    layout: Optional[Layout] = None

    def _init_layout(self, imgs: List[ImageMat]):
        imgs_info = [i.info for i in imgs]
        num_images = len(imgs_info)
        if num_images == 0:
            raise ValueError("No input images info for doing tile.")
        tile_width = self.tile_width
        tile_height = math.ceil(num_images / tile_width)

        col_widths = [0] * tile_width
        row_heights = [0] * tile_height

        for idx, info in enumerate(imgs_info):
            row, col = divmod(idx, tile_width)
            h, w = info.H, info.W
            if w > col_widths[col]:
                col_widths[col] = w
            if h > row_heights[row]:
                row_heights[row] = h

        total_width = sum(col_widths)
        total_height = sum(row_heights)
        channels = imgs_info[0].C

        if channels == 1:
            canvas = np.zeros((total_height, total_width), dtype=imgs[0].data().dtype)
        else:
            canvas = np.zeros((total_height, total_width, channels), dtype=imgs[0].data().dtype)

        layout = TileNumpyImages.Layout(
            tile_width=tile_width,
            tile_height=tile_height,
            col_widths=col_widths,
            row_heights=row_heights,
            total_width=total_width,
            total_height=total_height,
            channels=channels,
        )
        layout._canvas = canvas
        return layout

    def validate_img(self, img_idx, img):
        img.require_np_uint()
        img.require_HWC()

    def validate(self, imgs, meta=...):
        color_types = {i.info.color_type for i in imgs}
        if len(color_types) != 1:
            raise ValueError(f"All images must have the same color_type, got {color_types}")
        super().validate(imgs, meta, run=False)
        self.layout = self._init_layout(self.input_mats)
        return self(self.input_mats, meta)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        layout = self.layout
        tile_width = layout.tile_width
        tile_height = layout.tile_height
        col_widths = layout.col_widths
        row_heights = layout.row_heights
        channels = layout.channels
        canvas = layout._canvas

        num_images = len(imgs_data)
        y_offset = 0
        for row in range(tile_height):
            x_offset = 0
            for col in range(tile_width):
                idx = row * tile_width + col
                if idx >= num_images:
                    break
                img: np.ndarray = imgs_data[idx]
                h, w = img.shape[:2]
                if channels == 1:
                    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
                else:
                    canvas[y_offset:y_offset + h, x_offset:x_offset + w, :channels] = img
                x_offset += col_widths[col]
            y_offset += row_heights[row]
        return [canvas]

    def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo] = []):
        if self.layout is None:
            raise ValueError("Layout not initialized. Call forward() first to build layout.")

        layout = self.layout
        tile_width = layout.tile_width
        col_widths = layout.col_widths
        row_heights = layout.row_heights

        x_prefix = [0]
        for w in col_widths[:-1]:
            x_prefix.append(x_prefix[-1] + w)

        y_prefix = [0]
        for h in row_heights[:-1]:
            y_prefix.append(y_prefix[-1] + h)

        self.pixel_idx_forward_T = []
        self.pixel_idx_backward_T = []

        for idx, _info in enumerate(imgs_info):
            row, col = divmod(idx, tile_width)
            x_off = x_prefix[col]
            y_off = y_prefix[row]

            T = np.array(
                [[1, 0, x_off], [0, 1, y_off], [0, 0, 1]],
                dtype=np.float32,
            )

            self.pixel_idx_forward_T.append(T.tolist())
            self.pixel_idx_backward_T.append(np.linalg.inv(T).tolist())

    def forward_transform_matrix(self, proc):
        res = super().forward_transform_matrix(proc)
        proc.bounding_box_xyxy = [np.vstack(proc.bounding_box_xyxy)]
        return res


class EncodeNumpyToJpeg(ImageMatProcessor):
    title: str = 'encode_numpy_to_jpeg'
    quality: int = 90

    def validate_img(self, img_idx, img):
        img.require_ndarray()
        img.require_HWC()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.JPEG):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        """
        Encodes a batch of NumPy images to JPEG format.
        """
        encoded_images = []
        for img in imgs_data:
            success, encoded = cv2.imencode(
                '.jpeg',
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)],
            )
            if not success:
                raise ValueError("JPEG encoding failed.")

        encoded_images.append(encoded)
        return encoded_images


class CvVideoRecorder(ImageMatProcessor):
    class VideoWriterWorker:
        def __init__(self, frame_interval=0.1, subset_s=100, queue_size=30, overlay_text: str = None):
            self.queue = queue.Queue(maxsize=queue_size)
            self.last_write_time = 0.0
            self.overlay_text = overlay_text
            self.frame_interval = frame_interval
            self.subset_s = subset_s
            self.thread = None

            self.writer = None
            self.writer_start_time = None
            self.file_counter = 0

        def writer_worker(self):
            while True:
                try:
                    frame = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame is None:
                    self.queue.task_done()
                    break

                now = time.time()
                if now - self.last_write_time < self.frame_interval:
                    self.queue.task_done()
                    continue

                if isinstance(frame, torch.Tensor):
                    frame = frame.permute(1, 2, 0).mul(255.0).clamp(0, 255).to(torch.uint8)
                    frame = frame.cpu().numpy()
                    frame = frame[..., [2, 1, 0]]

                if self.overlay_text:
                    frame = frame.copy()
                    cv2.putText(
                        frame,
                        self.overlay_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if self.writer is None or (now - self.writer_start_time) >= self.subset_s:
                    if self.writer is not None:
                        self.writer.release()

                    self.file_counter += 1
                    filename = f"{self.base_filename}-{self.file_counter:03d}{self.file_ext}"
                    fourcc = cv2.VideoWriter_fourcc(*self.codec)
                    self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.w, self.h))
                    self.writer_start_time = now

                try:
                    self.writer.write(frame)
                    self.last_write_time = now
                except Exception:
                    pass
                self.queue.task_done()

            if self.writer:
                self.writer.release()

        def build_writer(self, filename: str, codec: str, fps, w: int, h: int):
            """
            Save initial writer settings; actual writer is created dynamically per subset.
            """
            self.base_filename = os.path.splitext(filename)[0]
            self.file_ext = os.path.splitext(filename)[1]
            self.codec = codec
            self.fps = fps
            self.w = w
            self.h = h
            return self

        def start(self):
            self.thread = threading.Thread(target=self.writer_worker, daemon=True)
            self.thread.start()

        def stop(self):
            self.queue.put_nowait(None)

    title: str = "cv_recorder"
    output_filename: str = "output.avi"
    codec: str = "XVID"
    fps: int = 30
    overlay_text: Optional[str] = None
    total_imgs: int = 0
    WHs: List[Tuple[int, int]] = []
    frame_interval: float = 0.0
    subset_s: int = 20
    _workers: List['CvVideoRecorder.VideoWriterWorker'] = []

    def model_post_init(self, context):
        self.frame_interval = 1.0 / self.fps
        return super().model_post_init(context)

    def validate_img(self, img_idx, img: ImageMat):
        if img.is_ndarray():
            img.require_ndarray()
            img.require_np_uint()
            img.require_HW_or_HWC()
            self.total_imgs += 1
            self.WHs.append((img.info.W, img.info.H))
        if img.is_torch_tensor():
            img.require_torch_float()
            img.require_BCHW()
            self.total_imgs += img.info.B
            for i in range(img.info.B):
                self.WHs.append((img.info.W, img.info.H))

    def on(self):
        self.start()
        return super().on()

    def off(self):
        self.stop()
        return super().off()

    def format_filename(self, suffix: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(self.output_filename)
        res = f"{base}{suffix}_{ts}{ext}"
        return res.replace('_', '-')

    def start(self):
        self._workers = []
        for idx in range(self.total_imgs):
            filename = self.format_filename(f'_{idx}' if self.total_imgs > 1 else '')
            w, h = self.WHs[idx]
            worker = CvVideoRecorder.VideoWriterWorker(
                frame_interval=self.frame_interval,
                subset_s=self.subset_s,
                queue_size=30,
            ).build_writer(filename, self.codec, self.fps, w, h)
            self._workers.append(worker)
        for w in self._workers:
            w.start()

    def stop(self):
        for w in self._workers:
            w.stop()
        self._workers = []

    def _ensure_three_channels(self, x: Union[np.ndarray, torch.Tensor]):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = np.repeat(x[..., None], 3, axis=-1)
            elif x.ndim == 3:
                if x.shape[-1] == 1:
                    x = np.repeat(x, 3, axis=-1)
            return x
        if isinstance(x, torch.Tensor):
            if x.ndim == 4:
                if x.shape[1] in (1, 3):
                    if x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
            return x
        return x

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        cnt = 0
        if len(self._workers) == 0:
            self.start()

        for idx, frame in enumerate(imgs_data):
            frame = self._ensure_three_channels(frame)
            try:
                if isinstance(frame, np.ndarray):
                    self._workers[cnt].queue.put_nowait(frame)
                    cnt += 1
                if isinstance(frame, torch.Tensor):
                    for f in frame:
                        self._workers[cnt].queue.put_nowait(f)
                        cnt += 1
            except queue.Full:
                pass
        return imgs_data

    def release(self):
        res = super().release()
        self.stop()
        return res

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
