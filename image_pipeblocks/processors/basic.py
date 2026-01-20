from __future__ import annotations

import json
import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from ..ImageMat import ColorType, ImageMat, ImageMatInfo, ImageMatProcessor


class DoingNothing(ImageMatProcessor):
    title: str = 'doing_nothing'

    def validate_img(self, img_idx, img):
        pass

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        return imgs_data


class BackUp(ImageMatProcessor):
    title: str = 'output_backup'
    device: str = ''
    save_results_to_meta: bool = True
    _backup_mats: List[ImageMat] = []

    def validate_img(self, idx, img: ImageMat):
        self.init_common_utility_methods(idx, img.is_ndarray())

    def get_backup_mats(self) -> List[ImageMat]:
        backup_mats = [
            ImageMat(color_type=inimg.info.color_type).build(img)
            for img, inimg in zip(self._backup_mats, self.input_mats)
        ]
        return backup_mats

    def forward_raw(
        self,
        imgs_data: List[Union[np.ndarray, torch.Tensor]],
        imgs_info: Optional[List[ImageMatInfo]] = None,
        meta: Optional[dict] = None,
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        self._backup_mats = []
        for i, img in enumerate(imgs_data):
            img = self._mat_funcs[i].copy_mat(img)
            if self.device == 'cpu' and isinstance(img, torch.Tensor):
                if img.dtype != np.uint8 or img.dtype != torch.uint8:
                    img = img * 255.0
                    img = self._mat_funcs[i].astype_uint8(img)
                img = self._mat_funcs[i].to_numpy(img)
                if img.shape[0] == 1:
                    img = img[0]
                if img.shape[-1] == 1:
                    img = img[..., 0]
            self._backup_mats.append(img)
        return imgs_data


class Lambda(ImageMatProcessor):
    title: str = 'lambda'
    config: dict = {}
    out_color_type: ColorType = ColorType.UNKNOWN
    _forward_raw: Callable = lambda imgs_data, imgs_info, meta: None

    def validate_img(self, img_idx, img):
        self.init_common_utility_methods(img_idx, img.is_ndarray())

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
        if self.out_color_type != ColorType.UNKNOWN:
            color_type = self.out_color_type
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        return self._forward_raw(imgs_data, imgs_info, meta)


GPS = None
try:
    from .gps import BaseGps, FileReplayGps, UsbGps

    class GPS(ImageMatProcessor):
        title: str = 'get_gps'
        port: str = 'gps.jsonl'
        save_results_to_meta: bool = True
        _gps: Optional[BaseGps] = None

        @staticmethod
        def coms():
            return BaseGps.coms()

        def change_port(self, port: str):
            self.off()
            self.port = port
            self.on()

        def get_state(self):
            if self._gps:
                return json.loads(self._gps.get_state().model_dump_json())
            else:
                return {}

        def get_latlon(self) -> List[float]:
            if self._gps:
                s = self._gps.get_state()
                return [s.lat, s.lon]
            else:
                return []

        def on(self):
            self.ini_gps()
            return super().on()

        def off(self):
            if self._gps:
                self._gps.close()
            del self._gps
            return super().off()

        def ini_gps(self):
            if os.path.isfile(self.port):
                self._gps = FileReplayGps()
            else:
                self._gps = UsbGps()
            self._gps.open(self.port)

        def validate_img(self, img_idx, img):
            if self._gps is None:
                self.ini_gps()

        def forward_raw(
            self,
            imgs_data: List[np.ndarray],
            imgs_info: List[ImageMatInfo] = [],
            meta={},
        ) -> List[np.ndarray]:
            return imgs_data
except Exception as e:
    print(e)
