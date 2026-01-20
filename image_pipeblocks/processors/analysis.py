from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import cv2
import numpy as np
import torch

from ..ImageMat import ColorType, ImageMat, ImageMatInfo, ImageMatProcessor, ShapeType
from .utils import hex2rgba, logger

if TYPE_CHECKING:
    from .detection import YOLO


class NumpyImageMask(ImageMatProcessor):
    title: str = "numpy_image_mask"
    mask_image_path: str
    mask_color_hex: str = "#000000FF"
    mask_color: Union[Tuple[int, int, int], List[int]] = (0, 0, 0)
    mask_alpha: float = 0
    mask_split: Optional[Tuple[int, int]] = (2, 2)
    _original_masks: list = []
    _resized_masks: list = []
    _revert_masks: list = []

    def model_post_init(self, context: Any) -> None:
        self.mask_color = np.array(hex2rgba(self.mask_color_hex)[:3], dtype=np.uint8).tolist()
        return super().model_post_init(context)

    def reload_masks(self):
        self.load_masks(self.mask_image_path, self.mask_split)

    def load_masks(self, mask_image_path: Optional[str], mask_split: Tuple[int, int]):
        self._original_masks = []
        self._resized_masks = []
        self._revert_masks = []
        self._numpy_make_mask_images(mask_image_path, mask_split)
        [self._numpy_adjust_mask(i, img) for i, img in enumerate(self.input_mats)]

    def _numpy_make_mask_images(self, mask_image_path: Optional[str], mask_split: Tuple[int, int]):
        if mask_image_path is None:
            return None

        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
        if mask_image is None:
            raise ValueError(f"Unable to read mask image from {mask_image_path}")

        try:
            if mask_split:
                mask_images = [
                    y
                    for x in np.split(mask_image, mask_split[1], axis=0)
                    for y in np.split(x, mask_split[0], axis=1)
                ]
            else:
                mask_images = [mask_image]
        except ValueError:
            logger("Error: Invalid mask split configuration.")
            return None
        self._original_masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in mask_images]

    def _numpy_adjust_mask(self, idx, img: ImageMat):
        gray_mask: np.ndarray = self._original_masks[idx]

        shape_type = img.info.shape_type
        h, w = img.info.H, img.info.W
        c = img.info.C if shape_type == ShapeType.HWC else None

        resized_mask = cv2.resize(gray_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if c:
            resized_mask = np.expand_dims(resized_mask, axis=-1)
            if c > 1:
                resized_mask = resized_mask.repeat(c, axis=-1)
        if idx != len(self._resized_masks):
            raise ValueError('size checking error.')
        self._resized_masks.append(resized_mask)
        revert_mask = np.full_like(resized_mask, self.mask_color[::-1], dtype=resized_mask.dtype)
        revert_mask = cv2.bitwise_and(revert_mask, ~resized_mask)
        self._revert_masks.append(revert_mask)

    def validate_img(self, img_idx: int, img: ImageMat):
        img.require_ndarray()
        img.require_np_uint()
        img.require_HW_or_HWC()
        if self.mask_alpha > 0:
            img.require_BGR()

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        if len(self._resized_masks) == 0:
            self.load_masks(self.mask_image_path, self.mask_split)

        keep_pixel = [cv2.bitwise_and(img, self._resized_masks[i]) for i, img in enumerate(imgs_data)]
        if self.mask_alpha > 0:
            transparent_pixel = [
                cv2.bitwise_and(img, ~self._resized_masks[i]) for i, img in enumerate(imgs_data)
            ]

            result = [
                keep
                + (
                    (trans * (1 - self.mask_alpha)).astype(np.uint8)
                    + (bg * self.mask_alpha).astype(np.uint8)
                )
                for keep, trans, bg in zip(keep_pixel, transparent_pixel, self._revert_masks)
            ]
            return result
        else:
            return keep_pixel


class TorchImageMask(NumpyImageMask):
    title: str = "torch_image_mask"
    img_cnt: int = 0
    _torch_original_masks: list = []

    gpu: bool = True
    multi_gpu: int = -1
    _torch_dtype: ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()

    def model_post_init(self, context):
        self.num_devices = self.devices_info(gpu=self.gpu, multi_gpu=self.multi_gpu)
        return super().model_post_init(context)

    def validate_img(self, img_idx: int, img: ImageMat):
        img.require_torch_float()
        img.require_BCHW()
        if self.mask_alpha > 0:
            img.require_RGB()

    def load_masks(self, mask_image_path: Optional[str], mask_split: Tuple[int, int]):
        self._original_masks = []
        self._resized_masks = []
        self._revert_masks = []
        self._torch_original_masks = []
        self._numpy_make_mask_images(mask_image_path, mask_split)
        img_cnt = 0
        if self.input_mats:
            for i, img in enumerate(self.input_mats):
                h, w = img.info.H, img.info.W
                if img.info.B > 1:
                    img_masks = []
                    for j in range(img.info.B):
                        resized_mask_np = cv2.resize(
                            self._original_masks[img_cnt], (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        img_masks.append(resized_mask_np)
                        img_cnt += 1
                    self._resized_masks.append(
                        torch.as_tensor(np.asarray(img_masks))
                        .unsqueeze(0)
                        .permute(1, 0, 2, 3)
                        .to(img.info.device)
                        .type(ImageMatInfo.torch_img_dtype())
                        .div(255.0)
                    )
                else:
                    resized_mask_np = cv2.resize(
                        self._original_masks[img_cnt], (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    self._resized_masks.append(
                        torch.as_tensor(resized_mask_np)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(img.info.device)
                        .type(ImageMatInfo.torch_img_dtype())
                        .div(255.0)
                    )
                    img_cnt += 1

            for i, img in enumerate(self.input_mats):
                h, w = img.info.H, img.info.W
                revert_mask_torch = torch.tensor(self.mask_color, dtype=self._resized_masks[i].dtype).view(
                    1, 3, 1, 1
                )
                revert_mask_torch = (
                    revert_mask_torch.expand(1, 3, h, w)
                    .clone()
                    .to(img.info.device)
                    .type(ImageMatInfo.torch_img_dtype())
                    .div(255.0)
                )
                revert_mask_torch = revert_mask_torch * (1.0 - self._resized_masks[i])
                self._revert_masks.append(revert_mask_torch)

    def forward_raw(
        self,
        imgs_data: List[torch.Tensor],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[torch.Tensor]:
        if len(self._resized_masks) == 0:
            self.load_masks(self.mask_image_path, self.mask_split)

        keep_pixel = [image * self._resized_masks[i] for i, image in enumerate(imgs_data)]

        if self.mask_alpha > 0:
            transparent_pixel = [image * (1.0 - self._resized_masks[i]) for i, image in enumerate(imgs_data)]
            return [
                keep + ((trans * (1 - self.mask_alpha)) + (bg * self.mask_alpha))
                for keep, trans, bg in zip(keep_pixel, transparent_pixel, self._revert_masks)
            ]
        else:
            return keep_pixel


class TorchDebayer(ImageMatProcessor):
    class Debayer5x5(torch.nn.Module):
        class Layout(enum.Enum):
            RGGB = (0, 1, 1, 2)
            GRBG = (1, 0, 2, 1)
            GBRG = (1, 2, 0, 1)
            BGGR = (2, 1, 1, 0)

        def __init__(self, layout: Layout = Layout.RGGB):
            super().__init__()
            self.layout = layout
            self.kernels = torch.nn.Parameter(
                torch.tensor(
                    [
                        [0, 0, -2, 0, 0],
                        [0, 0, 4, 0, 0],
                        [-2, 4, 8, 4, -2],
                        [0, 0, 4, 0, 0],
                        [0, 0, -2, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, -2, 0, -2, 0],
                        [-2, 8, 10, 8, -2],
                        [0, -2, 0, -2, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, -2, 0, 0],
                        [0, -2, 8, -2, 0],
                        [1, 0, 10, 0, 1],
                        [0, -2, 8, -2, 0],
                        [0, 0, -2, 0, 0],
                        [0, 0, -3, 0, 0],
                        [0, 4, 0, 4, 0],
                        [-3, 0, 12, 0, -3],
                        [0, 4, 0, 4, 0],
                        [0, 0, -3, 0, 0],
                    ]
                ).view(4, 1, 5, 5).float()
                / 16.0,
                requires_grad=False,
            )

            self.index = torch.nn.Parameter(
                self._index_from_layout(layout),
                requires_grad=False,
            )

        def forward(self, x):
            B, C, H, W = x.shape

            xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
            planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
            planes = torch.cat((planes, x), 1)
            rgb = torch.gather(
                planes,
                1,
                self.index.repeat(
                    1,
                    1,
                    torch.div(H, 2, rounding_mode="floor"),
                    torch.div(W, 2, rounding_mode="floor"),
                ).expand(B, -1, -1, -1),
            )
            return torch.clamp(rgb, 0, 1)

        def _index_from_layout(self, layout: Layout = Layout) -> torch.Tensor:
            rggb = torch.tensor(
                [
                    [4, 1],
                    [2, 3],
                    [0, 4],
                    [4, 0],
                    [3, 2],
                    [1, 4],
                ]
            ).view(1, 3, 2, 2)
            return {
                layout.RGGB: rggb,
                layout.GRBG: torch.roll(rggb, 1, -1),
                layout.GBRG: torch.roll(rggb, 1, -2),
                layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
            }.get(layout)

    title: str = 'torch_debayer'
    _debayer_models: List['TorchDebayer.Debayer5x5'] = []
    _input_devices = []

    def validate_img(self, img_idx, img):
        img.require_torch_tensor()
        img.require_BCHW()
        img.require_BAYER()
        self._input_devices.append(img.info.device)
        model = TorchDebayer.Debayer5x5().to(img.info.device).to(img.info._dtype)
        self._debayer_models.append(model)

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[torch.Tensor],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[torch.Tensor]:
        debayered_imgs = []
        for i, img in enumerate(imgs_data):
            model = self._debayer_models[i % len(self._debayer_models)]
            debayered_imgs.append(model(img))
        return debayered_imgs


class MockTorchDebayer(ImageMatProcessor):
    title: str = 'mock_torch_debayer'
    _debayer_models: List['TorchDebayer.Debayer5x5'] = []
    _input_devices = []

    def validate_img(self, img_idx, img):
        img.require_torch_tensor()
        img.require_BCHW()
        img.require_BAYER()
        self._input_devices.append(img.info.device)

        def model(x: torch.Tensor) -> torch.Tensor:
            return x.repeat(1, 3, 1, 1)

        self._debayer_models.append(model)

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[torch.Tensor],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[torch.Tensor]:
        debayered_imgs = []
        for i, img in enumerate(imgs_data):
            img = img.repeat(1, 3, 1, 1)
            debayered_imgs.append(img)
        return debayered_imgs


class SlidingWindowSplitter(ImageMatProcessor):
    title: str = "sliding_window"
    stride: Optional[Tuple[int, int]] = None
    window_size: Tuple[int, int] = (1280, 1280)
    imgs_idx: Dict[int, list] = {}
    output_offsets_xyxy: List[List[Tuple[int, int, int, int]]] = []
    save_results_to_meta: bool = True

    def validate_img(self, img_idx: int, img: ImageMat):
        if self.stride is None:
            self.stride = self.window_size
        img.require_np_uint()
        img.require_HW_or_HWC()
        H, W = img.info.H, img.info.W
        wH, wW = self.window_size
        if wH > H or wW > W:
            raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
        out_mats: List[ImageMat] = []
        for i, v in self.imgs_idx.items():
            img = validated_imgs[i]
            out_mats += [img for _ in v]

        self.out_mats = [
            ImageMat(color_type=out_mats[i].info.color_type).build(img)
            for i, img in enumerate(converted_raw_imgs)
        ]
        return self.out_mats

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        out_imgs: List[np.ndarray] = []
        output_offsets_xyxy = []
        imgs_idx = {}

        for i, img in enumerate(imgs_data):
            H, W = img.shape[0], img.shape[1]
            wH, wW = self.window_size
            sH, sW = self.stride

            windows_list = []
            offsets_xyxy = []

            for row_start in range(0, H - wH + 1, sH):
                for col_start in range(0, W - wW + 1, sW):
                    window = img[row_start:row_start + wH, col_start:col_start + wW, :]
                    windows_list.append(window)
                    offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))

            image_mats = [w for w in windows_list]

            imgs_idx[i] = list(range(len(out_imgs), len(out_imgs) + len(image_mats)))
            out_imgs += image_mats
            output_offsets_xyxy.append(offsets_xyxy)

        self.imgs_idx = imgs_idx
        self.output_offsets_xyxy = output_offsets_xyxy
        return out_imgs

    def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo] = []):
        self.pixel_idx_forward_T = []
        self.pixel_idx_backward_T = []

        for offsets in self.output_offsets_xyxy:
            for offset in offsets:
                x1, y1, _, _ = offset

                transform_matrix = np.array(
                    [
                        [1.0, 0.0, -x1],
                        [0.0, 1.0, -y1],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

                self.pixel_idx_forward_T.append(transform_matrix.tolist())
                self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())


class SlidingWindowMerger(ImageMatProcessor):
    title: str = "sliding_window_merge"
    sliding_window_splitter_uuid: str = ''
    yolo_uuid: str = ''
    _sliding_window_splitter: 'SlidingWindowSplitter' = None

    def validate_img(self, img_idx: int, img: ImageMat):
        img.require_np_uint()
        img.require_HW_or_HWC()

    def forward_raw_yolo(self, sw_yolo_proc: 'YOLO'):
        detections_per_window = sw_yolo_proc.bounding_box_xyxy
        transforms = self._sliding_window_splitter.pixel_idx_backward_T

        merged_detections = [
            np.empty((0, 6), dtype=np.float32) for i in range(len(self._sliding_window_splitter.input_mats))
        ]

        if len(detections_per_window) != len(transforms):
            raise ValueError(
                f"Number of detections_per_window({len(detections_per_window)}) and transforms({len(transforms)}) must match."
            )

        transform_detections = [None] * len(transforms)

        for i, dets in enumerate(detections_per_window):
            T = transforms[i]
            transform_detections[i] = dets
            if len(dets) == 0:
                continue
            T = np.array(T)
            coords = dets[:, :4]
            ones = np.ones((coords.shape[0], 1))
            xy1 = np.concatenate([coords[:, :2], ones], axis=1)
            xy2 = np.concatenate([coords[:, 2:], ones], axis=1)

            xy1_trans = (T @ xy1.T).T
            xy2_trans = (T @ xy2.T).T
            new_boxes = np.concatenate([xy1_trans[:, :2], xy2_trans[:, :2], dets[:, 4:]], axis=1)
            transform_detections[i] = new_boxes

        for img_idx, info in enumerate(self._sliding_window_splitter.input_mats):
            tile_indices: list = self._sliding_window_splitter.imgs_idx[img_idx]
            merged_detections[img_idx] = np.vstack([transform_detections[i] for i in tile_indices])
        return merged_detections

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
        color_types = []
        for img_idx, info in enumerate(self._sliding_window_splitter.input_mats):
            offsets: List[Tuple[int, int, int, int]] = self._sliding_window_splitter.output_offsets_xyxy[img_idx]
            tile_indices: list = self._sliding_window_splitter.imgs_idx[img_idx]
            tile_idx = tile_indices[-1]
            color_types.append(validated_imgs[tile_idx].info.color_type)

        self.out_mats = [ImageMat(color_type=color_types[i]).build(img) for i, img in enumerate(converted_raw_imgs)]
        return self.out_mats

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        """Merge sliding window outputs back into full-size images."""
        self._sliding_window_splitter = meta[self.sliding_window_splitter_uuid]

        merged_imgs = [None] * len(self._sliding_window_splitter.input_mats)
        for img_idx, img in enumerate(self._sliding_window_splitter.input_mats):
            if self.out_mats and self.out_mats[img_idx].data() is not None:
                merged = self.out_mats[img_idx].data()
            else:
                H, W, channels = img.info.H, img.info.W, img.info.C
                if channels > 1:
                    merged = np.zeros((H, W, channels), dtype=np.uint8)
                else:
                    merged = np.zeros((H, W), dtype=np.uint8)

            offsets: List[Tuple[int, int, int, int]] = self._sliding_window_splitter.output_offsets_xyxy[img_idx]
            tile_indices: list = self._sliding_window_splitter.imgs_idx[img_idx]
            for i, tile_idx in enumerate(tile_indices):
                x1, y1, x2, y2 = offsets[i]
                merged[y1:y2, x1:x2] = imgs_data[tile_idx]
            merged_imgs[img_idx] = merged

        if self.yolo_uuid:
            yolo: 'YOLO' = meta[self.yolo_uuid]
            yolo.bounding_box_xyxy = self.forward_raw_yolo(yolo)

        return merged_imgs
