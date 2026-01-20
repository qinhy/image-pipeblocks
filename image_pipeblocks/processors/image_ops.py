from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..ImageMat import ColorType, ImageMat, ImageMatInfo, ImageMatProcessor


class CropImageToDivisibleByNum(ImageMatProcessor):
    num: int = 32
    title: str = 'crop_to_divisible_by_32'
    hs: List[int] = []
    ws: List[int] = []

    def model_post_init(self, context):
        self.title = self.title.replace('32', str(self.num))
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        img.require_np_uint()
        img.require_HW_or_HWC()
        h, w = img.info.H, img.info.W
        new_h = h - (h % self.num)
        new_w = w - (w % self.num)
        self.hs.append(new_h)
        self.ws.append(new_w)

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        processed_imgs = []
        for i, img in enumerate(imgs_data):
            h, w = self.hs[i], self.ws[i]
            processed_imgs.append(img[:h, :w])
        return processed_imgs


class GaussianBlur(ImageMatProcessor):
    title: str = 'gaussian_blur'
    ksize: int = 5
    sigma: float = 0

    def model_post_init(self, context):
        self.ksize = self.ksize if self.ksize % 2 == 1 else self.ksize + 1  # Ensure it's odd
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        img.require_np_uint()
        img.require_HW_or_HWC()
        h, w = img.info.H, img.info.W
        if h < self.ksize or w < self.ksize:
            raise ValueError(f"Image at index {img_idx} is smaller than the kernel size.")

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        processed_imgs = []
        for img in imgs_data:
            blurred = cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigma)
            processed_imgs.append(blurred)
        return processed_imgs


class CvDebayer(ImageMatProcessor):
    title: str = 'cv_debayer'
    format: int = cv2.COLOR_BAYER_BG2BGR

    def get_output_color_type(self):
        """Determine output color type based on the OpenCV conversion format."""
        bayer_to_color_map = {
            cv2.COLOR_BAYER_BG2BGR: ColorType.BGR,
            cv2.COLOR_BAYER_GB2BGR: ColorType.BGR,
            cv2.COLOR_BAYER_RG2BGR: ColorType.BGR,
            cv2.COLOR_BAYER_GR2BGR: ColorType.BGR,
            cv2.COLOR_BAYER_BG2RGB: ColorType.RGB,
            cv2.COLOR_BAYER_GB2RGB: ColorType.RGB,
            cv2.COLOR_BAYER_RG2RGB: ColorType.RGB,
            cv2.COLOR_BAYER_GR2RGB: ColorType.RGB,
            cv2.COLOR_BAYER_BG2GRAY: ColorType.GRAYSCALE,
            cv2.COLOR_BAYER_GB2GRAY: ColorType.GRAYSCALE,
            cv2.COLOR_BAYER_RG2GRAY: ColorType.GRAYSCALE,
            cv2.COLOR_BAYER_GR2GRAY: ColorType.GRAYSCALE,
        }
        return bayer_to_color_map.get(self.format, ColorType.UNKNOWN)

    def validate_img(self, img_idx, img):
        img.require_BAYER()
        img.require_np_uint()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
        color_type = self.get_output_color_type()
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        return [cv2.cvtColor(i, self.format) for i in imgs_data]


class NumpyRGBToNumpyBGR(ImageMatProcessor):
    title: str = 'numpy_rgb_to_bgr'

    def validate_img(self, img_idx, img):
        img.require_RGB()
        img.require_np_uint()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.BGR):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        return [img[:, :, [2, 1, 0]] for img in imgs_data]


class NumpyBGRToTorchRGB(ImageMatProcessor):
    title: str = 'numpy_bgr_to_torch_rgb'
    gpu: bool = True
    multi_gpu: int = -1
    _torch_dtype: ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()

    def model_post_init(self, context):
        self.num_devices = self.devices_info(gpu=self.gpu, multi_gpu=self.multi_gpu)
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        img.require_ndarray()
        img.require_HWC()
        img.require_BGR()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[torch.Tensor]:
        """
        Converts a batch of BGR images (NumPy) to RGB tensors (Torch).
        """
        torch_images = []
        for i, img in enumerate(imgs_data):
            device = self.num_devices[i % self.num_gpus]
            img_tensor = torch.as_tensor(img[:, :, [2, 1, 0]]).permute(2, 0, 1).contiguous()
            img_tensor = img_tensor.to(device, non_blocking=True).type(self._torch_dtype).div(255.0).unsqueeze(0)
            torch_images.append(img_tensor)
        return torch_images


class NumpyPadImage(ImageMatProcessor):
    """
    Pads an image using numpy's np.pad.
    Supports constant, edge, reflect, etc.
    """

    title: str = "numpy_pad_image"
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]] = ((10, 10), (10, 10))
    pad_value: int = 0
    mode: str = "constant"
    pad_widths: list = []

    def validate_img(self, img_idx: int, img: ImageMat):
        img.require_np_uint()
        img.require_HW_or_HWC()

        C = img.info.C
        if C == 2:
            pad_width = self.pad_width
        elif C == 3:
            pad_width = self.pad_width + ((0, 0),)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        self.pad_widths.append(pad_width)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        padded_imgs = []
        for i, img in enumerate(imgs_data):
            pad_width = self.pad_widths[i]
            padded_img = np.pad(img, pad_width, mode=self.mode, constant_values=self.pad_value)
            padded_imgs.append(padded_img)
        return padded_imgs

    def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo] = []):
        self.pixel_idx_forward_T = []
        self.pixel_idx_backward_T = []

        for info in imgs_info:
            transform_matrix = np.eye(3, dtype=np.float32)
            transform_matrix = np.eye(3, dtype=np.float32)

            pad_top, pad_bottom = self.pad_width[0]
            pad_left, pad_right = self.pad_width[1]

            T = np.array(
                [
                    [1, 0, pad_left],
                    [0, 1, pad_top],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            transform_matrix = T
            self.pixel_idx_forward_T.append(transform_matrix.tolist())
            self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())


class NumpyBayerToTorchBayer(ImageMatProcessor):
    title: str = 'numpy_bayer_to_torch_bayer'
    gpu: bool = True
    multi_gpu: int = -1
    _torch_dtype: ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()
    _tensor_models: list = []

    def model_post_init(self, context):
        self.num_devices = self.devices_info(gpu=self.gpu, multi_gpu=self.multi_gpu)
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        img.require_ndarray()
        img.require_HW()
        img.require_BAYER()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.BAYER):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[torch.Tensor]:
        """
        Converts a batch of Bayer images (NumPy) to Bayer tensors (Torch).
        """
        device_image_batches = defaultdict(list)

        for i, img in enumerate(imgs_data):
            device = self.num_devices[i % self.num_gpus]
            device_image_batches[device].append(img)

        torch_images = []
        for device, image_list in device_image_batches.items():
            batch_tensor = torch.stack([torch.as_tensor(img).unsqueeze(0) for img in image_list])
            batch_tensor = batch_tensor.to(device).type(self._torch_dtype).div(255.0)
            torch_images.append(batch_tensor)

        return torch_images


class NumpyGrayToTorchGray(NumpyBayerToTorchBayer):
    title: str = 'numpy_gray_to_torch_gray'
    gpu: bool = True
    multi_gpu: int = -1
    _torch_dtype: ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()
    _tensor_models: list = []

    def validate_img(self, img_idx, img):
        img.require_ndarray()
        img.require_HW()
        img.require_GRAYSCALE

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.GRAYSCALE):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)


class TorchRGBToNumpyBGR(ImageMatProcessor):
    title: str = 'torch_rgb_to_numpy_bgr'
    _numpy_dtype: Any = ImageMatInfo.numpy_img_dtype()
    _to_torch_dtype: Any = torch.uint8

    def validate_img(self, img_idx, img):
        img.require_torch_tensor()
        img.require_BCHW()
        img.require_RGB()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.BGR):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[torch.Tensor],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        bgr_images = []
        for img in imgs_data:
            img = img.permute(0, 2, 3, 1).mul(255.0).clamp(0, 255).to(self._to_torch_dtype)
            img = img.cpu().numpy()
            img = img[..., [2, 1, 0]]
            bgr_images += [i for i in img]
        return bgr_images


class TorchGrayToNumpyGray(ImageMatProcessor):
    title: str = 'torchgay_to_numpy_gray'
    _numpy_dtype: Any = ImageMatInfo.numpy_img_dtype()
    _to_torch_dtype: Any = torch.uint8

    def validate_img(self, img_idx, img):
        img.require_torch_tensor()
        img.require_BCHW()
        img.require_GRAYSCALE()

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.GRAYSCALE):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def forward_raw(
        self,
        imgs_data: List[torch.Tensor],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        bgr_images = []
        for img in imgs_data:
            img = img.permute(0, 2, 3, 1).mul(255.0).clamp(0, 255).to(self._to_torch_dtype)
            img = img.cpu().numpy()
            bgr_images += [i for i in img]
        return bgr_images


class TorchResize(ImageMatProcessor):
    title: str = 'torch_resize'
    target_size: Tuple[int, int]
    mode: str = "bilinear"

    def validate_img(self, img_idx, img):
        img.require_torch_tensor()
        img.require_BCHW()

    def forward_raw(
        self,
        imgs_data: List[torch.Tensor],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[torch.Tensor]:
        """
        Resizes a batch of PyTorch images to the target size.
        """
        resized_images = []
        for img in imgs_data:
            resized_img = F.interpolate(img, size=self.target_size, mode=self.mode)
            resized_images.append(resized_img)
        return resized_images

    def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo] = []):
        self.pixel_idx_forward_T = []
        self.pixel_idx_backward_T = []

        for info in imgs_info:
            for i in range(info.B):
                transform_matrix = np.eye(3, dtype=np.float32)

                scale_x = self.target_size[1] / info.W
                scale_y = self.target_size[0] / info.H

                T = np.array(
                    [
                        [scale_x, 0, 0],
                        [0, scale_y, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )

                transform_matrix = T
                self.pixel_idx_forward_T.append(transform_matrix.tolist())
                self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())


class CVResize(ImageMatProcessor):
    title: str = 'cv_resize'
    target_size: Tuple[int, int]
    interpolation: int = cv2.INTER_LINEAR

    def validate_img(self, img_idx, img):
        img.require_ndarray()
        img.require_HW_or_HWC()

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        """
        Resizes a batch of NumPy images using OpenCV.
        """
        resized_images = []
        for img in imgs_data:
            resized_img = cv2.resize(
                img,
                (self.target_size[1], self.target_size[0]),
                interpolation=self.interpolation,
            )
            resized_images.append(resized_img)
        return resized_images

    def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo] = []):
        self.pixel_idx_forward_T = []
        self.pixel_idx_backward_T = []

        for info in imgs_info:
            transform_matrix = np.eye(3, dtype=np.float32)

            scale_x = self.target_size[1] / info.W
            scale_y = self.target_size[0] / info.H

            T = np.array(
                [
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            transform_matrix = T
            self.pixel_idx_forward_T.append(transform_matrix.tolist())
            self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())
