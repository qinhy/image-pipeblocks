# ===============================
# Standard Library Imports
# ===============================
from collections import defaultdict
import enum
import json
import math
from multiprocessing import shared_memory
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# ===============================
# Third-Party Library Imports
# ===============================
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pydantic import BaseModel, Field
from PIL import Image

# ===============================
# Ultralytics YOLO Imports
# ===============================
from ultralytics import YOLO
from ultralytics.utils import ops

# ===============================
# TensorRT & CUDA Imports
# ===============================
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ===============================
# Custom Modules
# ===============================
from shmIO import NumpyUInt8SharedMemoryStreamIO
from ImageMat import ColorType, ImageMat, ImageMatGenerator, ImageMatInfo, ImageMatProcessor, ShapeType


logger = print

def hex2rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA format."""
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    alpha = int(hex_color[6:8], 16) if len(hex_color) == 8 else 255
    return rgba + (alpha,)

class GeneralTensorRTInferenceModel:
    class CudaDeviceContext:
        def __init__(self, device_index):
            self.device = cuda.Device(device_index)
            self.context = None

        def __enter__(self):
            self.context = self.device.make_context()
            return self.context  # optional, in case you want to access it

        def __exit__(self, exc_type, exc_value, traceback):
            # Pop and destroy context on exit
            self.context.pop()
            # self.context.detach()

    np2torch_dtype = {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32: torch.int32,
    }

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem,name,shape,dtype,size):
            self.host = host_mem
            self.device = device_mem
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.size = size

        def __repr__(self):
            return ( f"{self.__class__.__name__}(host=0x{id(self.host):x}"
                     f",device=0x{id(self.device):x})"
                     f",name={self.name}"
                     f",shape={self.shape}"
                     f",dtype={self.dtype}"
                     f",size={self.size})")

    def __init__(self,engine_path, device, input_name='input', output_name='output'):
        self.engine_path=engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.input_shape = None
        self.output_shape = None
        self.dtype = None
        self.device = torch.device(device)
        # self.set_device(self.device.index)
        if self.device.index>0:
            with GeneralTensorRTInferenceModel.CudaDeviceContext(self.device.index):
                self.load_trt(engine_path,input_name,output_name)
                # self warmup
                self(torch.rand(*self.input_shape,device=self.device,dtype=self.np2torch_dtype[self.dtype]))
                self.load_trt(engine_path,input_name,output_name)
        else:
            self.load_trt(engine_path,input_name,output_name)
            # self warmup
            self(torch.rand(*self.input_shape,device=self.device,dtype=self.np2torch_dtype[self.dtype]))
            self.load_trt(engine_path,input_name,output_name)


    def load_trt(self, engine_path, input_name='input', output_name='output',verb=True):
        """Load a TensorRT engine file and prepare context and buffers."""
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_shape = [*self.engine.get_tensor_shape(input_name)]
        self.output_shape = [*self.engine.get_tensor_shape(output_name)]
        self.dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))

        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        if verb:
            print(f"[TensorRT] Loaded engine: {engine_path}, dtype: {self.dtype}")
            print(f"  Input shape: {self.input_shape}")
            print(f"  Output shape: {self.output_shape}")

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            hdm = self.HostDeviceMem(host_mem,device_mem,name,shape,dtype,size)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(hdm)
            else:
                outputs.append(hdm)
        return inputs, outputs, bindings

    def _transfer_torch2cuda(self, tensor: torch.Tensor, device_mem: HostDeviceMem):
        num_bytes = tensor.element_size() * tensor.nelement()
        cuda.memcpy_dtod_async(
            dest=int(device_mem.device),
            src=int(tensor.data_ptr()),
            size=num_bytes,
            stream=self.stream
        )

    def _transfer_cuda2torch(self, device_mem: HostDeviceMem):
        torch_dtype = self.np2torch_dtype[self.dtype]
        out_tensor = torch.empty(self.output_shape, device=self.device, dtype=torch_dtype)

        num_bytes = out_tensor.element_size() * out_tensor.nelement()
        cuda.memcpy_dtod_async(
            dest=int(out_tensor.data_ptr()),
            src=int(device_mem.device),
            size=num_bytes,
            stream=self.stream
        )
        return out_tensor

    def infer(self, inputs: list[torch.Tensor]):
        x = inputs[0]
        assert torch.is_tensor(x), "Input must be a torch.Tensor!"
        assert x.is_cuda, "Torch input must be on CUDA!"
        assert x.dtype == self.np2torch_dtype[self.dtype], f"Expected dtype {self.np2torch_dtype[self.dtype]}, got {x.dtype}"
        assert x.device == self.device, "Torch input must be on same CUDA!"
        return self.raw_infer(x)

    def raw_infer(self, x:torch.Tensor):
        [self._transfer_torch2cuda(x, mem) for x, mem in zip([x], self.inputs)]
        self.context.execute_v2(bindings=self.bindings)
        # outputs = [self._transfer_cuda2torch(mem) for mem in self.outputs]
        self.stream.synchronize()
        return []#outputs
        
    def __call__(self, input_data):
        outputs = self.infer([input_data])
        return outputs[0] if len(outputs) == 1 else outputs

class Processors:        
    class NumpyBGRToTorchRGB(ImageMatProcessor):
        title:str='numpy_bgr_to_torch_rgb'
        gpu:bool=True
        multi_gpu:int=-1
        _torch_dtype:ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()

        def model_post_init(self, context):
            self.num_devices = self.devices_info(gpu=self.gpu,multi_gpu=self.multi_gpu)
            return super().model_post_init(context)
        
        def validate_img(self, img_idx, img):            
            img.require_ndarray()
            img.require_HWC()
            img.require_BGR()
            
        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:
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

    class NumpyBayerToTorchBayer(ImageMatProcessor):
        # to BCHW
        title:str='numpy_bayer_to_torch_bayer'
        gpu:bool=True
        multi_gpu:int=-1
        _torch_dtype:ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()
        _tensor_models:list = []

        def model_post_init(self, context):
            self.num_devices = self.devices_info(gpu=self.gpu,multi_gpu=self.multi_gpu)
            return super().model_post_init(context)

        def validate_img(self, img_idx, img):
            img.require_ndarray()
            img.require_HW()
            img.require_BAYER()
            
        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.BAYER):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:            
            """
            Converts a batch of Bayer images (NumPy) to Bayer tensors (Torch).
            """
            # Create a dictionary to hold lists of images for each device
            device_image_batches = defaultdict(list)

            # Step 1: Organize images by device
            for i, img in enumerate(imgs_data):
                device = self.num_devices[i % self.num_gpus]
                device_image_batches[device].append(img)

            # Step 2: Create batched tensors per device
            torch_images = []
            for device, image_list in device_image_batches.items():
                # [B, C, H, W]
                batch_tensor = torch.stack([torch.as_tensor(img).unsqueeze(0) for img in image_list]) 
                batch_tensor = batch_tensor.to(device).type(self._torch_dtype).div(255.0)
                torch_images.append(batch_tensor)

            # torch_images = []
            # for i, img in enumerate(imgs_data):
            #     tensor_img = torch.as_tensor(img).to(self.num_devices[i % self.num_gpus]).type(self._torch_dtype
            #                               ).div(255.0).unsqueeze(0).unsqueeze(0)
            #     torch_images.append(tensor_img)
            return torch_images

    class TorchRGBToNumpyBGR(ImageMatProcessor):
        title:str='torch_rgb_to_numpy_bgr'
        _numpy_dtype:Any = ImageMatInfo.numpy_img_dtype()
        _to_torch_dtype:Any = torch.uint8

        def validate_img(self, img_idx, img):
            img.require_torch_tensor()
            img.require_BCHW()
            img.require_RGB()

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.BGR):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

        def forward_raw(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            bgr_images = []
            for img in imgs_data:
                img = img.permute(0, 2, 3, 1).mul(255.0).clamp(0, 255
                    ).to(self._to_torch_dtype)
                img = img.cpu().numpy()
                img = img[..., [2,1,0]]  # Convert RGB to BGR
                if len(img)>1:
                    bgr_images+=[i for i in img]
                else:
                    bgr_images.append(img)
            return bgr_images

    class TorchResize(ImageMatProcessor):
        title:str='torch_resize'
        target_size:Tuple[int, int]
        mode:str="bilinear"

        def validate_img(self, img_idx, img):
            img.require_torch_tensor()
            img.require_BCHW()

        def forward_raw(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:            
            """
            Resizes a batch of PyTorch images to the target size.
            """
            resized_images = []
            for img in imgs_data:
                resized_img = F.interpolate(img, size=self.target_size, mode=self.mode, align_corners=False)
                resized_images.append(resized_img)
            return resized_images

        def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo]=[]):
            self.pixel_idx_forward_T = []
            self.pixel_idx_backward_T = []

            for info in imgs_info:
                for i in range(info.B):
                    transform_matrix = np.eye(3, dtype=np.float32)  # Identity base
        
                    scale_x = self.target_size[1] / info.W
                    scale_y = self.target_size[0] / info.H

                    T = np.array([
                        [scale_x, 0,       0],
                        [0,       scale_y, 0],
                        [0,       0,       1]
                    ], dtype=np.float32)

                    transform_matrix = T
                    self.pixel_idx_forward_T.append(transform_matrix.tolist())
                    self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())
       
    class NumpyImageMask(ImageMatProcessor):
        title: str = "numpy_image_mask"
        mask_image_path: Optional[str] = None
        mask_color: str = "#00000080"
        mask_split: Tuple[int, int] = (2, 2)
        _masks:list = None

        def model_post_init(self, context: Any) -> None:
            self._masks = self._make_mask_images(self.mask_image_path, self.mask_split, self.mask_color)
            if self._masks is None:
                logger('[NumpyImageMask] Warning: no mask image loaded. This block will do nothing.')
            return super().model_post_init(context)

        def gray2rgba_mask_image(self, gray_mask_img: np.ndarray, hex_color: str) -> Image.Image:
            """Convert a grayscale mask to an RGBA image with the specified color."""
            select_color = np.array(hex2rgba(hex_color), dtype=np.uint8)
            background = np.array([255, 255, 255, 0], dtype=np.uint8)

            condition = gray_mask_img == 0
            condition = condition[..., None]
            color_mask_img = np.where(condition, select_color, background)

            return Image.fromarray(cv2.cvtColor(color_mask_img, cv2.COLOR_BGRA2RGBA))

        def _make_mask_images(self, mask_image_path: Optional[str], mask_split: Tuple[int, int], preview_color: str):
            if mask_image_path is None:
                return None

            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
            if mask_image is None:
                raise ValueError(f"Unable to read mask image from {mask_image_path}")

            # Split mask into sub-masks
            try:
                mask_images = [
                    sub_mask
                    for row in np.split(mask_image, mask_split[0], axis=0)
                    for sub_mask in np.split(row, mask_split[1], axis=1)
                ]
            except ValueError:
                logger("Error: Invalid mask split configuration.")
                return None

            # Convert to grayscale and apply preview color
            gray_masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in mask_images]
            preview_masks = [self.gray2rgba_mask_image(gray, preview_color) for gray in gray_masks]

            return {"original": gray_masks, "preview": preview_masks}

        def _adjust_mask(self, images: List[ImageMat]):
            self._masks["resized_masks"] = [
                None for _ in self._masks["original"]
            ]

            for i, img in enumerate(images):
                gray_mask: np.ndarray = self._masks["original"][i]

                shape_type = img.info.shape_type
                h, w = img.info.H, img.info.W
                c = img.info.C if shape_type == ShapeType.HWC else None

                # Resize the mask to match image dimensions
                resized_mask = cv2.resize(gray_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Expand dimensions if needed
                if c:
                    resized_mask = np.expand_dims(resized_mask, axis=-1)
                    if c > 1:
                        resized_mask = resized_mask.repeat(c, axis=-1)

                self._masks["resized_masks"][i] = resized_mask

        def validate_img(self, img_idx:int, img:ImageMat):
            img.require_ndarray()
            img.require_np_uint()
            img.require_HW_or_HWC()

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:            
            if self._masks is None:
                return imgs_data
            
            if "resized_masks" not in self._masks:
                self._adjust_mask([img for img in self.input_mats])

            masks = self._masks["resized_masks"]
            return [cv2.bitwise_and(image, mask) for image, mask in zip(imgs_data, masks)]

    class TorchImageMask(NumpyImageMask):
        title: str = "torch_image_mask"

        def validate_img(self, img_idx: int, img: ImageMat):
            img.require_torch_float()
            img.require_BCHW()

        def _make_mask_images(
            self, mask_image_path: Optional[str], mask_split: Tuple[int, int], preview_color: str
        ) -> Optional[Dict[str, List[Any]]]:
            data = super()._make_mask_images(mask_image_path, mask_split, preview_color)

            if data is None:
                return None

            # Convert grayscale masks to PyTorch tensors in BCHW format, normalized
            data["torch_original"] = [
                torch.as_tensor(mask).unsqueeze(0).unsqueeze(0).to(
                ImageMatInfo.torch_img_dtype()) / 255.0
                for mask in data["original"]
            ]
            return data

        def _adjust_mask(self, images: List[ImageMat]):
            self._masks["resized_masks"] = [None for _ in self._masks["original"]]

            for i, img in enumerate(images):
                gray_mask: np.ndarray = self._masks["original"][i]

                h, w = img.info.H, img.info.W
                # Resize using OpenCV (still in NumPy), then convert to torch tensor
                resized_mask_np = cv2.resize(gray_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                resized_mask_torch = torch.as_tensor(resized_mask_np
                                        ).unsqueeze(0).unsqueeze(0).type(
                                        ImageMatInfo.torch_img_dtype()).div(255.0)
                if img.info.device != 'cpu':
                    resized_mask_torch = resized_mask_torch.to(img.info.device)
                self._masks["resized_masks"][i] = resized_mask_torch

        def forward_raw(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:
            
            if self._masks is None:
                return imgs_data

            if "resized_masks" not in self._masks:
                self._adjust_mask([img for img in self.input_mats])

            masks = self._masks["resized_masks"]
            return [image * mask for image, mask in zip(imgs_data, masks)]

    class TorchDebayer(ImageMatProcessor):
        ### Define the `Debayer5x5` PyTorch Model
        # The `Debayer5x5` model applies a **5x5 convolution filter** to interpolate missing 
        # color information from a Bayer pattern.
        # in list of Bx1xHxW tensor [0.0 ~ 1.0)
        # out list of Bx3xHxW tensor [0.0 ~ 1.0)

        class Debayer5x5(torch.nn.Module):
            # from https://github.com/cheind/pytorch-debayer
            """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

            Requires BG-Bayer color filter array layout. That is,
            the image[1,1]='B', image[1,2]='G'. This corresponds
            to OpenCV naming conventions.

            Compared to Debayer2x2 this method does not use upsampling.
            Compared to Debayer3x3 the algorithm gives sharper edges and
            less chromatic effects.

            ## References
            Malvar, Henrique S., Li-wei He, and Ross Cutler.
            "High-quality linear interpolation for demosaicing of Bayer-patterned
            color images." 2004
            """
            class Layout(enum.Enum):
                """Possible Bayer color filter array layouts.

                The value of each entry is the color index (R=0,G=1,B=2)
                within a 2x2 Bayer block.
                """
                RGGB = (0, 1, 1, 2)
                GRBG = (1, 0, 2, 1)
                GBRG = (1, 2, 0, 1)
                BGGR = (2, 1, 1, 0)

            def __init__(self, layout: Layout = Layout.RGGB):
                super(Processors.TorchDebayer.Debayer5x5, self).__init__()
                self.layout = layout
                # fmt: off
                self.kernels = torch.nn.Parameter(
                    torch.tensor(
                        [
                            # G at R,B locations
                            # scaled by 16
                            [ 0,  0, -2,  0,  0], # noqa
                            [ 0,  0,  4,  0,  0], # noqa
                            [-2,  4,  8,  4, -2], # noqa
                            [ 0,  0,  4,  0,  0], # noqa
                            [ 0,  0, -2,  0,  0], # noqa

                            # R,B at G in R rows
                            # scaled by 16
                            [ 0,  0,  1,  0,  0], # noqa
                            [ 0, -2,  0, -2,  0], # noqa
                            [-2,  8, 10,  8, -2], # noqa
                            [ 0, -2,  0, -2,  0], # noqa
                            [ 0,  0,  1,  0,  0], # noqa

                            # R,B at G in B rows
                            # scaled by 16
                            [ 0,  0, -2,  0,  0], # noqa
                            [ 0, -2,  8, -2,  0], # noqa
                            [ 1,  0, 10,  0,  1], # noqa
                            [ 0, -2,  8, -2,  0], # noqa
                            [ 0,  0, -2,  0,  0], # noqa

                            # R at B and B at R
                            # scaled by 16
                            [ 0,  0, -3,  0,  0], # noqa
                            [ 0,  4,  0,  4,  0], # noqa
                            [-3,  0, 12,  0, -3], # noqa
                            [ 0,  4,  0,  4,  0], # noqa
                            [ 0,  0, -3,  0,  0], # noqa

                            # R at R, B at B, G at G
                            # identity kernel not shown
                        ]
                    ).view(4, 1, 5, 5).float() / 16.0,
                    requires_grad=False,
                )
                # fmt: on

                self.index = torch.nn.Parameter(
                    # Below, note that index 4 corresponds to identity kernel
                    self._index_from_layout(layout),
                    requires_grad=False,
                )

            def forward(self, x):
                """Debayer image.

                Parameters
                ----------
                x : Bx1xHxW tensor [0.0 ~ 1.0)
                    Images to debayer

                Returns
                -------
                rgb : Bx3xHxW tensor [0.0 ~ 1.0)
                    Color images in RGB channel order.
                """
                B, C, H, W = x.shape

                xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
                planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
                planes = torch.cat(
                    (planes, x), 1
                )  # Concat with input to give identity kernel Bx5xHxW
                rgb = torch.gather(
                    planes,
                    1,
                    self.index.repeat(
                        1,
                        1,
                        torch.div(H, 2, rounding_mode="floor"),
                        torch.div(W, 2, rounding_mode="floor"),
                    ).expand(
                        B, -1, -1, -1
                    ),  # expand for singleton batch dimension is faster
                )
                return torch.clamp(rgb, 0, 1)

            def _index_from_layout(self, layout: Layout = Layout) -> torch.Tensor:
                """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

                Note, the index corresponding to the identity kernel is 4, which will be
                correct after concatenating the convolved output with the input image.
                """
                #       ...
                # ... b g b g ...
                # ... g R G r ...
                # ... b G B g ...
                # ... g r g r ...
                #       ...
                # fmt: off
                rggb = torch.tensor(
                    [
                        # dest channel r
                        [4, 1],  # pixel is R,G1
                        [2, 3],  # pixel is G2,B
                        # dest channel g
                        [0, 4],  # pixel is R,G1
                        [4, 0],  # pixel is G2,B
                        # dest channel b
                        [3, 2],  # pixel is R,G1
                        [1, 4],  # pixel is G2,B
                    ]
                ).view(1, 3, 2, 2)
                # fmt: on
                return {
                    layout.RGGB: rggb,
                    layout.GRBG: torch.roll(rggb, 1, -1),
                    layout.GBRG: torch.roll(rggb, 1, -2),
                    layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
                }.get(layout)


            #### Key Features:
            # - Implements **Malvar-He-Cutler** algorithm for Bayer interpolation.
            # - Supports **different Bayer layouts** (`RGGB`, `GRBG`, `GBRG`, `BGGR`).
            # - Uses **fixed convolution kernels** for demosaicing.
        title:str='torch_debayer'
        _debayer_models:List['Processors.TorchDebayer.Debayer5x5'] = []
        _input_devices = []  # To track device of each input tensor

        def validate_img(self, img_idx, img):
            img.require_torch_tensor()
            img.require_BCHW()
            img.require_BAYER()
            # Save input device for tracking
            self._input_devices.append(img.info.device)
            # Initialize and store model on the corresponding device
            model = Processors.TorchDebayer.Debayer5x5().to(img.info.device).to(img.info._dtype)
            self._debayer_models.append(model)

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)
        
        def forward_raw(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:
            debayered_imgs = []
            for i, img in enumerate(imgs_data):
                model = self._debayer_models[i % len(self._debayer_models)]  # Fetch model from pre-assigned list
                debayered_imgs.append(model(img))
            return debayered_imgs

    class MockTorchDebayer(ImageMatProcessor):
        title:str='mock_torch_debayer'
        _debayer_models:List['Processors.TorchDebayer.Debayer5x5'] = []
        _input_devices = []  # To track device of each input tensor

        def validate_img(self, img_idx, img):
            img.require_torch_tensor()
            img.require_BCHW()
            img.require_BAYER()
            # Save input device for tracking
            self._input_devices.append(img.info.device)
            # Initialize and store model on the corresponding device
            def model(x: torch.Tensor) -> torch.Tensor:
                # x: Bx1xHxW — Bayer pattern grayscale input
                # Repeat the single channel into 3 channels (RGB)
                return x.repeat(1, 3, 1, 1)  # Bx3xHxW
            self._debayer_models.append(model)

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)
        
        def forward_raw(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:
            debayered_imgs = []
            for i, img in enumerate(imgs_data):
                # model = self._debayer_models[i % len(self._debayer_models)]  # Fetch model from pre-assigned list
                img = img.repeat(1, 3, 1, 1)  # Bx3xHxW
                debayered_imgs.append(img)
            return debayered_imgs

    class YOLOv5(ImageMatProcessor):
        title:str='YOLOv5_detections'
        gpu:bool=True
        multi_gpu:int=-1
        _torch_dtype:ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()

        modelname: str = 'yolov5s6u.pt'
        imgsz: int = -1
        conf: float = 0.6
        max_det: int = 300
        class_names: Optional[Dict[int, str]] = None
        save_results_to_meta: bool = True
        
        plot_imgs:bool = True
        use_official_predict:bool = True

        yolo_verbose:bool = False

        nms_iou:float = 0.7
        _models:dict = {}

        def model_post_init(self, context):
            self.num_devices = self.devices_info(gpu=self.gpu,multi_gpu=self.multi_gpu)
            default_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
            self.class_names = self.class_names if self.class_names is not None else default_names
            return super().model_post_init(context)

        def validate_img(self, img_idx, img):
                img.require_square_size()
                if img.is_ndarray():
                    img.require_ndarray()
                    img.require_HWC()
                    img.require_RGB()
                elif img.is_torch_tensor():
                    img.require_torch_tensor()
                    img.require_BCHW()
                    img.require_RGB()
                else:
                    raise TypeError("Unsupported image type for YOLO")
                
                if self.imgsz<0:
                    self.imgsz = img.info.W
        
        def forward_raw(self, imgs_data: List[Union[np.ndarray, torch.Tensor]], imgs_info: List[ImageMatInfo]=[], meta={}) -> List["Any"]:
            if len(self._models)==0:
                self.build_models(imgs_info)
            if self.use_official_predict:
                imgs,yolo_results_xyxycc = self.official_predict(imgs_data,imgs_info)
            else:
                imgs,yolo_results_xyxycc = self.predict(imgs_data,imgs_info)

            self.bounding_box_xyxy = yolo_results_xyxycc
            return imgs if len(imgs)>0 else imgs_data

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)
        
        def build_models(self,imgs_info: List[ImageMatInfo]):
            for d in self.num_devices:
                if d not in self._models:
                    model = YOLO(self.modelname, task='detect').to(d)
                    if hasattr(model, 'names'):
                        self.class_names = model.names

                    if not self.use_official_predict:
                        model = model.type(self._torch_dtype)
                        self._models[d] = lambda img,model=model:model.model(img)
                    else:
                        self._models[d] = lambda img,model=model:model.predict(img,
                                                      conf=self.conf,verbose=self.yolo_verbose,
                                                   imgsz=self.imgsz, half=(self._torch_dtype==torch.float16))

                        
        def official_predict(self, imgs_data: List[Union[np.ndarray, torch.Tensor]], imgs_info: List[ImageMatInfo]=[]):
            imgs = []
            yolo_results = []
            for i,(img,info) in enumerate(zip(imgs_data,imgs_info)):
                device = info.device
                yolo_result = self._models[device](img)
                if isinstance(yolo_result, list):
                    yolo_results += yolo_result

            yolo_results_xyxycc = [None]*len(yolo_results)
            for i,yolo_result in enumerate(yolo_results):
                if self.plot_imgs:
                    imgs.append(yolo_result.plot())
                
                if hasattr(yolo_result, 'boxes'):
                    boxes = yolo_result.boxes                    
                    # Convert to numpy: [x1, y1, x2, y2, conf, class_id]
                    xyxycc = torch.cat([
                        boxes.xyxy,                  # (N, 4)
                        boxes.conf.view(-1, 1),      # (N, 1)
                        boxes.cls.view(-1, 1)        # (N, 1)
                    ], dim=1).cpu().numpy()          # (N, 6)
                else:
                    xyxycc = np.zeros((0, 6), dtype=np.float32)  # no detections

                yolo_results_xyxycc[i] = xyxycc
            return imgs,yolo_results_xyxycc
        
        def predict(self, imgs_data: List[Union[np.ndarray, torch.Tensor]], imgs_info: List[ImageMatInfo]=[]):
            yolo_results = []
            for img,info in zip(imgs_data,imgs_info):
                yolo_model:YOLO = self._models[info.device]
                preds, feature_maps = yolo_model(img)                
                print(preds[:,4:,:].max())
                preds = ops.non_max_suppression(
                    preds,
                    self.conf,
                    self.nms_iou,
                    classes = None,
                    agnostic= False,                    
                    max_det = self.max_det,
                    nc      = len(self.class_names),
                    end2end = False,
                    rotated = False,
                )
                if isinstance(preds, list):
                    yolo_results += preds
                else:
                    yolo_results.append(preds)
            yolo_results_xyxycc:list[np.ndarray] = [r.cpu().numpy() for r in yolo_results]
            return [],yolo_results_xyxycc
        
    class YOLOv5TRT(YOLOv5):
        title:str = 'YOLOv5_TRT_detections'
        modelname: str = 'yolov5s6u.engine'
        use_official_predict:bool = False
        conf: float = 0.0001        

        def build_models(self,imgs_info: List[ImageMatInfo]):
            config = dict(imgsz=self.imgsz,                           
                        conf=self.conf,verbose=self.yolo_verbose,
                        half=(self._torch_dtype==torch.float16))
            self._models = {}
            for i,d in enumerate(self.num_devices):
                with torch.cuda.device(d):
                    modelname = self.modelname.replace('.trt',(f'_{d}.trt').replace(':','@'))
                    yolo = GeneralTensorRTInferenceModel(modelname,d,'images','output0')
                self._models[d] = yolo
                
            self.use_official_predict = False            
            self.print('load mode by config :',config)

        def predict(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[]):
            yolo_results = []
            for img,info in zip(imgs_data,imgs_info):
                yolo_model = self._models[info.device]
                print(img.device,img.dtype,yolo_model.device)
                preds = yolo_model(img)
                time.sleep(1)
                # print(preds.shape,preds.dtype)
                # print(preds[:,4:,:].max())
            #     preds = ops.non_max_suppression(
            #         preds,
            #         self.conf,
            #         self.nms_iou,
            #         classes = None,
            #         agnostic= False,
            #         max_det = self.max_det,
            #         nc      = len(self.class_names),
            #         in_place= False,
            #         end2end = False,
            #         rotated = False,
            #     )
            #     if isinstance(preds, list):
            #         yolo_results += preds
            #     else:
            #         yolo_results.append(preds)
            # yolo_results_xyxycc:list[np.ndarray] = [r.cpu().numpy() for r in yolo_results]
            # print(yolo_results_xyxycc)
            yolo_results_xyxycc:list[np.ndarray] = [np.empty((0, 6), dtype=np.float32) for _ in range(4)]
            # print(yolo_results_xyxycc)
            return [],yolo_results_xyxycc
        
    @staticmethod    
    def dumps(pipes:list[ImageMatProcessor]):
        return json.dumps([p.model_dump() for p in pipes])
    
    @staticmethod
    def loads(pipes_json:str)->list[ImageMatProcessor]:
        processors = {k: v for k, v in Processors.__dict__.items() if '__' not in k}
        return [processors[f'{p["uuid"].split(":")[0]}'](**p) 
                for p in json.loads(pipes_json)]

    @staticmethod    
    def run_once(imgs,meta={},
            pipes:list['ImageMatProcessor']=[],
            validate=False):
        if validate:
            try:
                for fn in pipes:
                    imgs,meta = fn.validate(imgs,meta)
            except Exception as e:
                logger(fn.uuid,e)
                raise e
        else:
            for fn in pipes:
                imgs,meta = fn(imgs,meta)
        return imgs,meta

    @staticmethod    
    def validate_once(gen:ImageMatGenerator,
            pipes:list['ImageMatProcessor']=[],
            meta = {}):
        for imgs in gen:
            ImageMatProcessors.run_once(imgs,meta,pipes,True)
            break