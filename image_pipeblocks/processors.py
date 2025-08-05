# ===============================
# Standard Library Imports
# ===============================
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import enum
import json
import math
from multiprocessing import shared_memory
import multiprocessing
import os
import queue
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
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
# Custom Modules
# ===============================
from .shmIO import NumpyUInt8SharedMemoryStreamIO
from .ImageMat import ColorType, ImageMat, ImageMatInfo, ImageMatProcessor, ShapeType


logger = print

def hex2rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA format."""
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    alpha = int(hex_color[6:8], 16) if len(hex_color) == 8 else 255
    return rgba + (alpha,)


try:
    # ===============================
    # TensorRT & CUDA Imports
    # ===============================
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
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
except:
    pass

class Processors:

    class DoingNothing(ImageMatProcessor):
        title:str='doing_nothing'
        def validate_img(self, img_idx, img):
            pass

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            return imgs_data
    try:        
        from .gps import BaseGps, FileReplayGps, UsbGps
        class GPS(ImageMatProcessor):
            title:str='get_gps'
            port:str = 'gps.jsonl'
            save_results_to_meta:bool = True
            _gps:BaseGps = None

            @staticmethod
            def coms():
                return BaseGps.coms()

            def change_port(self,port:str):
                self.off()
                self.port = port
                self.on() 

            def get_state(self):
                if self._gps:
                    return json.loads(self._gps.get_state().model_dump_json())
                else:
                    return {}
            
            def get_latlon(self)->list[float]:
                if self._gps:
                    s = self._gps.get_state()
                    return [s.lat,s.lon]
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
                    self._gps:BaseGps = FileReplayGps()
                else:                
                    self._gps:BaseGps = UsbGps()
                self._gps.open(self.port)

            def validate_img(self, img_idx, img):
                if self._gps is None:
                    self.ini_gps()

            def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
                return imgs_data
    except:
        pass

    class CropToDivisibleBy32(ImageMatProcessor):
        title: str = 'crop_to_divisible_by_32'
        hs:list[int] = []
        ws:list[int] = []
        
        def validate_img(self, img_idx, img):
            img.require_np_uint()
            img.require_HW_or_HWC()
            h, w = img.info.H,img.info.W
            new_h = h - (h % 32)
            new_w = w - (w % 32)
            self.hs.append(new_h)
            self.ws.append(new_w)

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo] = [], meta={}) -> List[np.ndarray]:
            processed_imgs = []
            for i,img in enumerate(imgs_data):
                h,w = self.hs[i],self.ws[i]
                processed_imgs.append(img[:h, :w])
            return processed_imgs
  
    class CvDebayer(ImageMatProcessor):
        title:str='cv_debayer'
        format:int=cv2.COLOR_BAYER_BG2BGR
            
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
        
        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            return [cv2.cvtColor(i,self.format) for i in imgs_data]

    class NumpyRGBToNumpyBGR(ImageMatProcessor):
        title:str='numpy_rgb_to_bgr'

        def validate_img(self, img_idx, img):
            img.require_RGB()
            img.require_np_uint()
        
        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.BGR):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)
        
        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            return [img[:, :, [2, 1, 0]] for img in imgs_data]
        
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

    class NumpyPadImage(ImageMatProcessor):
        """
        Pads an image using numpy's np.pad.
        Supports constant, edge, reflect, etc.
        """
        title: str = "numpy_pad_image"
        pad_width: Tuple[Tuple[int, int], Tuple[int, int]] = ((10, 10), (10, 10))  # ((top, bottom), (left, right))
        pad_value: int = 0
        mode: str = "constant"  # Options: 'constant', 'edge', 'reflect', 'symmetric', etc.
        pad_widths:list=[]

        def validate_img(self, img_idx: int, img: ImageMat):
            img.require_np_uint()
            img.require_HW_or_HWC()

            C = img.info.C
            if C == 2:
                # Grayscale image: pad HxW
                pad_width = self.pad_width
            elif C == 3:
                # Color image: pad HxW only, leave channel unchanged
                pad_width = self.pad_width + ((0, 0),)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            self.pad_widths.append(pad_width)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            padded_imgs = []
            for i,img in enumerate(imgs_data):
                pad_width = self.pad_widths[i]
                padded_img = np.pad(img, pad_width, mode=self.mode, constant_values=self.pad_value)
                padded_imgs.append(padded_img)
            return padded_imgs

        def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo]=[]):
            self.pixel_idx_forward_T = []
            self.pixel_idx_backward_T = []

            for info in imgs_info:
                transform_matrix = np.eye(3, dtype=np.float32)  # Identity base
                transform_matrix = np.eye(3, dtype=np.float32)  # Identity base

                pad_top, pad_bottom = self.pad_width[0]
                pad_left, pad_right = self.pad_width[1]

                # Create transform matrix for pixel mapping
                T = np.array([
                    [1, 0, pad_left],
                    [0, 1, pad_top],
                    [0, 0, 1]
                ], dtype=np.float32)

                transform_matrix = T
                self.pixel_idx_forward_T.append(transform_matrix.tolist())
                self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())

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
                bgr_images+=[i for i in img]
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
        
    class CVResize(ImageMatProcessor):
        title:str='cv_resize'
        target_size: Tuple[int, int]
        interpolation:int=cv2.INTER_LINEAR

        def validate_img(self, img_idx, img):            
            img.require_ndarray()
            img.require_HW_or_HWC()

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            """
            Resizes a batch of NumPy images using OpenCV.
            """
            resized_images = []
            for img in imgs_data:
                resized_img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                                        interpolation=self.interpolation)
                resized_images.append(resized_img)
            return resized_images

        def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo]=[]):
            self.pixel_idx_forward_T = []
            self.pixel_idx_backward_T = []

            for info in imgs_info:
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
        
    class TileNumpyImages(ImageMatProcessor):
        class Layout(BaseModel):
            tile_width:int
            tile_height:int

            col_widths:list[int]
            row_heights:list[int]

            total_width:int
            total_height:int

            channels:int # 1, 3
            _canvas:Any

        title:str='tile_numpy_images'
        tile_width:int
        layout:Optional[Layout] = None

        def _init_layout(self, imgs:list[ImageMat]):
            imgs_info = [i.info for i in imgs]
            num_images = len(imgs_info)
            if num_images == 0:
                raise ValueError("No input images info for doing tile.")
            tile_width = self.tile_width
            tile_height = math.ceil(num_images / tile_width)

            # Compute max width for each column, max height for each row
            col_widths = [0] * tile_width
            row_heights = [0] * tile_height

            for idx, info in enumerate(imgs_info):
                row, col = divmod(idx, tile_width)
                h, w = info.H,info.W
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

            layout =  Processors.TileNumpyImages.Layout(
                tile_width=tile_width,
                tile_height=tile_height,
                col_widths=col_widths,
                row_heights=row_heights,
                total_width=total_width,
                total_height=total_height,
                channels=channels)
            layout._canvas=canvas
            return layout

        def validate_img(self, img_idx, img):            
            img.require_np_uint()
            img.require_HWC()
        
        def validate(self, imgs, meta = ...):
            color_types = {i.info.color_type for i in imgs}
            if len(color_types) != 1:
                raise ValueError(f"All images must have the same color_type, got {color_types}")
            super().validate(imgs, meta, run=False)
            self.layout = self._init_layout(self.input_mats)
            return self(self.input_mats, meta)
        
        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
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
                    img:np.ndarray = imgs_data[idx]
                    h, w = img.shape[:2]
                    if channels == 1:
                        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
                    else:
                        canvas[y_offset:y_offset + h, x_offset:x_offset + w, :channels] = img
                    x_offset += col_widths[col]
                y_offset += row_heights[row]
            return [canvas]

        def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo]=[]):
            # Ensure the layout exists (may be first call on this instance)
            if self.layout is None:
                raise ValueError("Layout not initialized. Call forward() first to build layout.")

            layout       = self.layout
            tile_width   = layout.tile_width
            col_widths   = layout.col_widths
            row_heights  = layout.row_heights

            # Pre-compute cumulative offsets so we don’t sum inside every loop
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
                x_off    = x_prefix[col]      # left edge of this column in canvas
                y_off    = y_prefix[row]      # top  edge of this row    in canvas

                # Pure translation (no scale / rotation)
                T = np.array([[1, 0, x_off],
                            [0, 1, y_off],
                            [0, 0,     1]], dtype=np.float32)

                self.pixel_idx_forward_T.append(T.tolist())
                self.pixel_idx_backward_T.append(np.linalg.inv(T).tolist())

        def forward_transform_matrix(self, proc):
            res = super().forward_transform_matrix(proc)
            proc.bounding_box_xyxy = [np.vstack(proc.bounding_box_xyxy)]
            return res
        
    class EncodeNumpyToJpeg(ImageMatProcessor):
        title:str='encode_numpy_to_jpeg'
        quality: int = 90

        def validate_img(self, img_idx, img):
            img.require_ndarray()
            img.require_HWC()
            
        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.JPEG):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            """
            Encodes a batch of NumPy images to JPEG format.
            """
            encoded_images = []
            for img in imgs_data:
                success, encoded = cv2.imencode('.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                                int(self.quality)])
                if not success:
                    raise ValueError("JPEG encoding failed.")

            encoded_images.append(encoded)
            return encoded_images   

    class CvVideoRecorder(ImageMatProcessor):
        class VideoWriterWorker:
            def __init__(self, frame_interval=0.1, subset_s=100,
                        queue_size=30, overlay_text: str = None):
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
                        frame = frame[..., [2, 1, 0]]  # RGB to BGR

                    if self.overlay_text:
                        frame = frame.copy()
                        cv2.putText(frame, self.overlay_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Check if new subset file is needed
                    if self.writer is None or (now - self.writer_start_time) >= self.subset_s:
                        if self.writer is not None:
                            self.writer.release()
                            # logger(f"Closed video file chunk {self.file_counter}")

                        self.file_counter += 1
                        filename = f"{self.base_filename}-{self.file_counter:03d}{self.file_ext}"
                        fourcc = cv2.VideoWriter_fourcc(*self.codec)
                        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.w, self.h))
                        self.writer_start_time = now
                        # logger(f"Started new video file: {filename}")

                    try:
                        self.writer.write(frame)
                        self.last_write_time = now
                    except Exception as e:
                        pass
                        # logger(f"Error writing frame: {e}")
                    self.queue.task_done()

                if self.writer:
                    self.writer.release()
                    # logger("Final video file closed.")

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
                # if self.thread:
                #     self.thread.join()


        title: str = "cv_recorder"
        output_filename: str = "output.avi"
        codec: str = "XVID"
        fps: int = 30
        overlay_text: Optional[str] = None
        total_imgs:int=0
        WHs:list[tuple[int,int]] = []
        frame_interval:float=0.0
        subset_s:int = 20 # seconds
        _workers: List['Processors.CvVideoRecorder.VideoWriterWorker'] = []

        def model_post_init(self, context):
            self.frame_interval = 1.0 / self.fps
            return super().model_post_init(context)

        def validate_img(self, img_idx, img: ImageMat):
            if img.is_ndarray():
                img.require_ndarray()
                img.require_np_uint()
                img.require_BGR()
                img.require_HWC()
                self.total_imgs+=1
                self.WHs.append((img.info.W,img.info.H))
            if img.is_torch_tensor():                
                img.require_torch_float()
                img.require_RGB()
                img.require_BCHW()
                self.total_imgs+=img.info.B
                for i in range(img.info.B):
                    self.WHs.append((img.info.W,img.info.H))

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
            return res.replace('_','-')

        def start(self):
            self._workers = []
            for idx in range(self.total_imgs):
                filename = self.format_filename(f'_{idx}' if self.total_imgs>1 else '')
                w,h = self.WHs[idx]
                worker = Processors.CvVideoRecorder.VideoWriterWorker(
                                frame_interval=self.frame_interval,
                                subset_s=self.subset_s,
                                queue_size=30).build_writer(
                                filename, self.codec, self.fps, w, h)
                self._workers.append(worker)
            for w in self._workers:w.start()

        def stop(self):
            for w in self._workers: w.stop()
            self._workers = []

        def forward_raw(self,imgs_data: List[np.ndarray],imgs_info: List[ImageMatInfo] = [],meta={},) -> List[np.ndarray]:
            cnt=0
            for idx, frame in enumerate(imgs_data):
                try:
                    if isinstance(frame,np.ndarray):
                        self._workers[cnt].queue.put_nowait(frame)
                        cnt+=1
                    if isinstance(frame,torch.Tensor):
                        for f in frame:
                            self._workers[cnt].queue.put_nowait(f)
                            cnt+=1
                except queue.Full:
                    pass  # drop
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

    class NumpyImageMask(ImageMatProcessor):
        title: str = "numpy_image_mask"
        mask_image_path: str
        mask_color_hex: str = "#000000FF"
        mask_color: Tuple[int, int, int] = (0,0,0)
        mask_alpha: float = 0
        mask_split: Optional[Tuple[int, int]] = (2, 2)
        _original_masks:list = []
        _resized_masks:list = []
        _revert_masks:list = []

        def model_post_init(self, context: Any) -> None:
            self.mask_color = np.array(hex2rgba(self.mask_color_hex)[:3], dtype=np.uint8)
            return super().model_post_init(context)
        
        def reload_masks(self):
            self.load_masks(self.mask_image_path, self.mask_split)

        def load_masks(self,mask_image_path: Optional[str], mask_split: Tuple[int, int]):
            self._original_masks = []
            self._resized_masks = []
            self._revert_masks = []
            self._numpy_make_mask_images(mask_image_path, mask_split)
            [self._numpy_adjust_mask(i,img) for i,img in enumerate(self.input_mats)]

        def _numpy_make_mask_images(self, mask_image_path: Optional[str], mask_split: Tuple[int, int]):
            if mask_image_path is None: return None

            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
            if mask_image is None:
                raise ValueError(f"Unable to read mask image from {mask_image_path}")

            # Split mask into sub-masks
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
            # Convert to grayscale and apply preview color
            self._original_masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in mask_images]

        def _numpy_adjust_mask(self, idx, img:ImageMat):
            gray_mask: np.ndarray = self._original_masks[idx]

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
            if idx!=len(self._resized_masks):raise ValueError('size checking error.')
            self._resized_masks.append(resized_mask)
            revert_mask = np.full_like(resized_mask, # to BGR
                                        self.mask_color[::-1], dtype=resized_mask.dtype)
            revert_mask = cv2.bitwise_and(revert_mask, ~resized_mask)
            self._revert_masks.append(revert_mask)

        def validate_img(self, img_idx:int, img:ImageMat):
            img.require_ndarray()
            img.require_np_uint()
            img.require_HW_or_HWC()
            if self.mask_alpha>0:
                img.require_BGR()

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[],
                        meta={}) -> List[np.ndarray]:
            if len(self._resized_masks)==0:
                self.load_masks(self.mask_image_path, self.mask_split)

            keep_pixel = [cv2.bitwise_and(img, self._resized_masks[i]) for i,img in enumerate(imgs_data)]
            if self.mask_alpha>0:
                transparent_pixel = [cv2.bitwise_and(img, ~self._resized_masks[i]) for i,img in enumerate(imgs_data)]
                
                result = [keep + ((trans*(1-self.mask_alpha)).astype(np.uint8)+(bg*self.mask_alpha).astype(np.uint8))
                        for keep, trans, bg in zip(keep_pixel, transparent_pixel, self._revert_masks)]
                return result
            else:
                return keep_pixel

    class TorchImageMask(NumpyImageMask):
        title: str = "torch_image_mask"
        img_cnt:int=0
        _torch_original_masks:list = []

        gpu:bool=True
        multi_gpu:int=-1
        _torch_dtype:ImageMat.TorchDtype = ImageMatInfo.torch_img_dtype()

        def model_post_init(self, context):
            self.num_devices = self.devices_info(gpu=self.gpu,multi_gpu=self.multi_gpu)
            return super().model_post_init(context)
        
        def validate_img(self, img_idx: int, img: ImageMat):
            img.require_torch_float()
            img.require_BCHW()
            if self.mask_alpha>0:
                img.require_RGB()

        def load_masks(self,mask_image_path: Optional[str], mask_split: Tuple[int, int]):
            self._original_masks = []
            self._resized_masks = []
            self._revert_masks = []
            self._torch_original_masks = []
            self._numpy_make_mask_images(mask_image_path,mask_split)
            img_cnt = 0
            if self.input_mats:
                for i,img in enumerate(self.input_mats):
                    h, w = img.info.H, img.info.W
                    if img.info.B>1:
                        img_masks=[]
                        for j in range(img.info.B):
                            resized_mask_np = cv2.resize(self._original_masks[img_cnt], (w, h), interpolation=cv2.INTER_NEAREST)
                            img_masks.append(resized_mask_np)
                            img_cnt+=1
                        self._resized_masks.append( torch.as_tensor(np.asarray(img_masks)
                                            ).unsqueeze(0).permute(1, 0, 2, 3).to(img.info.device
                                            ).type(ImageMatInfo.torch_img_dtype()).div(255.0))
                    else:
                        resized_mask_np = cv2.resize(self._original_masks[img_cnt], (w, h), interpolation=cv2.INTER_NEAREST)
                        self._resized_masks.append( torch.as_tensor(resized_mask_np
                                            ).unsqueeze(0).unsqueeze(0).to(img.info.device
                                            ).type(ImageMatInfo.torch_img_dtype()).div(255.0))
                        img_cnt+=1

                for i,img in enumerate(self.input_mats):
                    h, w = img.info.H, img.info.W                                            
                    revert_mask_torch = torch.tensor(self.mask_color, dtype=self._resized_masks[i].dtype).view(1, 3, 1, 1)
                    revert_mask_torch = revert_mask_torch.expand(1, 3, h, w).clone().to(img.info.device).type(
                                            ImageMatInfo.torch_img_dtype()).div(255.0)
                    revert_mask_torch = revert_mask_torch*(1.0 - self._resized_masks[i])
                    self._revert_masks.append(revert_mask_torch)

        def forward_raw(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[torch.Tensor]:
            if len(self._resized_masks)==0:
                self.load_masks(self.mask_image_path, self.mask_split)

            keep_pixel = [image * self._resized_masks[i] for i,image in enumerate(imgs_data)]

            if self.mask_alpha>0:
                transparent_pixel = [image * (1.0 - self._resized_masks[i]) for i,image in enumerate(imgs_data)]          
                return [keep + ((trans*(1-self.mask_alpha))+(bg*self.mask_alpha))
                        for keep, trans, bg in zip(keep_pixel, transparent_pixel, self._revert_masks)]
            else:
                return keep_pixel

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

    class SlidingWindowSplitter(ImageMatProcessor):

        title: str = "sliding_window"
        stride: Optional[Tuple[int, int]] = None
        window_size: Tuple[int, int] = (1280, 1280)
        imgs_idx:dict[int,list] = {}
        # input_imgs_info:list[ImageMatInfo] = []
        output_offsets_xyxy:list[ List[Tuple[int, int, int, int]] ] = []
        save_results_to_meta:bool =True

        def validate_img(self, img_idx:int, img:ImageMat):
            if self.stride is None:
                self.stride = self.window_size
            img.require_np_uint()
            img.require_HW_or_HWC()
            H, W = img.info.H,img.info.W
            wH, wW = self.window_size
            if wH > H or wW > W:
                raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
            # 1:N mapping
            out_mats:list[ImageMat] = []
            for i,v in self.imgs_idx.items():
                img = validated_imgs[i]
                out_mats += [img for _ in v]

            self.out_mats = [
                ImageMat(color_type=out_mats[i].info.color_type).build(img)
                for i,img in enumerate(converted_raw_imgs)]
            return self.out_mats            

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            
            out_imgs:list[np.ndarray] = []
            output_offsets_xyxy = []
            imgs_idx = {}

            for i, img in enumerate(imgs_data):
                # windows, offsets = self._split_numpy(img.data(), meta)            
                H, W = img.shape[0],img.shape[1]
                wH, wW = self.window_size
                sH, sW = self.stride

                windows_list = []
                offsets_xyxy = []
                
                for row_start in range(0, H - wH + 1, sH):
                    for col_start in range(0, W - wW + 1, sW):
                        window = img[row_start:row_start + wH, col_start:col_start + wW, :]
                        windows_list.append(window)
                        offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))  # Can be adjusted
            
                image_mats = [w for w in windows_list]

                imgs_idx[i] = list(range(len(out_imgs), len(out_imgs) + len(image_mats)))
                out_imgs+=image_mats
                output_offsets_xyxy.append(offsets_xyxy)

            self.imgs_idx = imgs_idx
            # self.input_imgs_info = [i.info.model_copy() for i in self.input_mats]
            self.output_offsets_xyxy = output_offsets_xyxy
            return out_imgs
        
        def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo] = []):
            self.pixel_idx_forward_T = []
            self.pixel_idx_backward_T = []

            for offsets in self.output_offsets_xyxy:
                for offset in offsets:
                    x1, y1, _, _ = offset  # top-left corner of window

                    # Forward: Full image → Window space (shift origin)
                    transform_matrix = np.array([
                        [1.0, 0.0, -x1],
                        [0.0, 1.0, -y1],
                        [0.0, 0.0,  1.0]
                    ], dtype=np.float32)

                    self.pixel_idx_forward_T.append(transform_matrix.tolist())
                    self.pixel_idx_backward_T.append(np.linalg.inv(transform_matrix).tolist())

    class SlidingWindowMerger(ImageMatProcessor):
        title: str = "sliding_window_merge"
        sliding_window_splitter_uuid:str = ''
        yolo_uuid:str = ''
        _sliding_window_splitter:'Processors.SlidingWindowSplitter'=None

        def validate_img(self, img_idx:int, img:ImageMat):
            img.require_np_uint()
            img.require_HW_or_HWC()
                
        def forward_raw_yolo(self,sw_yolo_proc: 'Processors.YOLO'):
            detections_per_window = sw_yolo_proc.bounding_box_xyxy
            transforms = self._sliding_window_splitter.pixel_idx_backward_T

            merged_detections = [np.empty((0, 6), dtype=np.float32)
                    for i in range(len(self._sliding_window_splitter.input_mats))]
            
            if len(detections_per_window) != len(transforms):
                raise ValueError(f"Number of detections_per_window({len(detections_per_window)}) and transforms({len(transforms)}) must match.")

            transform_detections = [None]*len(transforms)

            for i,dets in enumerate(detections_per_window):
                T = transforms[i]
                transform_detections[i] = dets
                if len(dets) == 0: continue
                T = np.array(T)  # 3x3
                coords = dets[:, :4]
                ones = np.ones((coords.shape[0], 1))
                xy1 = np.concatenate([coords[:, :2], ones], axis=1)
                xy2 = np.concatenate([coords[:, 2:], ones], axis=1)
                
                xy1_trans = (T @ xy1.T).T
                xy2_trans = (T @ xy2.T).T
                new_boxes = np.concatenate([xy1_trans[:, :2], xy2_trans[:, :2], dets[:, 4:]], axis=1)
                transform_detections[i] = new_boxes
            
            for img_idx,info in enumerate(self._sliding_window_splitter.input_mats):
                tile_indices:list = self._sliding_window_splitter.imgs_idx[img_idx]
                merged_detections[img_idx] = np.vstack([transform_detections[i] for i in tile_indices])                
            return merged_detections

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
            color_types = []
            for img_idx,info in enumerate(self._sliding_window_splitter.input_mats):
                offsets:list[tuple[int,int,int,int]] = self._sliding_window_splitter.output_offsets_xyxy[img_idx]
                tile_indices:list = self._sliding_window_splitter.imgs_idx[img_idx]
                tile_idx = tile_indices[-1]
                color_types.append(validated_imgs[tile_idx].info.color_type)
                
            self.out_mats = [ImageMat(color_type=color_types[i]).build(img)
                for i,img in enumerate(converted_raw_imgs)]
            return self.out_mats

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            """Merge sliding window outputs back into full-size images."""
            self._sliding_window_splitter = meta[self.sliding_window_splitter_uuid]

            merged_imgs = [None]*len(self._sliding_window_splitter.input_mats)
            for img_idx,img in enumerate(self._sliding_window_splitter.input_mats):                
                if self.out_mats and self.out_mats[img_idx].data() is not None:
                    merged = self.out_mats[img_idx].data()
                else:
                    H, W, channels = img.info.H, img.info.W, img.info.C
                    if channels>1:
                        merged = np.zeros((H, W, channels), dtype=np.uint8)
                    else:
                        merged = np.zeros((H, W), dtype=np.uint8)

                offsets:list[tuple[int,int,int,int]] = self._sliding_window_splitter.output_offsets_xyxy[img_idx]
                tile_indices:list = self._sliding_window_splitter.imgs_idx[img_idx]
                for i,tile_idx in enumerate(tile_indices):
                    x1, y1, x2, y2 = offsets[i]
                    merged[y1:y2, x1:x2] = imgs_data[tile_idx]
                merged_imgs[img_idx] = merged

            if self.yolo_uuid:
                yolo:Processors.YOLO = meta[self.yolo_uuid]
                yolo.bounding_box_xyxy = self.forward_raw_yolo(yolo)

            return merged_imgs

    class YOLO(ImageMatProcessor):
        title:str='YOLO_detections'
        gpu:bool=True
        multi_gpu:int=-1
        _torch_dtype:Any = ImageMatInfo.torch_img_dtype()

        modelname: str = 'yolov5s6u.pt'
        imgsz: int = -1
        conf: Union[float,dict[int,float]] = 0.6
        min_conf: float = 0.6
        max_det: int = 300
        class_names: Optional[Dict[int, str]] = None
        save_results_to_meta: bool = True
        
        plot_imgs:bool = True
        use_official_predict:bool = True

        yolo_verbose:bool = False

        nms_iou:float = 0.7
        _models:dict = {}
        devices:list[str] = []

        def change_model(self,modelname:str):
            self.modelname = modelname
            self.model_post_init(None)
            self.build_models()

        def model_post_init(self, context):
            self.num_devices = self.devices_info(gpu=self.gpu,multi_gpu=self.multi_gpu)
            default_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
            self.class_names = self.class_names if self.class_names is not None else default_names
            if isinstance(self.conf,dict):
                self.min_conf=min(list(self.conf.values()))
            else:
                self.min_conf=self.conf
            return super().model_post_init(context)

        def validate_img(self, img_idx, img):
                # img.require_square_size()
                if img.is_ndarray():
                    img.require_ndarray()
                    img.require_HWC()
                    img.require_RGB() # TODO:YOLO input form?
                elif img.is_torch_tensor():
                    img.require_torch_tensor()
                    img.require_BCHW()
                    img.require_RGB()
                else:
                    raise TypeError("Unsupported image type for YOLO")
                
                if self.imgsz<0:
                    self.imgsz = img.info.W
                
                self.devices.append(img.info.device)

        def forward_raw(self, imgs_data: List[Union[np.ndarray, torch.Tensor]], imgs_info: List[ImageMatInfo]=[], meta={}) -> List["Any"]:
            if len(self._models)==0:
                self.build_models()
            if self.use_official_predict:
                imgs,yolo_results_xyxycc = self.official_predict(imgs_data,imgs_info)
            else:
                imgs,yolo_results_xyxycc = self.predict(imgs_data,imgs_info)

            # After prediction
            # x1, y1, x2, y2, conf, class_id
            self.bounding_box_xyxy = []
            for xyxycc in yolo_results_xyxycc:
                confs = xyxycc[:,4]
                ids = xyxycc[:,5]
                thresholds = [self.conf[int(i)] if isinstance(self.conf,dict) else self.conf for i in ids]
                self.bounding_box_xyxy.append(xyxycc[confs>thresholds])

            res = imgs if len(imgs)>0 else imgs_data
            return res

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)
        
        def build_models(self):
            for d in self.num_devices:
                if d not in self._models:
                    model = YOLO(self.modelname, task='detect').to(d)
                    if hasattr(model, 'names'):
                        self.class_names = model.names

                    if not self.use_official_predict:
                        model = model.type(self._torch_dtype)
                        def model_predict(img,model=model):
                            res = model.model(img)
                            return res
                        self._models[d] = model_predict
                    else:
                        def model_predict(img,model=model,device=d,
                                          conf=self.min_conf,verbose=self.yolo_verbose,
                                          imgsz=self.imgsz, half=(self._torch_dtype==torch.float16)):
                            res = model.predict(source=img,conf=conf,verbose=verbose,half=half)
                                                # device=device,
                                                # imgsz=imgsz, half=half)
                            return res
                        self._models[d] = model_predict
                        
        def official_predict(self, imgs_data: List[Union[np.ndarray, torch.Tensor]], imgs_info: List[ImageMatInfo]=[]):
            imgs = []
            yolo_results = []
            for i,img in enumerate(imgs_data):
                device = self.num_devices[i % self.num_gpus]
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
            for i,img in enumerate(imgs_data):
                device = self.num_devices[i % self.num_gpus]
                yolo_model:YOLO = self._models[device]
                preds, feature_maps = yolo_model(img)                
                preds = ops.non_max_suppression(
                    preds,
                    self.min_conf,
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

    class SimpleTracking(ImageMatProcessor):
        title:str='simple_tracking'
        detect_frames_thre:int= 10
        ignore_frames_thre:int= 3
        queue_len:int= 20
        
        detector_uuid:str
        frame_cnt:int = 0
        frame_recs:list[dict[int,deque[Tuple[int,ImageMat|None]]]] = []
            # {0:0,1:0,2:0}, # each class continuous cnt
        frame_save_queue:queue.Queue[tuple[str,list[ImageMat]]] = queue.Queue()
        frame_save_queue_len:int= 1000
        class_names:Dict[str,str] = {}
        all_cls_id:set[int] = set()

        save_jpeg:bool=False
        save_mp4:bool=False
        save_jpeg_gps:bool=False
        save_dir:str=''
        gps_uuid:str=''

        timezone:int=9#(UTC+9)

        _det_processor:'Processors.YOLO'= None
        _gps_processor:'Processors.GPS'= None
        
        def model_post_init(self, context):
            self.queue_len = self.detect_frames_thre+self.ignore_frames_thre+1
            if  self.save_jpeg or self.save_mp4 or self.save_jpeg_gps:
                if not os.path.isdir(self.save_dir):
                    raise ValueError(f'save_dir {self.save_dir} not exist')
                self.save_dir = os.path.abspath(self.save_dir)
                self.frame_save_queue = queue.Queue(maxsize=self.frame_save_queue_len)
                save_thread = threading.Thread(target=self._save_worker, daemon=True)
                save_thread.start()
            return super().model_post_init(context)

        def validate_img(self, img_idx, img):
            img.require_ndarray()
            img.require_HWC()
            img.require_BGR()

        def validate(self, imgs, meta = ..., run=True):
            if self.detector_uuid and self.detector_uuid not in meta:
                raise ValueError(f"detector_uuid {self.detector_uuid} not found in meta")
            if self.gps_uuid and self.gps_uuid not in meta:
                raise ValueError(f"gps_uuid {self.gps_uuid} not found in meta")
            return super().validate(imgs, meta, run)

        def _init_frame_recs(self,size=0):
            self.class_names = self._det_processor.class_names
            self.all_cls_id = set(list(self.class_names.keys()))
        
            for _ in range(size):
                frame_rec = {k:deque(maxlen=self.queue_len) for k in self.class_names.keys()}
                for k,v in frame_rec.items():
                    for i in range(self.queue_len):
                        v.append((0,None))
                self.frame_recs.append(frame_rec)

        def _save_worker(self):
            while True:
                class_name,imagemat_list = self.frame_save_queue.get()
                if imagemat_list is None:return

                timestamp = imagemat_list[len(imagemat_list)//2].timestamp
                # Create timezone-aware UTC datetime
                dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                # Convert to (UTC+N)
                dt_jst = dt_utc + timedelta(hours=self.timezone)
                # Create file name string (JST)
                timestamp = dt_jst.strftime("%Y%m%d_%H%M%S")
                filename = f'{self.save_dir}/{timestamp}_{class_name}'

                os.makedirs(filename,exist_ok=True)

                for i,img in enumerate(imagemat_list):
                    fn = f'{filename}/{i}.jpeg'
                    cv2.imwrite(fn, img.data())
                    if self.save_jpeg_gps:
                        self._gps_processor._gps.set_jpeg_gps_location(
                            fn,img.info.latlon[0],img.info.latlon[1],fn)
                
                if self.save_mp4:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    writer = cv2.VideoWriter(f'{filename}/video.mp4', fourcc, 1, (img.info.W, img.info.H))
                    for i,img in enumerate(imagemat_list):
                        writer.write(img.data())
                    writer.release()
                self.frame_save_queue.task_done()

        def forward_raw(self, imgs_data:list[np.ndarray], imgs_info = ..., meta=...):
            self.frame_cnt += 1
            if self.detector_uuid:
                self._det_processor = meta[self.detector_uuid]
            if self.gps_uuid:         
                self._gps_processor = meta[self.gps_uuid]         
            if not self.frame_recs:
                self._init_frame_recs(len(imgs_data))

            for idx,img in enumerate(imgs_data):
                frame_rec = self.frame_recs[idx]
                detections = self._det_processor.bounding_box_xyxy[idx]

                cls_id = set()
                if len(detections)>0:
                    # x1, y1, x2, y2, conf, cls_id
                    cls_id = set(detections[:,5].flatten().tolist())
            
                for i in cls_id:
                    imgc = None
                    if  self.save_jpeg or self.save_mp4:
                        info = imgs_info[idx]
                        imgc = ImageMat(color_type=ColorType.BGR,info=info).unsafe_update_mat(img.copy())
                    frame_rec[i].append((frame_rec[i][-1][0]+1,imgc))
                    
                for i in self.all_cls_id - cls_id:
                    if frame_rec[i][-1][0]>0:
                        frame_rec[i].append((frame_rec[i][-1][0]-1,None))

                for k,q in frame_rec.items():
                    res = q[-self.ignore_frames_thre-1][0]>self.detect_frames_thre
                    if not res: continue

                    for i in range(-self.ignore_frames_thre,0,-1):
                        res = res and q[i-1][0] > q[i][0]
                        
                    if res:# vertify 
                        # do saving
                        print('save_jpeg',)
                        if  self.save_jpeg or self.save_mp4:
                            to_save=[q.pop()[1] for _ in range(self.queue_len)][::-1]
                            to_save=[i for i in to_save if i is not None]
                            if self.save_jpeg_gps:
                                for i in to_save:
                                    latlon=self._gps_processor.get_latlon()
                                    if latlon:
                                        i.info.latlon = (latlon[0],latlon[1])
                            self.frame_save_queue.put((self.class_names[k],to_save))
                        # clear up
                        for _ in range(self.queue_len):
                            q.append((0,None))

            return imgs_data
        
        def release(self):
            self.frame_save_queue.put(None)
            return super().release()
    
    class DrawYOLO(ImageMatProcessor):
        title:str = 'draw_yolo'
        draw_box_color: Tuple[int, int, int] = Field((0, 255, 0), description="Bounding box color (B, G, R)")
        draw_text_color: Tuple[int, int, int] = Field((255, 255, 255), description="Label text color (B, G, R)")
        draw_font_scale: float = Field(0.5, description="Font scale for label text")
        draw_thickness: int = Field(2, description="Line thickness for box and text")
        class_names:Dict[str,str] = {}
        class_colors_code:dict = {0: "FF3838",1: "FF9D97",2: "FF701F",3: "FFB21D",4: "CFD231",5: "48F90A",
                            6: "92CC17",7: "3DDB86",8: "1A9334",9: "00D4BB",10: "2C99A8",11: "00C2FF",12: "344593"}
        class_colors:Dict = {}
        yolo_uuid:str=''
        _yolo_processor:'Processors.YOLO'= None

        @staticmethod
        def jp2en(text):
            try:                
                import pykakasi
                kks = pykakasi.kakasi()
                text = kks.convert(text)
            except:
                pass
            return ''.join([i['hepburn'].replace('mono','butsu')  for i in text])
        
        @staticmethod
        def hex_to_bgr(hex_str: str):
            hex_str = hex_str.lstrip('#')
            return (int(hex_str[4:6], 16), int(hex_str[2:4], 16), int(hex_str[0:2], 16))

        def model_post_init(self, context):
            self.class_colors = {k:self.hex_to_bgr(v) for k,v in self.class_colors_code.items()}
            return super().model_post_init(context)

        def validate_img(self, img_idx, img):
            img.require_np_uint()
            img.require_BGR()
            img.require_HWC()
        
        def forward_raw(self, imgs_data:list[np.ndarray], imgs_info = ..., meta=...):
            res = []
                
            if self.yolo_uuid:
                self._yolo_processor = meta[self.yolo_uuid]
                if len(self.class_names)!=len(self._yolo_processor.class_names):
                    self.class_names = {k:self.jp2en(v) for k,v in self._yolo_processor.class_names.items()}

            for idx, img in enumerate(imgs_data):
                res.append(self.draw(img,
                                     self._yolo_processor.bounding_box_xyxy[idx],
                                     self.class_names,
                                     self.class_colors,))
            return res
            
        def draw(self, img: np.ndarray,
                detections: np.ndarray,
                class_names: Dict[str,str] = [],
                class_colors: Dict[str,str] = []) -> np.ndarray:
            """
            Draw YOLO-style bounding boxes and labels on the image.

            Parameters:
                img (np.ndarray): Image to draw on.
                detections (np.ndarray): Array of detections [x1, y1, x2, y2, conf, cls_id].
                class_names (List[str], optional): Class names for label display.
                class_colors: 0: "FF3838",1: "c58282",...
            Returns:
                np.ndarray: Annotated image.
            """
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
                cls_id = int(cls_id)

                # Compose label text
                if class_names and 0 <= cls_id < len(class_names):
                    label = f"{class_names[cls_id]} {conf:.2f}"
                else:
                    label = f"ID {cls_id} {conf:.2f}"

                box_color = class_colors[cls_id] if len(class_colors)>0 else (0,255,0)

                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    self.draw_thickness
                )

                # First: draw text with thicker stroke in edge color (e.g., black)
                cv2.putText(
                    img,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.draw_font_scale,
                    (0, 0, 0),  # Black edge
                    self.draw_thickness + 2,  # Thicker for outline
                    cv2.LINE_AA
                )

                cv2.putText(
                    img,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.draw_font_scale,
                    box_color,  # Use same color for text or choose another
                    self.draw_thickness,
                    cv2.LINE_AA
                )
            return img

    class CvImageViewer(ImageMatProcessor):

        title:str = 'cv_image_viewer'
        window_name_prefix: str = Field(default='ImageViewer', description="Prefix for window name")
        resizable: bool = Field(default=False, description="Whether window is resizable")
        scale: Optional[float] = Field(default=None, description="Scale factor for displayed image")
        overlay_texts: List[str] = Field(default_factory=list, description="Text overlays for images")
        save_on_key: Optional[int] = Field(default=ord('s'), description="Key code to trigger image save")
        window_names:list[str] = []
        mouse_pos:tuple[int,int] = (0, 0)  # for showing mouse coords

        yolo_uuid: Optional[str] = Field(default=None, description="UUID key to fetch YOLO results from meta")
        _yolo_processor: 'Processors.YOLO' = None
        draw_text_color: tuple = (255, 255, 255)  # White
        draw_font_scale: float = 0.5
        draw_thickness: int = 2
        draw_yolo: Optional['Processors.DrawYOLO'] = None

        def model_post_init(self, context):
            self.draw_yolo = Processors.DrawYOLO()
            return super().model_post_init(context)

        def validate_img(self, img_idx, img: ImageMat):
            img.require_ndarray()
            img.require_np_uint()
            win_name = f'{self.window_name_prefix}:{img_idx}'
            self.window_names.append(win_name)
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL if self.resizable else cv2.WINDOW_AUTOSIZE)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:            
            if self.yolo_uuid:
                self._yolo_processor = meta[self.yolo_uuid]

            scale = self.scale
            overlay_texts = self.overlay_texts

            for idx, img in enumerate(imgs_data):
                img = img.copy()

                # Optional overlay text
                text = overlay_texts[idx] if idx < len(overlay_texts) else ""
                if text:
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, self.draw_text_color, self.draw_thickness, cv2.LINE_AA)

                # Optional YOLO detection overlays
                if self._yolo_processor and idx<len(self._yolo_processor.bounding_box_xyxy):
                    self.draw_yolo.draw(img,self._yolo_processor.bounding_box_xyxy[idx],self._yolo_processor.class_names)
                
                win_name = self.window_names[idx]

                if scale is not None:
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                cv2.imshow(win_name, img)
                self.cv2_keys(idx,img)
            return imgs_data
        
        def cv2_keys(self,idx,img):
            key = cv2.waitKey(1) & 0xFF
            if self.save_on_key and key == self.save_on_key:
                filename = f'image_{idx}.png'
                cv2.imwrite(filename, img)
                logger(f'Saved {filename}')
            elif key == ord('e'):
                new_text = input(f"Enter new overlay text for image {idx}: ")
                if idx < len(self.overlay_texts):
                    self.overlay_texts[idx] = new_text
                else:
                    self.overlay_texts.append(new_text)

        def release(self):
            try:
                [cv2.destroyWindow(n) for n in self.window_names]
            except Exception:
                pass

        def __del__(self):
            self.release()

    class YOLOTRT(YOLO):
        title:str = 'YOLO_TRT_detections'
        modelname: str = 'yolov5s6u.engine'
        use_official_predict:bool = False
        conf: float = 0.6

        def build_models(self):
            self._models = {}
            for i,d in enumerate(self.num_devices):
                with torch.cuda.device(d):
                    modelname = self.modelname.replace('.trt',(f'_{d}.trt').replace(':','@'))
                    yolo = GeneralTensorRTInferenceModel(modelname,d,'images','output0')
                self._models[d] = yolo                
            self.use_official_predict = False            

        def predict(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo]=[]):
            yolo_results = []
            for i,img in enumerate(imgs_data):
                info = imgs_info[i]
                yolo_model = self._models[info.device]
                preds = yolo_model(img)
                preds = ops.non_max_suppression(
                    preds,
                    self.min_conf,
                    self.nms_iou,
                    classes = None,
                    agnostic= False,
                    max_det = self.max_det,
                    nc      = len(self.class_names),
                    in_place= False,
                    end2end = False,
                    rotated = False,
                )
                if isinstance(preds, list):
                    yolo_results += preds
                else:
                    yolo_results.append(preds)
            yolo_results_xyxycc:list[np.ndarray] = [r.cpu().numpy() for r in yolo_results]
            # yolo_results_xyxycc:list[np.ndarray] = [np.empty((0, 6), dtype=np.float32) for _ in range(4)]
            return [],yolo_results_xyxycc
        
class ImageMatProcessors(BaseModel):
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
        try:
            for fn in pipes:
                imgs,meta = (fn.validate if validate else fn)(imgs,meta)
        except Exception as e:
            logger(fn.uuid,e)
            raise e
        return imgs,meta
        
    @staticmethod    
    def run(gen,
            pipes:list['ImageMatProcessor']=[],
            meta = {},validate_once=False):
        if isinstance(pipes, str):
            pipes = ImageMatProcessors.loads(pipes)
        for imgs in gen:
            ImageMatProcessors.run_once(imgs,meta,pipes,validate_once)
            if validate_once:return

    @staticmethod    
    def validate_once(gen,
            pipes:list['ImageMatProcessor']=[]):
        ImageMatProcessors.run(gen,pipes,validate_once=True)

    @staticmethod
    def worker(pipes_serialized):
        pipes = ImageMatProcessors.loads(pipes_serialized)
        imgs,meta = [],{}
        while True:
            for fn in pipes:
                imgs,meta = fn(imgs,meta)

    @staticmethod
    def run_async(pipes: list[ImageMatProcessor] | str):
        if isinstance(pipes, str):
            pipes_serialized = pipes
        else:
            pipes_serialized = ImageMatProcessors.dumps(pipes)
            
        p = multiprocessing.Process(target=ImageMatProcessors.worker, args=(pipes_serialized,))
        p.start()
        return p