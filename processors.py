
# Standard Library Imports
from collections import defaultdict
import enum
import json
import math
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from shmIO import NumpyUInt8SharedMemoryStreamIO
from ImageMat import ColorType, ImageMat, ImageMatGenerator, ImageMatInfo, ImageMatProcessor, ShapeType
from pydantic import BaseModel, Field
from PIL import Image

logger = print

def hex2rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA format."""
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    alpha = int(hex_color[6:8], 16) if len(hex_color) == 8 else 255
    return rgba + (alpha,)

class Processors:

    class DoingNothing(ImageMatProcessor):
        title:str='doing_nothing'
        def validate_img(self, img_idx, img):
            pass

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            return imgs_data
        
    class CropToDivisibleBy32(ImageMatProcessor):
        title: str = 'crop_to_divisible_by_32'
        hs:list[int] = []
        ws:list[int] = []
        def validate_img(self, img_idx, img):
            img.require_np_uint()
            img.require_HW_or_HWC()

        def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=None):
            return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo] = [], meta={}) -> List[np.ndarray]:
            if len(self.hs)==0:
                for img in imgs_data:
                    h, w = img.shape[:2]
                    new_h = h - (h % 32)
                    new_w = w - (w % 32)
                    self.hs.append(new_h)
                    self.ws.append(new_w)
            processed_imgs = []
            for img,h,w in zip(imgs_data,self.hs,self.ws):
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

        def validate_img(self, img_idx: int, img: ImageMat):
            img.require_np_uint()
            img.require_HW_or_HWC()

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:                
            
            padded_imgs = []
            for img,info in zip(imgs_data,imgs_info):
                C = info.C
                if C == 2:
                    # Grayscale image: pad HxW
                    pad_width = self.pad_width
                elif C == 3:
                    # Color image: pad HxW only, leave channel unchanged
                    pad_width = self.pad_width + ((0, 0),)
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
                    
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
            return super().validate(imgs, meta)
        
        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:
            if self.layout is None:
                self.layout = self._init_layout(self.input_mats)
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
                if self.quality is not None:
                    success, encoded = cv2.imencode('.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                                    int(self.quality)])
                else:
                    success, encoded = cv2.imencode('.jpeg', img)
                
                if not success:
                    raise ValueError("JPEG encoding failed.")

                encoded_images.append(encoded)
                return encoded_images

    class NumpyUInt8SharedMemoryWriter(ImageMatProcessor):
        title:str='np_uint8_shm_writer'
        writers:list[NumpyUInt8SharedMemoryStreamIO.StreamWriter] = []
        stream_key_prefix:str

        def validate_img(self, img_idx, img: ImageMat):
            img.require_ndarray()
            img.require_np_uint()
            stream_key = f"{self.stream_key_prefix}:{img_idx}"
            wt = NumpyUInt8SharedMemoryStreamIO.writer(stream_key, img.data().shape)
            wt.build_buffer()

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:            
            for wt,img in zip(self.writers,imgs_data):
                wt.write(img)
            return imgs_data

    class CvVideoRecorder(ImageMatProcessor):
        output_filename: str = "output.avi",
        codec: str = 'XVID',
        fps: int = 30,
        scale: Optional[float] = None,
        overlay_text: Optional[str] = None
        _writer = None
        recording:bool = False
        frame_size:Optional[Tuple[int, int]] = None

        def validate_img(self, img_idx, img: ImageMat):
            if img_idx>0:
                raise RuntimeError("CvVideoRecorder cannot process multiple images at once")
            img.require_ndarray()
            img.require_np_uint()
            self.start(img.data().shape)

        def forward_raw(self, imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[np.ndarray]:            
            for img in imgs_data:self.write_frame(img)
            return imgs_data
        
        def start(self, frame_shape: Tuple[int, int]):
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.frame_size = (frame_shape[1], frame_shape[0])  # (width, height)
            self._writer = cv2.VideoWriter(self.output_filename, fourcc, self.fps, self.frame_size)
            self.recording = True
            logger(f"Started recording to {self.output_filename}")

        def stop(self):
            if self._writer:
                self._writer.release()
                self._writer = None
            self.recording = False
            logger("Stopped recording.")

        def write_frame(self, frame: np.ndarray):
            if not self.recording:
                raise RuntimeError("Recording has not started. Call start() first.")

            # Resize frame if needed
            if self.scale is not None:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)

            # Overlay text
            if self.overlay_text:
                cv2.putText(frame, self.overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Ensure frame matches output size and type
            if self.frame_size is not None and (frame.shape[1], frame.shape[0]) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            if frame.dtype != np.uint8 or (frame.ndim != 3 or frame.shape[2] != 3):
                raise ValueError("Frame must be uint8 BGR for VideoWriter")

            self._writer.write(frame)

        def __del__(self):
            try:
                self.stop()
            except Exception:
                pass

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

            self.out_mats = [ImageMat(color_type=i.info.color_type).build(img)
                for i,img in zip(out_mats,converted_raw_imgs)]
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
                
        def forward_raw_yolo(self,sw_yolo_proc: 'Processors.YOLOv5'):
            detections_per_window = sw_yolo_proc.bounding_box_xyxy
            transforms = self._sliding_window_splitter.pixel_idx_backward_T

            merged_detections = [np.empty((0, 6), dtype=np.float32)
                    for i in range(len(self._sliding_window_splitter.input_mats))]
            
            if len(detections_per_window) != len(transforms):
                raise ValueError(f"Number of detections_per_window({len(detections_per_window)}) and transforms({len(transforms)}) must match.")

            transform_detections = [None]*len(transforms)

            for i, (dets, T) in enumerate(zip(detections_per_window, transforms)):
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
                for tile_idx, (x1, y1, x2, y2) in zip(tile_indices, offsets):pass
                color_types.append(validated_imgs[tile_idx].info.color_type)
                
            self.out_mats = [ImageMat(color_type=color_type if color_type is None else color_type
                ).build(img)
                for color_type,img in zip(color_types,converted_raw_imgs)]
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
                for tile_idx, (x1, y1, x2, y2) in zip(tile_indices, offsets):
                    merged[y1:y2, x1:x2] = imgs_data[tile_idx]
                merged_imgs[img_idx] = merged

            if self.yolo_uuid:
                yolo:Processors.YOLOv5 = meta[self.yolo_uuid]
                yolo.bounding_box_xyxy = self.forward_raw_yolo(yolo)

            return merged_imgs

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
            from ultralytics import YOLO
            tmp_model = YOLO(self.modelname, task='detect')
            if not hasattr(tmp_model, 'names'):
                self.class_names = self.class_names if self.class_names is not None else default_names
            else:
                self.class_names = tmp_model.names
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
            from ultralytics import YOLO
            for d in self.num_devices:
                if d not in self._models:
                    self._models[d] = YOLO(self.modelname, task='detect').to(d)
                    if not self.use_official_predict:
                        self._models[d] = self._models[d].type(self._torch_dtype)

        def official_predict(self, imgs_data: List[Union[np.ndarray, torch.Tensor]], imgs_info: List[ImageMatInfo]=[]):
            imgs = []
            yolo_results = []
            for i,(img,info) in enumerate(zip(imgs_data,imgs_info)):
                device = info.device
                yolo_result = self._models[device](img, conf=self.conf, verbose=self.yolo_verbose,
                                                   imgsz=self.imgsz, half=(self._torch_dtype==torch.float16))
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
            from ultralytics.utils import ops
            from ultralytics import YOLO
            yolo_results = []
            for img,info in zip(imgs_data,imgs_info):
                yolo_model:YOLO = self._models[info.device]
                preds, feature_maps = yolo_model.model(img)
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
        
    class CvImageViewer(ImageMatProcessor):

        class DrawYOLO(BaseModel):
            # === Drawing configuration (Pydantic-managed) ===
            draw_box_color: Tuple[int, int, int] = Field((0, 255, 0), description="Bounding box color (B, G, R)")
            draw_text_color: Tuple[int, int, int] = Field((255, 255, 255), description="Label text color (B, G, R)")
            draw_font_scale: float = Field(0.5, description="Font scale for label text")
            draw_thickness: int = Field(2, description="Line thickness for box and text")
            # === Main method ===
            def draw(
                self,
                img: np.ndarray,
                detections: np.ndarray,
                class_names: List[str] = []
            ) -> np.ndarray:
                """
                Draw YOLO-style bounding boxes and labels on the image.

                Parameters:
                    img (np.ndarray): Image to draw on.
                    detections (np.ndarray): Array of detections [x1, y1, x2, y2, conf, cls_id].
                    class_names (List[str], optional): Class names for label display.

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

                    # Draw bounding box
                    cv2.rectangle(
                        img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.draw_box_color,
                        self.draw_thickness
                    )

                    # Draw label
                    cv2.putText(
                        img,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.draw_font_scale,
                        self.draw_text_color,
                        self.draw_thickness,
                        cv2.LINE_AA
                    )

                return img

        title:str = 'cv_image_viewer'
        window_name_prefix: str = Field(default='ImageViewer', description="Prefix for window name")
        resizable: bool = Field(default=False, description="Whether window is resizable")
        scale: Optional[float] = Field(default=None, description="Scale factor for displayed image")
        overlay_texts: List[str] = Field(default_factory=list, description="Text overlays for images")
        save_on_key: Optional[int] = Field(default=ord('s'), description="Key code to trigger image save")
        window_names:list[str] = []
        mouse_pos:tuple[int,int] = (0, 0)  # for showing mouse coords

        yolo_uuid: Optional[str] = Field(default=None, description="UUID key to fetch YOLO results from meta")
        _yolo_processor: 'Processors.YOLOv5' = None
        draw_text_color: tuple = (255, 255, 255)  # White
        draw_font_scale: float = 0.5
        draw_thickness: int = 2
        draw_yolo: DrawYOLO = DrawYOLO()

        def validate_img(self, img_idx, img: ImageMat):
            img.require_ndarray()
            img.require_np_uint()
            win_name = f'{self.window_name_prefix}:{img_idx}'
            self.window_names.append(win_name)
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL if self.resizable else cv2.WINDOW_AUTOSIZE)

        def _draw_yolo(self, img: np.ndarray, yolo_detection: np.ndarray, class_names=[]) -> np.ndarray:
            return self.draw_yolo.draw(img,yolo_detection,class_names)
            # for det in yolo_detection:
                # x1, y1, x2, y2, conf, cls_id = det
                # label = f"{class_names[int(cls_id)]} {conf:.2f}"
                # # Draw box
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.draw_box_color, self.draw_thickness)
                # # Draw label
                # cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #             self.draw_font_scale, self.draw_text_color, self.draw_thickness, cv2.LINE_AA)
            # return img
        
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
                    self._draw_yolo(img,self._yolo_processor.bounding_box_xyxy[idx],self._yolo_processor.class_names)

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

        def __del__(self):
            try:
                [cv2.destroyWindow(n) for n in self.window_names]
            except Exception:
                pass

    class YoloTRT(YOLOv5):
        title:str = 'YOLOv5_TRT_detections'
        modelname: str = 'yolov5s6u.engine'

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