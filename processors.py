
# Standard Library Imports
import enum
import math
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import cv2
import numpy as np
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from shmIO import NumpyUInt8SharedMemoryStreamIO
from ImageMat import ImageMat, ImageMatProcessor

class CvDebayerBlock(ImageMatProcessor):
    def __init__(self, format=cv2.COLOR_BAYER_BG2BGR, save_results_to_meta=False):
        super().__init__('cv_debayer',save_results_to_meta)
        self.format = format
        self.save_results_to_meta = save_results_to_meta

    def get_output_color_type(self):
        """Determine output color type based on the OpenCV conversion format."""
        bayer_to_color_map = {
            cv2.COLOR_BAYER_BG2BGR: "BGR",
            cv2.COLOR_BAYER_GB2BGR: "BGR",
            cv2.COLOR_BAYER_RG2BGR: "BGR",
            cv2.COLOR_BAYER_GR2BGR: "BGR",
            cv2.COLOR_BAYER_BG2RGB: "RGB",
            cv2.COLOR_BAYER_GB2RGB: "RGB",
            cv2.COLOR_BAYER_RG2RGB: "RGB",
            cv2.COLOR_BAYER_GR2RGB: "RGB",
            cv2.COLOR_BAYER_BG2GRAY: "grayscale",
            cv2.COLOR_BAYER_GB2GRAY: "grayscale",
            cv2.COLOR_BAYER_RG2GRAY: "grayscale",
            cv2.COLOR_BAYER_GR2GRAY: "grayscale",
        }
        return bayer_to_color_map.get(self.format, "unknown")

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before debayering. Ensures that the color type is 'bayer' 
        and the images have 1 channel.
        Also updates input_mat_infos and out_mat_infos.
        """
        self.input_mats = [i for i in imgs]

        validated_imgs: List[ImageMat] = []
        for img in imgs:
            img.require_BAYER()
            # img.require_ndarray()
            img.require_np_uint()
            validated_imgs.append(img)

        # Perform debayering after validation
        output_color_type = self.get_output_color_type()        
        debayered_imgs = self.forward_raw([img.data() for img in imgs])
        self.out_mats = [ImageMat(i, color_type=output_color_type) for i in debayered_imgs]
        return self.forward(validated_imgs, meta)
    
    def forward_raw(self,imgs_data)->List[np.ndarray]:
        return [cv2.cvtColor(i,self.format) for i in imgs_data]

class TorchDebayerBlock(ImageMatProcessor):
    ### Define the `Debayer5x5` PyTorch Model
    # The `Debayer5x5` model applies a **5x5 convolution filter** to interpolate missing 
    # color information from a Bayer pattern.

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
            super(TorchDebayerBlock.Debayer5x5, self).__init__()
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

    def __init__(self, save_results_to_meta=False):
        super().__init__('torch_debayer',save_results_to_meta)
        self.num_devices = self.devices_info()  # Number of GPUs available for processing
        self.save_results_to_meta = save_results_to_meta
        self.debayer_models:List[TorchDebayerBlock.Debayer5x5] = []
        self.input_devices = []  # To track device of each input tensor

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """Validate input images and initialize debayer models."""
        self.input_mats = [i for i in imgs]
        self.input_devices = []

        for img in self.input_mats:
            img.require_torch_tensor()
            img.require_BCHW()
            img.require_BAYER()

            # Save input device for tracking
            self.input_devices.append(img.info.device)

            # Initialize and store model on the corresponding device
            model = TorchDebayerBlock.Debayer5x5().to(img.info.device).to(img.info.dtype)
            self.debayer_models.append(model)
    
    
        # Perform debayering after validation
        debayered_imgs = self.forward_raw([img.data() for img in imgs])
        self.out_mats = [ImageMat(i, color_type="RGB") for i in debayered_imgs]
        processed_imgs, meta = self.forward(imgs, meta)
        return processed_imgs, meta
    
    def forward_raw(self,imgs_data:List[torch.Tensor])->List[torch.Tensor]:
        debayered_imgs = []
        for i, img in enumerate(imgs_data):
            model = self.debayer_models[i % len(self.debayer_models)]  # Fetch model from pre-assigned list
            debayered_imgs.append(model(img))
        return debayered_imgs

class TorchRGBToNumpyBGRBlock(ImageMatProcessor):
    def __init__(self, save_results_to_meta=False):
        super().__init__('torch_rgb_to_numpy_bgr',save_results_to_meta)
        self.num_devices = self.devices_info()  # Number of available GPUs
        self.save_results_to_meta = save_results_to_meta

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before conversion.
        Ensures that they are PyTorch tensors in RGB format with BCHW shape.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            img.require_torch_tensor()
            img.require_BCHW()
            img.require_RGB()
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type="BGR") for img in converted_imgs]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Converts a batch of RGB tensors (torch.Tensor) to BGR images (NumPy).
        """
        bgr_images = []
        for img in imgs_data:
            if img.device.type != 'cpu':
                img = img.cpu()  # Move tensor to CPU before conversion

            img = img.squeeze(0).permute(1, 2, 0).numpy()  # Convert BCHW to HWC
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)  
            # Normalize if needed
            img = img[:, :, ::-1]  # Convert RGB to BGR
            bgr_images.append(img)
        return bgr_images

class NumpyBGRToTorchRGBBlock(ImageMatProcessor):
    def __init__(self, dtype=None, save_results_to_meta=False,gpu=True,multi_gpu=-1):
        super().__init__('numpy_bgr_to_torch_rgb',save_results_to_meta)
        self.num_devices = self.devices_info(gpu=gpu,multi_gpu=multi_gpu)
        self.dtype = dtype if dtype else torch.float32
        self.save_results_to_meta = save_results_to_meta

        # Model function to convert BGR numpy to Torch RGB tensor
        def get_model(device, dtype=self.dtype):
            return lambda img: torch.tensor(img[:, :, ::-1].copy(), dtype=dtype, device=device
                                            ).div(255.0).permute(2, 0, 1).unsqueeze(0)

        self.tensor_models = []
        for device in self.num_devices:
            model = get_model(device)
            self.tensor_models.append((model, device))

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before conversion.
        Ensures they are NumPy BGR images with HWC shape.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            img.require_ndarray()
            img.require_HWC()
            img.require_BGR()
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type="RGB") for img in converted_imgs]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Converts a batch of BGR images (NumPy) to RGB tensors (Torch).
        """
        torch_images = []
        for i, img in enumerate(imgs_data):
            device = self.tensor_models[i % self.num_gpus][1] if self.num_gpus > 0 else 'cpu'
            tensor_img = torch.tensor(img[:, :, ::-1].copy(), dtype=self.dtype, device=device
                                      ).div(255.0).permute(2, 0, 1).unsqueeze(0)
            torch_images.append(tensor_img)
        return torch_images

class NumpyBayerToTorchBayerBlock(ImageMatProcessor):
    def __init__(self, dtype=None, save_results_to_meta=False,gpu=True,multi_gpu=-1):
        super().__init__('numpy_bayer_to_torch_bayer',save_results_to_meta)
        self.num_devices = self.devices_info(gpu=gpu,multi_gpu=multi_gpu)
        self.dtype = dtype if dtype else torch.float32
        self.save_results_to_meta = save_results_to_meta

        # Model function to convert Bayer numpy to Torch Bayer tensor
        def get_model(device, dtype=self.dtype):
            return lambda img: torch.tensor(img.copy(), dtype=dtype, device=device
                                    ).div(255.0).unsqueeze(0).unsqueeze(0)

        self.tensor_models = []
        for device in self.num_devices:
            model = get_model(device)
            self.tensor_models.append((model, device))

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before conversion.
        Ensures they are NumPy Bayer images with HW shape.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            img.require_ndarray()
            img.require_HW()
            img.require_BAYER()
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type="bayer") for img in converted_imgs]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Converts a batch of Bayer images (NumPy) to Bayer tensors (Torch).
        """
        torch_images = []
        for i, img in enumerate(imgs_data):
            device = self.tensor_models[i % self.num_gpus][1] if self.num_gpus > 0 else 'cpu'
            tensor_img = torch.tensor(img.copy(), dtype=self.dtype, device=device
                                    ).div(255.0).unsqueeze(0).unsqueeze(0)
            torch_images.append(tensor_img)
        return torch_images

class TorchResizeBlock(ImageMatProcessor):
    def __init__(self, target_size: Tuple[int, int], mode="bilinear", save_results_to_meta=False):
        """
        Initializes the TorchResizeBlock for resizing images.

        Args:
            target_size (Tuple[int, int]): (height, width) for the resized images.
            mode (str): Interpolation mode ('bilinear', 'nearest', 'bicubic', etc.).
            save_results_to_meta (bool): Whether to store resized images in metadata.
        """
        super().__init__('torch_resize')
        self.target_size = target_size
        self.mode = mode
        self.save_results_to_meta = save_results_to_meta

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before resizing.
        Ensures they are PyTorch tensors with BCHW shape.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs:List[ImageMat] = []

        for img in imgs:
            img.require_torch_tensor()
            img.require_BCHW()
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type=validated_imgs[i].info.color_type) for i,img in enumerate(converted_imgs)]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Resizes a batch of PyTorch images to the target size.
        """
        resized_images = []
        for img in imgs_data:
            resized_img = F.interpolate(img, size=self.target_size, mode=self.mode, align_corners=False)
            resized_images.append(resized_img)
        return resized_images

class CVResizeBlock(ImageMatProcessor):
    def __init__(self, target_size: Tuple[int, int], interpolation=cv2.INTER_LINEAR,
                  save_results_to_meta=False):
        """
        Initializes the CVResizeBlock for resizing images using OpenCV.

        Args:
            target_size (Tuple[int, int]): (height, width) for the resized images.
            interpolation (int): OpenCV interpolation method.
            save_results_to_meta (bool): Whether to store resized images in metadata.
        """
        super().__init__('cv_resize')
        self.target_size = target_size
        self.interpolation = interpolation
        self.save_results_to_meta = save_results_to_meta

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before resizing.
        Ensures they are NumPy images in HWC or HW format.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            img.require_ndarray()
            img.require_HW_or_HWC()
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type=validated_imgs[i].info.color_type
                                  ) for i,img in enumerate(converted_imgs)]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Resizes a batch of NumPy images using OpenCV.
        """
        resized_images = []
        for img in imgs_data:
            resized_img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                                      interpolation=self.interpolation)
            resized_images.append(resized_img)
        return resized_images
    
class TileNumpyImagesBlock(ImageMatProcessor):
    def __init__(self, tile_width: int, save_results_to_meta=False):
        super().__init__('tile_numpy_images', save_results_to_meta)
        self.tile_width = tile_width
        self.save_results_to_meta = save_results_to_meta
        self.layout = {}

    def _init_layout(self, imgs_data):
        """
        Calculate tiling grid parameters and create canvas.
        Returns:
            dict with keys: tile_width, tile_height, col_widths, row_heights,
            total_width, total_height, channels, canvas
        """
        num_images = len(imgs_data)
        tile_width = self.tile_width
        tile_height = math.ceil(num_images / tile_width)

        # Compute max width for each column, max height for each row
        col_widths = [0] * tile_width
        row_heights = [0] * tile_height

        for idx, img in enumerate(imgs_data):
            row, col = divmod(idx, tile_width)
            h, w = img.shape[:2]
            if w > col_widths[col]:
                col_widths[col] = w
            if h > row_heights[row]:
                row_heights[row] = h

        total_width = sum(col_widths)
        total_height = sum(row_heights)
        channels = imgs_data[0].shape[2] if imgs_data[0].ndim == 3 else 1

        if channels == 1:
            canvas = np.zeros((total_height, total_width), dtype=imgs_data[0].dtype)
        else:
            canvas = np.zeros((total_height, total_width, channels), dtype=imgs_data[0].dtype)

        return {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "col_widths": col_widths,
            "row_heights": row_heights,
            "total_width": total_width,
            "total_height": total_height,
            "channels": channels,
            "canvas": canvas
        }

    def validate(self, imgs: list[ImageMat], meta: dict = {}):
        if len(imgs) == 0:
            raise ValueError("No images to tile.")

        color_types = {i.info.color_type for i in imgs}
        if len(color_types) != 1:
            raise ValueError(f"All images must have the same color_type, got {color_types}")

        validated_imgs:list[ImageMat] = []
        for img in imgs:
            img.require_np_uint()
            img.require_HWC()
            validated_imgs.append(img)

        imgs_data = [img.data() for img in validated_imgs]
        self.layout = self._init_layout(imgs_data)
        tiled_imgs = self.forward_raw(imgs_data)
        self.out_mats = [ImageMat(tiled_imgs[0], color_type=imgs[0].info.color_type)]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: list[np.ndarray]) -> list[np.ndarray]:
        layout = self.layout
        tile_width = layout['tile_width']
        tile_height = layout['tile_height']
        col_widths = layout['col_widths']
        row_heights = layout['row_heights']
        channels = layout['channels']
        canvas = layout['canvas']

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


class EncodeNumpyToJpegBlock(ImageMatProcessor):
    def __init__(self, quality: int = None, save_results_to_meta=False):
        """
        Initializes the EncodeNumpyToJpegBlock for encoding images to JPEG.

        Args:
            quality (int, optional): JPEG quality (1-100). Defaults to OpenCV's standard.
            save_results_to_meta (bool): Whether to store encoded images in metadata.
        """
        super().__init__('encode_numpy_to_jpeg',save_results_to_meta)
        self.quality = quality
        self.save_results_to_meta = save_results_to_meta

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before encoding.
        Ensures they are NumPy images in HWC format.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            img.require_ndarray()
            img.require_HWC()
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        encoded_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [i.copy() for i in validated_imgs]
        for i,d in zip(self.out_mats,encoded_imgs):
            i.info.color_type = 'jpeg'
            i._img_data = d
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
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

class MergeYoloResultsBlock(ImageMatProcessor):
    def __init__(self,yolo_results_uuid):
        super().__init__('merge_yolo_results', True)
        self.yolo_results_uuid = yolo_results_uuid

    def forward(self, imgs: List[ImageMat], meta: Dict) -> Tuple[List[ImageMat], Dict]:
        """
        Merges YOLO detection results from multiple images.
        """
        # Retrieve YOLO results from meta
        results = meta.get(self.yolo_results_uuid, [])

        if not results:
            return imgs, meta  # No YOLO results to merge

        # If only one result, no need to merge
        if len(results) == 1:
            result = results[0]

        # If results contain bounding boxes (PyTorch format)
        elif hasattr(results[0], 'boxes'):
            boxes = torch.cat([res.boxes.data.cpu() for res in results])
            result = results[0].new()  # Create a new result object
            result.update(boxes=boxes)  # Update with merged bounding boxes

        # If results are NumPy arrays
        elif isinstance(results[0], np.ndarray):
            result = np.vstack(results)  # Stack NumPy arrays along first axis

        # Update meta with merged results
        meta[self.uuid] = result
        return imgs, meta

# TODO
class YOLOBlock(ImageMatProcessor):
    def __init__(
        self, 
        modelname: str = 'yolov5s6u.pt', 
        conf: float = 0.6, 
        cpu: bool = False, 
        names: Optional[Dict[int, str]] = None,
        save_results_to_meta: bool = False
    ):
        super().__init__('YOLO_detections', save_results_to_meta=save_results_to_meta)
        self.modelname = modelname
        self.conf = conf
        self.cpu = cpu

        default_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.models = []

        tmp_model = YOLO(modelname, task='detect')        
        if not hasattr(tmp_model, 'names'):
            self.names = names if names is not None else default_names
        else:
            self.names = tmp_model.names

    def validate(self, imgs: List[ImageMat], meta: Optional[Dict] = None):
        if meta is None:
            meta = {}

        self.input_mats = [i for i in imgs]
        devices = {i.info.device for i in imgs}
        models = {d: YOLO(self.modelname, task='detect').to(d) for d in devices}
        self.models = []

        for img in imgs:
            if img.is_ndarray():
                img.require_ndarray()
                img.require_HWC()
                img.require_RGB()
                model = models[img.info.device]
                det_fn = lambda x, m=model: m(x, conf=self.conf, verbose=False)
                self.models.append(det_fn)
            elif img.is_torch_tensor():
                img.require_torch_tensor()
                img.require_BCHW()
                img.require_RGB()
                model = models[img.info.device]
                det_fn = lambda x, m=model: m([*x], conf=self.conf, verbose=False)
                self.models.append(det_fn)
            else:
                raise TypeError("Unsupported image type for YOLOBlock")

        self.out_mats = [img.copy() for img in imgs]
        return self.forward(imgs, meta)

    def forward_raw(self, imgs_data: List[Union[np.ndarray, torch.Tensor]]):
        results = []
        for fn, img in zip(self.models, imgs_data):
            dets = fn(img)
            if isinstance(dets, list) and len(dets) == 1:
                dets = dets[0]
            results.append(dets)
        return results

    def forward(self, imgs: List[ImageMat], meta: Optional[Dict] = None):
        if meta is None:
            meta = {}
        meta['class_names'] = self.names
        img_data = [imgmat.data() for imgmat in imgs]
        meta[self.title] = self.forward_raw(img_data)

        return imgs, meta

    def _torch_transform(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # Accept np.ndarray or torch.Tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)  # HWC -> CHW
        elif img.ndim == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

# TODO
class YoloRTBlock(YOLOBlock):
    def __init__(
        self, 
        modelname: str = 'yolov5s6u.engine', 
        conf: float = 0.6, 
        cpu: bool = False, 
        names: Optional[Dict[int, str]] = None,
        save_results_to_meta: bool = False
    ):
        super().__init__(modelname, conf, cpu, names, save_results_to_meta)
        self.title = 'YOLO_RT_detections'

    def _torch_transform(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        img = super()._torch_transform(img)
        # Use float16 for TensorRT models
        if isinstance(img, torch.Tensor):
            img = img.to(torch.float16)
        return img

class NumpyUInt8SharedMemoryWriter(ImageMatProcessor):
    def __init__(self, stream_key_prefix: str):
        super().__init__('np_uint8_shm_writer')
        self.writers:list[NumpyUInt8SharedMemoryStreamIO.StreamWriter] = []
        self.stream_key_prefix = stream_key_prefix

    def validate_img(self, img_idx, img: ImageMat):
        img.require_ndarray()
        img.require_np_uint()
        stream_key = f"{self.stream_key_prefix}:{img_idx}"
        wt = NumpyUInt8SharedMemoryStreamIO.writer(stream_key, img.data().shape)
        wt.build_buffer()

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        for wt,img in zip(self.writers,imgs_data):
            wt.write(img)
        return imgs_data

class CvImageViewer(ImageMatProcessor):
    def __init__(self, window_name_prefix: str = 'ImageViewer',
                 resizable: bool = True,    
                 scale: Optional[float] = None,
                 overlay_texts: Optional[List[str]] = None,
                 save_on_key: Optional[int] = ord('s'),
                 ):
        super().__init__('cv_image_viewer')
        self.window_name_prefix = f'{window_name_prefix}:{uuid.uuid4()}'
        self.resizable = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
        self.scale = scale
        self.overlay_texts = overlay_texts or []
        self.save_on_key = save_on_key
        self.window_names = []
        self.mouse_pos = (0, 0)  # for showing mouse coords

    def validate_img(self, img_idx, img: ImageMat):
        img.require_ndarray()
        img.require_np_uint()
        win_name = f'{self.window_name_prefix}:{img_idx}'
        self.window_names.append(win_name)
        cv2.namedWindow(win_name, self.resizable)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        scale = self.scale
        overlay_texts = self.overlay_texts
        save_on_key = self.save_on_key

        for idx,img in enumerate(imgs_data):            
            img = imgs_data[idx].copy()
            if scale is not None:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            # Overlay text
            text = overlay_texts[idx] if idx < len(overlay_texts) else ""
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            win_name = self.window_names[idx]
            cv2.imshow(win_name, img)
            key = cv2.waitKey(1) & 0xFF

            if save_on_key and key == save_on_key:
                filename = f'image_{idx}.png'
                cv2.imwrite(filename, img)
                print(f'Saved {filename}')
            elif key == ord('e'):  # Edit overlay text
                new_text = input(f"Enter new overlay text for image {idx}: ")
                if idx < len(overlay_texts):
                    overlay_texts[idx] = new_text
                else:
                    overlay_texts.append(new_text)

        return imgs_data

    def __del__(self):
        try:
            [cv2.destroyWindow(n) for n in self.window_names]
        except Exception:
            pass

class CvVideoRecorder(ImageMatProcessor):
    def __init__(self, 
                 output_filename: str = "output.avi",
                 codec: str = 'XVID',
                 fps: int = 30,
                 scale: Optional[float] = None,
                 overlay_text: Optional[str] = None):
        super().__init__('cv_video_recorder')        
        self.output_filename = output_filename
        self.codec = codec
        self.fps = fps
        self.scale = scale
        self.overlay_text = overlay_text or ""

        self.writer = None
        self.recording = False
        self.frame_size = None

    def validate_img(self, img_idx, img: ImageMat):
        if img_idx>0:
            raise RuntimeError("CvVideoRecorder cannot process multiple images at once")
        img.require_ndarray()
        img.require_np_uint()
        self.start(img.data().shape)

    def forward_raw(self, imgs: list[np.ndarray])->list[np.ndarray]:
        for img in imgs:self.write_frame(img)
        return imgs
    
    def start(self, frame_shape: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.frame_size = (frame_shape[1], frame_shape[0])  # (width, height)
        self.writer = cv2.VideoWriter(self.output_filename, fourcc, self.fps, self.frame_size)
        self.recording = True
        print(f"Started recording to {self.output_filename}")

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        print("Stopped recording.")

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

        self.writer.write(frame)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass




