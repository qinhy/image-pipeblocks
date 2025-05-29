
# Standard Library Imports
import enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import cv2
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from ultralytics import YOLO

class ShapeType(str, enum.Enum):
    HWC = 'HWC'
    HW = 'HW'
    BCHW = 'BCHW'

class ColorType(str, enum.Enum):
    BAYER = 'bayer'
    GRAYSCALE = 'grayscale'
    RGB = 'RGB'
    BGR = 'BGR'
    JPEG = 'jpeg'

COLOR_TYPE_CHANNELS = {
    ColorType.BAYER: [1],
    ColorType.GRAYSCALE: [1],
    ColorType.RGB: [3],
    ColorType.BGR: [3],
}

class ImageMatInfo(BaseModel):
    type: Optional[str] = None
    dtype: Optional[Union[np.dtype, torch.dtype]] = None
    device: str = ''
    shape_type: Optional[ShapeType] = None
    max_value: Optional[Union[int, float]] = None
    B: Optional[int] = None
    C: Optional[int] = None
    H: int = 0
    W: int = 0
    color_type: Optional[ColorType] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, img_data: Union[np.ndarray, torch.Tensor],
                  color_type: Union[str, ColorType], *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parse color_type to Enum
        try:
            color_type = ColorType(color_type)
        except ValueError:
            raise ValueError(
                f"Invalid color type: {color_type}. Must be one of {[c.value for c in ColorType]}")

        self.type = type(img_data).__name__

        if isinstance(img_data, np.ndarray):
            self.dtype = img_data.dtype
            self.device = "cpu"
            self.max_value = 255 if img_data.dtype == np.uint8 else 1.0

            if img_data.ndim == 3:
                self.shape_type = ShapeType.HWC
                self.H, self.W, self.C = img_data.shape
            elif img_data.ndim == 2:
                self.shape_type = ShapeType.HW
                self.H, self.W = img_data.shape
                self.C = 1
            else:
                raise ValueError("NumPy array must be 2D (HW) or 3D (HWC).")

        elif isinstance(img_data, torch.Tensor):
            self.dtype = img_data.dtype
            self.device = str(img_data.device)
            self.max_value = 1.0

            if img_data.ndim == 4:
                self.shape_type = ShapeType.BCHW
                self.B, self.C, self.H, self.W = img_data.shape
            else:
                raise ValueError("Torch tensor must be 4D (BCHW).")

        else:
            raise TypeError(f"img_data must be np.ndarray or torch.Tensor, got {type(img_data)}")

        # Channel count validation
        if color_type in COLOR_TYPE_CHANNELS:
            expected_channels = COLOR_TYPE_CHANNELS[color_type]
            if self.C not in expected_channels:
                raise ValueError(
                    f"Invalid color type '{color_type.value}' for image with {self.C} channels. "
                    f"Expected {expected_channels} channels. Data shape: {img_data.shape}"
                )
        self.color_type = color_type

class ImageMat:
    def __init__(self, img_data: Union[np.ndarray, torch.Tensor], 
                 color_type: Union[str, ColorType], info: Optional[ImageMatInfo] = None):
        if img_data is None:
            raise ValueError("img_data cannot be None")
        self.info = info or ImageMatInfo(img_data, color_type=color_type)
        self._img_data = img_data

    def copy(self) -> 'ImageMat':
        """Return a deep copy of the ImageMat object."""
        if isinstance(self._img_data, np.ndarray):
            return ImageMat(self._img_data.copy(), color_type=self.info.color_type)
        elif isinstance(self._img_data, torch.Tensor):
            return ImageMat(self._img_data.clone(), color_type=self.info.color_type)
        else:
            raise TypeError("img_data must be np.ndarray or torch.Tensor")

    def update_mat(self, img_data: Union[np.ndarray, torch.Tensor]) -> 'ImageMat':
        """Update image data and refresh metadata."""
        self._img_data = img_data
        self.info = ImageMatInfo(img_data, color_type=self.info.color_type)
        return self

    def unsafe_update_mat(self, img_data: Union[np.ndarray, torch.Tensor]) -> 'ImageMat':
        """Update the image data without updating metadata (use with caution)."""
        self._img_data = img_data
        return self

    def data(self) -> Union[np.ndarray, torch.Tensor]:
        """Return the image data."""
        return self._img_data

    # --- Type/Shape/Color Requirement Methods ---
    def is_ndarray(self):
        return isinstance(self._img_data, np.ndarray)
    
    def is_torch_tensor(self):
        return isinstance(self._img_data, torch.Tensor)

    def require_ndarray(self):
        if not isinstance(self._img_data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(self._img_data)}")
        self.require_HW_or_HWC()

    def require_np_uint(self):
        self.require_ndarray()
        if self._img_data.dtype not in (np.uint8, np.uint16):
            raise TypeError("Image data must be np.uint8 or np.uint16.")

    def require_torch_tensor(self):
        if not isinstance(self._img_data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(self._img_data)}")
        self.require_BCHW()

    def require_shape_type(self, shape_type: ShapeType):
        if self.info.shape_type != shape_type:
            raise TypeError(f"Expected {shape_type.value} format, got {self.info.shape_type}")

    def require_color_type(self, color_type: ColorType):
        if self.info.color_type != color_type:
            raise TypeError(f"Expected {color_type.value} image, got {self.info.color_type}")

    def require_BAYER(self):      self.require_color_type(ColorType.BAYER)
    def require_GRAYSCALE(self):  self.require_color_type(ColorType.GRAYSCALE)
    def require_RGB(self):        self.require_color_type(ColorType.RGB)
    def require_BGR(self):        self.require_color_type(ColorType.BGR)
    def require_JPEG(self):       self.require_color_type(ColorType.JPEG)

    def require_HWC(self):        self.require_shape_type(ShapeType.HWC)
    def require_HW(self):         self.require_shape_type(ShapeType.HW)
    def require_HW_or_HWC(self):
        if self.info.shape_type not in {ShapeType.HWC, ShapeType.HW}:
            raise TypeError(f"Expected HWC or HW format, got {self.info.shape_type}")

    def require_BCHW(self):       self.require_shape_type(ShapeType.BCHW)

class ImageMatProcessor:    
    class MetaData(BaseModel):        
        model_config = {"arbitrary_types_allowed": True}

    def __init__(self,title='ImageMatProcessor', save_results_to_meta=False):
        self.title = title
        self.uuid = uuid.uuid4()
        self.save_results_to_meta = save_results_to_meta
        self._enable = True
        self.input_mats: List[ImageMat] = []
        self.out_mats: List[ImageMat] = []

    def on(self):
        self._enable = True

    def off(self):
        self._enable = False
        
    def devices_info(self,gpu=True,multi_gpu=-1):
        self.num_devices = ['cpu']
        if gpu and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            if multi_gpu <= 0 or multi_gpu > self.num_gpus:
                # Use all available GPUs
                self.num_devices = [f"cuda:{i}" for i in range(self.num_gpus)]
            else:
                # Use specified number of GPUs
                self.num_devices = [f"cuda:{i}" for i in range(multi_gpu)]
        return self.num_devices
    
    def __call__(self, imgs: List[ImageMat], meta: dict = {}):    
        if self._enable:
            output_imgs, meta = self.forward(imgs, meta)
            
        if self.save_results_to_meta:
            meta[self.uuid] = [i.copy() for i in output_imgs]

        return output_imgs, meta
    
    def validate(self, imgs: List[ImageMat], meta: dict = {}):
        self.input_mats = [i for i in imgs]
        imgs, meta = self.forward(imgs, meta)
        self.out_mats = [i for i in imgs]
        return imgs, meta
    
    def forward_raw(self, imgs: List[Any]) -> List["Any"]:
        raise NotImplementedError()
    
    def forward(self, imgs: List["ImageMat"], meta: Dict) -> Tuple[List["ImageMat"],Dict]:
        forwarded_imgs = self.forward_raw([img.data() for img in imgs])
        output_imgs = [self.out_mats[i].unsafe_update_mat(forwarded_imgs[i]) for i in range(len(forwarded_imgs))]        
        return output_imgs, meta

    def get_converted_imgs(self, meta: Dict = {}):
        """Retrieve forwarded images from metadata if stored."""
        return meta.get(self.uuid, None)

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
        """
        Initializes the TileNumpyImagesBlock for tiling images.

        Args:
            tile_width (int): Number of images per row in the tiled output.
            save_results_to_meta (bool): Whether to store tiled images in metadata.
        """
        super().__init__('tile_numpy_images',save_results_to_meta)
        self.tile_width = tile_width
        self.save_results_to_meta = save_results_to_meta

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before tiling.
        Ensures they are NumPy images in HWC format.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            img.require_ndarray()
            img.require_HWC()
            validated_imgs.append(img)

        # Create new ImageMat instance for output
        tiled_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(tiled_imgs[0], color_type=validated_imgs[0].info.color_type)]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Tiles a batch of NumPy images into a single large image.
        """
        # Get image dimensions
        h, w = imgs_data[0].shape[:2]
        num_images = len(imgs_data)
        tile_height = math.ceil(num_images / self.tile_width)  # Number of rows needed

        # Create a blank canvas
        tile = np.zeros((tile_height * h, self.tile_width * w, 3), dtype=imgs_data[0].dtype)

        # Place images in the tile
        for i, img in enumerate(imgs_data):
            r, c = divmod(i, self.tile_width)
            tile[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

        return [tile]

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
        # self.model = YOLO(modelname, task='detect')
        # if cpu:
        #     self.model = self.model.cpu()
        # if not hasattr(self.model, 'names'):
        #     self.names = names if names is not None else default_names
        # else:
        #     self.names = self.model.names

    def validate(self, imgs: List[ImageMat], meta: Optional[Dict] = None):
        if meta is None:
            meta = {}
        self.input_mats = [i for i in imgs]
        models = set([i.info.device for i in imgs])
        models = {i:YOLO(self.modelname, task='detect').to(i) for i in models}
        self.models = []
        
        for i,img in enumerate(imgs):

            if img.is_ndarray():                
                img.require_ndarray()
                img.require_HWC()
                img.require_RGB()
                model = models[img.info.device]
                # img is ndarray HWC RGB
                det_fn = lambda img:model(img, conf=self.conf, verbose=False)[0]
                self.models.append(det_fn)


            if img.is_torch_tensor():
                img.require_torch_tensor()
                img.require_BCHW()
                img.require_RGB()
                # img is torch_tensor BCHW RGB put [CHW ...]
                det_fn = lambda img:model([*img], conf=self.conf, verbose=False)
                self.models.append(det_fn)

            self.models.append(models[img.info.device])

        self.out_mats = [img.copy() for img in imgs]
        validated_imgs, meta = self.forward(imgs, meta)
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[Union[np.ndarray, torch.Tensor]]):
        for i,img in enumerate(imgs_data):
            # Run YOLO model (works for both batch and single)
            if isinstance(img, torch.Tensor) and img.ndim == 4:
                dets = self.models[i]([*img], conf=self.conf, verbose=False)
            else:
                dets = self.models[i](img, conf=self.conf, verbose=False)
            if isinstance(dets, list) and len(dets) == 1:
                dets = dets[0]
        return dets

    def forward(self, imgs: List[ImageMat], meta: Optional[Dict] = None):
        if meta is None:
            meta = {}
        meta['class_names'] = self.names
        meta[self.title] = []

        for imgmat in imgs:
            img_data = imgmat.data()
            detections = self.forward_raw(img_data)
            meta[self.title].append(detections)

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

    def _infer(self, img: Union[np.ndarray, torch.Tensor]):
        assert hasattr(img, 'shape'), 'image is not a valid array or tensor!'
        img = self._torch_transform(img)
        # Run YOLO model (works for both batch and single)
        if isinstance(img, torch.Tensor) and img.ndim == 4:
            dets = self.model([*img], conf=self.conf, verbose=False)
        else:
            dets = self.model(img, conf=self.conf, verbose=False)
        if isinstance(dets, list) and len(dets) == 1:
            dets = dets[0]
        return dets

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
