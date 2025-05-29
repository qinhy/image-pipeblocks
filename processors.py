
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

class ImageMatInfo(BaseModel):
    type: Optional[Union[np.ndarray, torch.Tensor]] = None
    dtype: Optional[Union[np.dtype, torch.dtype]] = None
    device: str = ''  # 'cpu' or 'cuda:n' , numpy is only cpu and torch.Tensor is only GPU
    shape_type: str = ''  # 'HWC' or 'HW' for numpy, 'BCHW' for torch
    max_value: Optional[Union[int, float]] = None # 255 for np, 1.0 for torch
    B: Optional[int] = None  # Batch dimension (only for torch)
    C: Optional[int] = None  # Channel dimension
    H: int = 0  # Height
    W: int = 0  # Width
    color_type: Optional[str] = None  # "Only 'bayer', 'grayscale', 'RGB', or 'BGR', 'jpeg'"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, img_data: Union[np.ndarray, torch.Tensor], color_type: str, *args, **kwargs):        
        super().__init__(*args,**kwargs)
        self.type = str(type(img_data)).replace("<class '","").replace("'>","")

        if isinstance(img_data, np.ndarray):
            self.dtype = img_data.dtype
            self.device = "cpu"  # NumPy only runs on CPU
            self.max_value = 255 if img_data.dtype == np.uint8 else 1.0

            if img_data.ndim == 3:
                self.shape_type = "HWC"
                self.H, self.W, self.C = img_data.shape
            elif img_data.ndim == 2:
                self.shape_type = "HW"
                self.H, self.W = img_data.shape
                self.C = 1
            else:
                raise ValueError("NumPy array must be HW or HWC format")

        elif isinstance(img_data, torch.Tensor):
            self.dtype = img_data.dtype
            self.device = str(img_data.device)
            self.max_value = 1.0  # Torch tensors are assumed to be normalized

            if img_data.ndim == 4:
                self.shape_type = "BCHW"
                self.B, self.C, self.H, self.W = img_data.shape
            else:
                raise ValueError("Torch tensor must be in BCHW format")

        else:
            raise TypeError(f"img_data must be a numpy array or torch tensor, got {type(img_data)}")


        # Validate color type with channel count
        valid_color_types = {
            "bayer": [1],
            "grayscale": [1],
            "RGB": [3],
            "BGR": [3],
        }

        if color_type not in valid_color_types:
            raise ValueError(f"Invalid color type: {color_type}. Must be one of {list(valid_color_types.keys())}")

        if self.C not in valid_color_types[color_type]:
            raise ValueError(
                f"Invalid color type '{color_type}' for image with {self.C} channels."
                f" Expected {valid_color_types[color_type]} channels."
                f" Data shape is {img_data.shape}."
            )

        self.color_type = color_type
        
class ImageMat:
    def __init__(self, img_data: Optional[Union[np.ndarray, torch.Tensor]], color_type: str, info=None):
        if img_data is None:
            raise ValueError("img_data cannot be None")
        if info is None:
            self.info = ImageMatInfo(img_data, color_type=color_type)
        self.img_data = img_data 
        
        """Return a deep copy of the ImageMat object."""
        if isinstance(self.img_data, np.ndarray):            
            self._copy = lambda : ImageMat(self.data().copy(),color_type=self.info.color_type)
        elif isinstance(self.img_data, torch.Tensor):
            self._copy = lambda : ImageMat(self.data().clone(),color_type=self.info.color_type)
        else:
            raise TypeError("img_data must be a numpy array or torch tensor")

    def copy(self):
        """Return a deep copy of the ImageMat object."""
        return self._copy()

    def update_mat(self, img_data: Union[np.ndarray, torch.Tensor]):
        """Update the image data and refresh metadata."""
        self.img_data = img_data
        self.info = ImageMatInfo(img_data, color_type=self.info.color_type)  # Preserve color type        
        return self

    def unsafe_update_mat(self, img_data: Union[np.ndarray, torch.Tensor]):
        """Update the image data without updating metadata (use with caution)."""
        self.img_data = img_data
        return self
    
    def data(self) -> Union[np.ndarray, torch.Tensor]:
        """Return the image data."""
        return self.img_data

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
        
    def gpu_info(self):        
        self.num_gpus = torch.cuda.device_count()
        # print(f'[{self.__class__.__name__}] Number of GPUs available: {self.num_gpus}')
        return self.num_gpus
    
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
        Validates input images before debayering. Ensures that the color type is 'bayer' and the images have 1 channel.
        Also updates input_mat_infos and out_mat_infos.
        """
        self.input_mats = [i for i in imgs]

        validated_imgs: List[ImageMat] = []
        for img in imgs:
            if img.info.color_type != "bayer":
                raise ValueError(f"Image color type must be 'bayer' but got '{img.info.color_type}'")

            if img.info.C != 1:
                raise ValueError(f"Image must have 1 channel for debayering, but got {img.info.C} channels.")

            if not isinstance(img.img_data, np.ndarray):
                raise TypeError("CvDebayerBlock only supports NumPy array images.")

            if img.img_data.dtype not in [np.uint8, np.uint16]:
                raise TypeError("Image data must be of type np.uint8 or np.uint16 for debayering.")

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
    # The `Debayer5x5` model applies a **5x5 convolution filter** to interpolate missing color information from a Bayer pattern.

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
        self.num_gpus = self.gpu_info()  # Number of GPUs available for processing
        self.save_results_to_meta = save_results_to_meta
        self.debayer_models:List[TorchDebayerBlock.Debayer5x5] = []
        self.input_devices = []  # To track device of each input tensor

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """Validate input images and initialize debayer models."""
        self.input_mats = [i for i in imgs]
        self.input_devices = []

        for i, info in enumerate([ii.info for ii in self.input_mats]):
            if 'torch.Tensor' != info.type:
                raise ValueError(f"Unsupported image type at index {i}: {info.type}. Expected torch.Tensor.")
            if info.shape_type != "BCHW":
                raise ValueError(f"Invalid shape type at index {i}: {info.shape_type}. Expected 'BCHW'.")
            if info.color_type != "bayer":
                raise ValueError(f"Invalid color type at index {i}: {info.color_type}. Expected 'bayer'.")

            # Ensure the tensor is already on the correct device
            device = torch.device(f'cuda:{i % self.num_gpus}' if torch.cuda.is_available() and self.num_gpus > 0 else 'cpu')
            if str(info.device) != str(device):
                raise ValueError(
                    f"Image at index {i} is on {info.device}, expected {device}. "
                    "Move it to the correct device before passing to forward()."
                )

            # Save input device for tracking
            self.input_devices.append(device)

            # Initialize and store model on the corresponding device
            model = TorchDebayerBlock.Debayer5x5().to(device).to(info.dtype)  # Assuming Processors.Debayer5x5() is defined elsewhere
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
        self.num_gpus = self.gpu_info()  # Number of available GPUs
        self.save_results_to_meta = save_results_to_meta

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        """
        Validates input images before conversion.
        Ensures that they are PyTorch tensors in RGB format with BCHW shape.
        """
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for img in imgs:
            if not isinstance(img.img_data, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, but got {type(img.img_data)}.")

            if img.info.shape_type != "BCHW":
                raise ValueError(f"Expected BCHW format, but got {img.info.shape_type}.")

            if img.info.color_type != "RGB":
                raise ValueError(f"Expected RGB image, but got {img.info.color_type}.")

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
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)  # Normalize if needed
            img = img[:, :, ::-1]  # Convert RGB to BGR
            bgr_images.append(img)
        return bgr_images

class NumpyBGRToTorchRGBBlock(ImageMatProcessor):
    def __init__(self, dtype=None, save_results_to_meta=False):
        super().__init__('numpy_bgr_to_torch_rgb',save_results_to_meta)
        self.num_gpus = self.gpu_info()
        self.dtype = dtype if dtype else torch.float32
        self.save_results_to_meta = save_results_to_meta

        # Model function to convert BGR numpy to Torch RGB tensor
        def get_model(device, dtype=self.dtype):
            return lambda img: torch.tensor(img[:, :, ::-1].copy(), dtype=dtype, device=device).div(255.0).permute(2, 0, 1).unsqueeze(0)

        self.tensor_models = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
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
            if not isinstance(img.img_data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, but got {type(img.img_data)}.")

            if img.info.shape_type != "HWC":
                raise ValueError(f"Expected HWC format, but got {img.info.shape_type}.")

            if img.info.color_type != "BGR":
                raise ValueError(f"Expected BGR image, but got {img.info.color_type}.")

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
            tensor_img = torch.tensor(img[:, :, ::-1].copy(), dtype=self.dtype, device=device).div(255.0).permute(2, 0, 1).unsqueeze(0)
            torch_images.append(tensor_img)
        return torch_images

class NumpyBayerToTorchBayerBlock(ImageMatProcessor):
    def __init__(self, dtype=None, save_results_to_meta=False):
        super().__init__('numpy_bayer_to_torch_bayer',save_results_to_meta)
        self.num_gpus = self.gpu_info()
        self.dtype = dtype if dtype else torch.float32
        self.save_results_to_meta = save_results_to_meta

        # Model function to convert Bayer numpy to Torch Bayer tensor
        def get_model(device, dtype=self.dtype):
            return lambda img: torch.tensor(img.copy(), dtype=dtype, device=device).div(255.0).unsqueeze(0).unsqueeze(0)

        self.tensor_models = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
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
            if not isinstance(img.img_data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, but got {type(img.img_data)}.")

            if img.info.shape_type != "HW":
                raise ValueError(f"Expected HW format, but got {img.info.shape_type}.")

            if img.info.color_type != "bayer":
                raise ValueError(f"Expected Bayer image, but got {img.info.color_type}.")

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
            tensor_img = torch.tensor(img.copy(), dtype=self.dtype, device=device).div(255.0).unsqueeze(0).unsqueeze(0)
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
            if not isinstance(img.img_data, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, but got {type(img.img_data)}.")

            if img.info.shape_type != "BCHW":
                raise ValueError(f"Expected BCHW format, but got {img.info.shape_type}.")

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
    def __init__(self, target_size: Tuple[int, int], interpolation=cv2.INTER_LINEAR, save_results_to_meta=False):
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
            if not isinstance(img.img_data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, but got {type(img.img_data)}.")

            if img.info.shape_type not in ["HWC", "HW"]:
                raise ValueError(f"Expected HWC or HW format, but got {img.info.shape_type}.")

            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type=validated_imgs[i].info.color_type) for i,img in enumerate(converted_imgs)]
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Resizes a batch of NumPy images using OpenCV.
        """
        resized_images = []
        for img in imgs_data:
            resized_img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=self.interpolation)
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
            if not isinstance(img.img_data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, but got {type(img.img_data)}.")

            if img.info.shape_type != "HWC":
                raise ValueError(f"Expected HWC format, but got {img.info.shape_type}.")

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
            if not isinstance(img.img_data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, but got {type(img.img_data)}.")

            if img.info.shape_type != "HWC":
                raise ValueError(f"Expected HWC format, but got {img.info.shape_type}.")

            validated_imgs.append(img)

        # Create new ImageMat instances for output
        encoded_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [i.copy() for i in validated_imgs]
        for i,d in zip(self.out_mats,encoded_imgs):
            i.info.color_type = 'jpeg'
            i.img_data = d
        return self.forward(validated_imgs, meta)

    def forward_raw(self, imgs_data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Encodes a batch of NumPy images to JPEG format.
        """
        encoded_images = []
        for img in imgs_data:
            if self.quality is not None:
                success, encoded = cv2.imencode('.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
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