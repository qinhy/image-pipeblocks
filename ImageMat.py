
# Standard Library Imports
import enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import numpy as np
from pydantic import BaseModel
import torch

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
        if self._img_data.dtype != np.uint8:
            raise TypeError(f"Image data must be np.uint8. Got {self._img_data.dtype}")

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

    @staticmethod    
    def run_once(imgs,meta={},
            pipes:list['ImageMatProcessor']=[],
            validate=False):
        if validate:
            for fn in pipes:
                imgs,meta = fn.validate(imgs,meta)
        else:
            for fn in pipes:
                imgs,meta = fn(imgs,meta)
        return imgs,meta

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
    
    def validate_img(self, img_idx, img):
        raise NotImplementedError()

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for i,img in enumerate(imgs):
            self.validate_img(i,img)
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.out_mats = [ImageMat(img, color_type=old.info.color_type
                                  ) for old,img in zip(validated_imgs,converted_imgs)]        
        return self.forward(validated_imgs, meta)
    
    def forward_raw(self, imgs: List[Any]) -> List["Any"]:
        raise NotImplementedError()
    
    def forward(self, imgs: List[ImageMat], meta: Dict) -> Tuple[List[ImageMat],Dict]:
        forwarded_imgs = self.forward_raw([img.data() for img in imgs])
        output_imgs = [self.out_mats[i].unsafe_update_mat(forwarded_imgs[i]) for i in range(len(forwarded_imgs))]        
        return output_imgs, meta

    def get_converted_imgs(self, meta: Dict = {}):
        """Retrieve forwarded images from metadata if stored."""
        return meta.get(self.uuid, None)

class ImageMatGenerator:
    """
    Abstract generator for List[ImageMat].
    """
    def __init__(self, imgs: List[ImageMat]=None, meta: Dict=None):
        pass

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self) -> List['ImageMat']:
        raise NotImplementedError()

    def reset(self):
        pass

    def __len__(self):
        return None
    
    def __next_one_raw__(self) -> Any:
        raise NotImplementedError()

class ImageMatGenerator:
    """
    Abstract base class for generating lists of ImageMat objects from various sources.
    Manages shared resource lifecycle.
    """

    def __init__(self, sources: list = None, color_modes: list = None):
        self.sources = sources or []
        self.color_modes = color_modes or []
        self._resources = []  # General-purpose resource registry
        self.source_generators = [self.create_source_generator(src) for src in self.sources]

    def register_resource(self, resource):
        self._resources.append(resource)

    def release_resources(self):
        """
        Calls .release() on all registered resources if available.
        """
        for res in self._resources:
            if hasattr(res, "release") and callable(res.release):
                res.release()
        self._resources.clear()

    def create_source_generator(self, source):
        raise NotImplementedError("Subclasses must implement `create_source_generator`")

    def iterate_raw_images(self):
        for generator in self.source_generators:
            yield from generator

    def __iter__(self):
        return self

    def __next__(self):
        for raw_image, color_mode in zip(self.iterate_raw_images(), self.color_modes):
            yield ImageMat(raw_image, color_mode)

    def reset_generators(self):
        pass

    def release(self):
        self.release_resources()

    def __del__(self):
        self.release()

    def __len__(self):
        return None
