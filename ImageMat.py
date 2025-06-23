
# Standard Library Imports
import enum
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import numpy as np
from pydantic import BaseModel
import torch

class DeviceType(str, enum.Enum):
    CPU = 'cpu'
    GPU = 'gpu'

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

# global setting
torch_img_dtype = torch.float16
numpy_img_dtype = np.uint8

class ImageMatInfo(BaseModel):

    class TorchDtype(BaseModel):
        is_floating_point: bool
        is_complex: bool
        is_signed: bool
        itemsize: int

    type: Optional[str] = None
    _dtype: Optional[Union[np.dtype, TorchDtype]] = None
    device: str = ''
    shape_type: Optional[ShapeType] = None
    max_value: Optional[Union[int, float]] = None
    B: Optional[int] = None
    C: Optional[int] = None
    H: int = 0
    W: int = 0
    color_type: Optional[ColorType] = None

    @staticmethod
    def torch_img_dtype():return torch_img_dtype
    @staticmethod
    def numpy_img_dtype():return numpy_img_dtype

    def build(self, img_data: Union[np.ndarray, torch.Tensor],
                  color_type: Union[str, ColorType]):
        # Parse color_type to Enum
        try:
            color_type = ColorType(color_type)
        except ValueError:
            raise ValueError(
                f"Invalid color type: {color_type}. Must be one of {[c.value for c in ColorType]}")

        self.type = type(img_data).__name__

        if isinstance(img_data, np.ndarray):
            self._dtype = img_data.dtype
            self.device = "cpu"
            self.max_value = 255 if img_data.dtype == numpy_img_dtype else 1.0

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
            self._dtype = img_data.dtype
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

        return self

class ImageMat(BaseModel):

    class TorchDtype(BaseModel):
        is_floating_point: bool
        is_complex: bool
        is_signed: bool
        itemsize: int

    info: Optional[ImageMatInfo] = None
    color_type: Union[str, ColorType]

    def build(self, img_data: Union[np.ndarray, torch.Tensor], info: Optional[ImageMatInfo] = None):
        if img_data is None:
            raise ValueError("img_data cannot be None")
        self.info = info or ImageMatInfo().build(img_data, color_type=self.color_type)
        self._img_data = img_data
        return self

    def copy(self) -> 'ImageMat':
        """Return a deep copy of the ImageMat object."""
        if isinstance(self._img_data, np.ndarray):
            return ImageMat(color_type=self.info.color_type).build(self._img_data.copy())
        elif isinstance(self._img_data, torch.Tensor):
            return ImageMat(color_type=self.info.color_type).build(self._img_data.clone())
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
        if self._img_data.dtype != numpy_img_dtype:
            raise TypeError(f"Image data must be {numpy_img_dtype}. Got {self._img_data.dtype}")

    def require_torch_tensor(self):
        if not isinstance(self._img_data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(self._img_data)}")
        self.require_BCHW()

    def require_torch_float(self):
        self.require_torch_tensor()
        if self._img_data.dtype != torch_img_dtype:
            raise TypeError(f"Image data must be {torch_img_dtype}. Got {self._img_data.dtype}")
        
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

class ImageMatProcessor(BaseModel):
    class MetaData(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
    title:str
    uuid:str = ''
    save_results_to_meta:bool = False
    _enable:bool = True
    meta:dict = {}
    input_mats: List[ImageMat] = []
    out_mats: List[ImageMat] = []
    num_devices:list[str] = ['cpu']
    num_gpus:int = 0
    _enable:bool = True

    def model_post_init(self, context: Any, /) -> None:
        if len(self.title)==0:
            self.title = self.__class__.__name__
        self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'
        return super().model_post_init(context)
    
    def on(self):
        self._enable = True

    def off(self):
        self._enable = False
        
    def devices_info(self,gpu=True,multi_gpu=-1):
        self.num_devices = ['cpu']
        self.num_gpus = 0
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
    
    def validate_img(self, img_idx:int, img:ImageMat):
        raise NotImplementedError()

    def validate(self, imgs: List[ImageMat], meta: Dict = {}):
        self.input_mats = [i for i in imgs]
        validated_imgs: List[ImageMat] = []

        for i,img in enumerate(imgs):
            self.validate_img(i,img)
            validated_imgs.append(img)

        # Create new ImageMat instances for output
        converted_raw_imgs = self.forward_raw([img.data() for img in validated_imgs])
        self.input_mats = validated_imgs
        self.build_out_mats(validated_imgs,converted_raw_imgs)
        return self.forward(validated_imgs, meta)
    
    def build_out_mats(self,validated_imgs: List[ImageMat],converted_raw_imgs,color_type=None):
        # 1:1 mapping
        self.out_mats = [ImageMat(color_type=old.info.color_type if color_type is None else color_type
            ).build(img)
            for old,img in zip(validated_imgs,converted_raw_imgs)]
    
    def forward_raw(self, imgs: List[Any]) -> List["Any"]:
        raise NotImplementedError()
    
    def forward(self, imgs: List[ImageMat], meta: Dict) -> Tuple[List[ImageMat],Dict]:
        forwarded_imgs = self.forward_raw([img.data() for img in imgs])
        output_imgs = [self.out_mats[i].unsafe_update_mat(forwarded_imgs[i]) for i in range(len(forwarded_imgs))]        
        return output_imgs, meta

    def get_converted_imgs(self, meta: Dict = {}):
        """Retrieve forwarded images from metadata if stored."""
        return meta.get(self.uuid, None)


class ImageMatGenerator(BaseModel):
    sources: list[str] = str
    color_types: list[ColorType] = []
    _resources = []  # General-purpose resource registry
    _source_generators = []  # General-purpose resource registry

    def model_post_init(self, context):
        self._source_generators = [self.create_source_generator(src) for src in self.sources]
        return super().model_post_init(context)

    def register_resource(self, resource):
        self._resources.append(resource)

    @staticmethod
    def has_func(obj, name):
        return callable(getattr(obj, name, None))

    def release_resources(self):
        cleanup_methods = [
            "exit", "end", "teardown",
            "stop", "shutdown", "terminate",
            "join", "cleanup", "deactivate",
            "release", "close", "disconnect",
            "destroy",
        ]

        for res in self._resources:
            for method in cleanup_methods:
                if self.has_func(res, method):
                    try:
                        getattr(res, method)()
                    except Exception as e:
                        print(f"Error during {method} on {res}: {e}")

        self._resources.clear()


    def create_source_generator(self, source):
        raise NotImplementedError("Subclasses must implement `create_source_generator`")

    def iterate_raw_images(self):
        for generator in self._source_generators:
            yield from generator

    def __iter__(self):
        return self

    def __next__(self):
        for raw_image, color_type in zip(self.iterate_raw_images(), self.color_types):
            yield ImageMat(color_type=color_type).build(raw_image)

    def reset_generators(self):
        pass

    def release(self):
        self.release_resources()

    def __del__(self):
        self.release()

    def __len__(self):
        return None
