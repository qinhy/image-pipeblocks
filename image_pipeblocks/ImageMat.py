
# Standard Library Imports
import enum
import json
import math
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from .shmIO import NumpyUInt8SharedMemoryStreamIO

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
    UNKNOWN = 'unknown'

COLOR_TYPE_CHANNELS = {
    ColorType.BAYER: [1],
    ColorType.GRAYSCALE: [1],
    ColorType.RGB: [3],
    ColorType.BGR: [3],
}

# global setting
torch_img_dtype = torch.float16
numpy_img_dtype = np.uint8

class MatOps:
    int32=np.int32
    uint8=np.uint8
    float32=np.float32
    float16=np.float16
    def mat(self, pylist, dtype, device=None): raise NotImplementedError()
    def eye(self, size, dtype, device=None): raise NotImplementedError()
    def ones(self, shape, dtype, device=None): raise NotImplementedError()
    def zeros(self, shape, dtype, device=None): raise NotImplementedError() 
    def hstack(self, arrays): raise NotImplementedError()
    def norm(self, x): raise NotImplementedError()
    def dot(self, a, b): raise NotImplementedError()
    def cross(self, a, b): raise NotImplementedError()
    def matmul(self, a, b): raise NotImplementedError()
    def to_numpy(self, x)->np.ndarray: raise NotImplementedError()
    def mean(self, x, dim=0): raise NotImplementedError()
    def median(self, x, dim=0): raise NotImplementedError()
    def std(self, x, dim=0): raise NotImplementedError()
    def max(self, x, dim=0): raise NotImplementedError()
    def min(self, x, dim=0): raise NotImplementedError()
    def abs(self, x): raise NotImplementedError()
    def stack(self, xs, dim=0): raise NotImplementedError()
    def cat(self, xs, dim=0): raise NotImplementedError()
    def reshape(self, x, shape): raise NotImplementedError()
    def copy_mat(self, x): raise NotImplementedError()
    def logical_and(self, a, b): raise NotImplementedError()
    def logical_or(self, a, b): raise NotImplementedError()
    def clip(self, x, min_val, max_val): raise NotImplementedError()    
    def astype_int32(self, x): raise NotImplementedError()
    def astype_uint8(self, x): raise NotImplementedError()
    def astype_float32(self, x): raise NotImplementedError()    
    def astype_float16(self, x): raise NotImplementedError()
    def nonzero(self, x): raise NotImplementedError()

class NumpyMatOps(MatOps):
    int32=np.int32
    uint8=np.uint8
    float32=np.float32
    float16=np.float16
    def mat(self, pylist, dtype, device=None): return np.array(pylist, dtype=dtype)
    def eye(self, size, dtype, device=None): return np.eye(size, dtype=dtype)
    def ones(self, shape, dtype, device=None): return np.ones(shape, dtype=dtype)
    def zeros(self, shape, dtype, device=None): return np.zeros(shape, dtype=dtype)
    def hstack(self, arrays): return np.hstack(arrays)
    def norm(self, x): return np.linalg.norm(x)
    def dot(self, a, b): return np.dot(a, b)
    def cross(self, a, b): return np.cross(a, b)
    def matmul(self, a, b): return a @ b
    def to_numpy(self, x)->np.ndarray: return x
    def mean(self, x, dim=0): return np.mean(x, axis=dim)
    def median(self, x, dim=0): return np.median(x, axis=dim)
    def std(self, x, dim=0): return np.std(x, axis=dim)
    def max(self, x, dim=0): return np.max(x, axis=dim)
    def min(self, x, dim=0): return np.min(x, axis=dim)
    def abs(self, x): return np.abs(x)
    def stack(self, xs, dim=0): return np.stack(xs, axis=dim)
    def cat(self, xs, dim=0): return np.concatenate(xs, axis=dim)
    def reshape(self, x, shape): return np.reshape(x, shape)
    def copy_mat(self, x): return x.copy()
    def logical_and(self, a, b): return np.logical_and(a, b)
    def logical_or(self, a, b): return np.logical_or(a, b)
    def clip(self, x, min_val, max_val): return np.clip(x, min_val, max_val)  
    def astype_int32(self, x): return x.astype(np.int32)
    def astype_uint8(self, x): return x.astype(np.uint8)
    def astype_float32(self, x): return x.astype(np.float32)        
    def astype_float16(self, x): return x.astype(np.float16)
    def nonzero(self, x) : return np.nonzero(x)


class TorchMatOps(MatOps):
    int32=torch.int32
    uint8=torch.uint8
    float32=torch.float32
    float16=torch.float16
    def mat(self, pylist, dtype, device=None): return torch.tensor(pylist, dtype=dtype, device=device)
    def eye(self, size, dtype, device=None): return torch.eye(size, dtype=dtype, device=device)
    def ones(self, shape, dtype, device=None): return torch.ones(shape, dtype=dtype, device=device)
    def zeros(self, shape, dtype, device=None): return torch.zeros(shape, dtype=dtype, device=device)
    def hstack(self, arrays): return torch.cat(arrays, dim=1)
    def norm(self, x): return torch.norm(x)
    def dot(self, a, b): return torch.dot(a, b)
    def cross(self, a, b): return torch.cross(a, b)
    def matmul(self, a, b): return torch.matmul(a, b)
    def to_numpy(self, x)->np.ndarray: return x.detach().permute(0, 2, 3, 1).cpu().numpy()
    def mean(self, x, dim=0): return torch.mean(x, dim=dim)
    def median(self, x, dim=0): return torch.median(x, dim=dim).values
    def std(self, x, dim=0): return torch.std(x, dim=dim, unbiased=False)
    def max(self, x, dim=0): return torch.max(x, dim=dim).values
    def min(self, x, dim=0): return torch.min(x, dim=dim).values
    def abs(self, x): return torch.abs(x)
    def stack(self, xs, dim=0): return torch.stack(xs, dim=dim)
    def cat(self, xs, dim=0): return torch.cat(xs, dim=dim)
    def reshape(self, x, shape): return x.reshape(shape)
    def copy_mat(self, x): return x.clone()
    def logical_and(self, a, b): return torch.logical_and(a, b)
    def logical_or(self, a, b): return torch.logical_or(a, b)
    def clip(self, x, min_val, max_val): return torch.clamp(x, min=min_val, max=max_val)
    def astype_int32(self, x): return x.type(dtype=torch.int32)
    def astype_uint8(self, x): return x.type(dtype=torch.uint8)
    def astype_float32(self, x): return x.type(dtype=torch.float32)
    def astype_float16(self, x): return x.type(dtype=torch.float16)
    def nonzero(self, x) : return torch.nonzero(x)


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
    raw_shape: List[int] = []
    max_value: Optional[Union[int, float]] = None
    B: Optional[int] = 1
    C: Optional[int] = None
    H: int = 0
    W: int = 0
    color_type: Optional[ColorType] = None
    latlon: Tuple[float,float] = (math.nan,math.nan) #gps
    class_name:str = ''
    path:str = ''
    uuid: str = ''

    @staticmethod
    def torch_img_dtype():return torch_img_dtype
    @staticmethod
    def numpy_img_dtype():return numpy_img_dtype

    def model_post_init(self, context):        
        self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'  
        return super().model_post_init(context)

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
            elif color_type != ColorType.JPEG:
                raise ValueError(f"NumPy array must be 2D (HW) or 3D (HWC). Got {img_data.shape}")

        elif isinstance(img_data, torch.Tensor):
            self._dtype = img_data.dtype
            self.device = str(img_data.device)
            self.max_value = 1.0

            if img_data.ndim == 4:
                self.shape_type = ShapeType.BCHW
                self.B, self.C, self.H, self.W = img_data.shape
            else:
                raise ValueError(f"Torch tensor must be 4D (BCHW). Got {img_data.shape}")
        else:
            raise TypeError(f"img_data must be np.ndarray or torch.Tensor, got {type(img_data)}")
        
        if img_data.max() > self.max_value:
            raise ValueError(f"max value must be {self.max_value}. Got {img_data.max()}")

        # Channel count validation
        if color_type in COLOR_TYPE_CHANNELS:
            expected_channels = COLOR_TYPE_CHANNELS[color_type]
            if self.C not in expected_channels:
                raise ValueError(
                    f"Invalid color type '{color_type.value}' for image with {self.C} channels. "
                    f"Expected {expected_channels} channels. Data shape: {img_data.shape}"
                )
        self.color_type = color_type
        self.raw_shape = [*img_data.shape]
        return self

class ImageMat(BaseModel):

    class TorchDtype(BaseModel):
        is_floating_point: bool
        is_complex: bool
        is_signed: bool
        itemsize: int

    info: ImageMatInfo = ImageMatInfo()
    color_type: Union[str, ColorType]
    timestamp:float = 0
    _img_data: np.ndarray|torch.Tensor = None

    shmIO_mode: Literal[False,'writer','reader'] = False
    shmIO_writer:Optional[NumpyUInt8SharedMemoryStreamIO.Writer] = None
    shmIO_reader:Optional[NumpyUInt8SharedMemoryStreamIO.Reader] = None

    @staticmethod
    def random(color_type,shape,lib='np',device='cpu'):
        if   lib=='np':
            data = np.random.randint(0, 255, size=shape, dtype=ImageMatInfo.numpy_img_dtype())
        elif lib=='torch':
            data = torch.rand(shape, dtype=ImageMatInfo.torch_img_dtype(), device=device)
        else:
            raise TypeError(f"Unsupported image lib type: {lib}")
        return ImageMat(color_type).build(data)
    
    def clone(self):
        info = self.info.model_copy()
        data = self._img_data.copy()
        return ImageMat(self.color_type).build(data,info)
    
    def zero_clone(self):
        info = self.info.model_copy()
        data = self._img_data.copy() * 0
        return ImageMat(self.color_type).build(data,info)

    def random_clone(self):
        info = self.info.model_copy()
        if isinstance(self._img_data, np.ndarray):
            data = np.random.randint(0, 255, size=self._img_data.shape, dtype=self.info._dtype)
        elif isinstance(self._img_data, torch.Tensor):
            data = torch.rand(self._img_data.shape, dtype=self.info._dtype, device=self.info.device)
        else:
            raise TypeError(f"Unsupported image data type: {type(self._img_data)}")
        return ImageMat(self.color_type).build(data,info)

    def model_post_init(self, context):
        if self.shmIO_writer and self.shmIO_reader:
            raise ValueError('One ImageMat shmIO cannot have both writer and reader.')
        if self.shmIO_writer:
            self.shmIO_writer.build_buffer()
        if self.shmIO_reader:
            self.shmIO_reader.build_buffer()
        return super().model_post_init(context)

    def build(self, img_data: Union[np.ndarray, torch.Tensor, str], info: Optional[ImageMatInfo] = None):
        self.info = info or ImageMatInfo().build(img_data, color_type=self.color_type)
        self.unsafe_update_mat(img_data)
        return self
    
    def build_shmIO(self, shmIO_mode:str=False,target_mat_info:'ImageMat'=None):
        self.shmIO_mode = shmIO_mode
        if self.shmIO_mode == 'writer' and self.info.device=='cpu':
            self.build_shmIO_writer()
        if self.shmIO_mode == 'reader' and target_mat_info and self.info.device=='cpu':
            self.build_shmIO_reader(target_mat_info)
        return self
    
    def build_shmIO_writer(self):
        self.shmIO_mode='writer'
        if self.info is None: raise ValueError("self.info cannot be None")
        self.shmIO_writer = NumpyUInt8SharedMemoryStreamIO.writer(self.info.uuid,self.info.raw_shape)
        return self

    def build_shmIO_reader(self,target_mat_info:'ImageMat'):
        self.shmIO_mode='reader'
        if self.info is None: raise ValueError("self.info cannot be None")
        self.shmIO_reader = NumpyUInt8SharedMemoryStreamIO.reader(target_mat_info.uuid,target_mat_info.raw_shape)
        return self

    def to_shmIO_writer(self):
        self.shmIO_mode='writer'
        self.shmIO_writer = NumpyUInt8SharedMemoryStreamIO.writer(self.shmIO_reader.shm_name,self.shmIO_reader.array_shape)
        self.shmIO_reader = None

    def to_shmIO_reader(self):
        self.shmIO_mode='reader'
        self.shmIO_reader = NumpyUInt8SharedMemoryStreamIO.reader(self.shmIO_writer.shm_name,self.shmIO_writer.array_shape)
        self.shmIO_writer = None

    def release(self):
        if self.shmIO_reader:
            self.shmIO_reader.close()
        if self.shmIO_writer:
            self.shmIO_writer.close()

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
        self.info = ImageMatInfo(img_data, color_type=self.info.color_type)
        self.unsafe_update_mat(img_data)
        return self

    def unsafe_update_mat(self, img_data: Union[np.ndarray, torch.Tensor]) -> 'ImageMat':
        """Update the image data without updating metadata (use with caution)."""
        if self.shmIO_mode:
            self.shmIO_writer.write(img_data)
        self._img_data = img_data
        self.timestamp = time.time()
        return self

    def data(self) -> Union[np.ndarray, torch.Tensor]:
        """Return the image data."""
        if self.shmIO_mode:
            return self.shmIO_reader.read()
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

    def require_BCHW(self, B=None, C=None):
        self.require_shape_type(ShapeType.BCHW)
        if B is not None and self.info.B != B:
            raise TypeError(f"Expected {B} batch dimension, got {self.info.B}")
        if C is not None and self.info.C != C:
            raise TypeError(f"Expected {C} channel dimension, got {self.info.C}")

    def require_square_size(self):
        if self.info.W != self.info.H:
            raise TypeError(f"Expected square size (W==H). Got {self.info.W} != {self.info.H}")

class ImageMatProcessor(BaseModel):
    class MetaData(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
    title:str
    uuid:str = ''

    bounding_box_owner_uuid: Optional[str] = None
    bounding_box_xyxy: List[ np.ndarray ] = Field([],exclude=True) # [img1_xyxy ... ] [ [[x,y,x,y]...] ... ]

    pixel_idx_forward_T : List[ List[List[float]] ] = Field([],exclude=True) #[eye(3,3)...] # [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pixel_idx_backward_T: List[ List[List[float]] ] = Field([],exclude=True) #[eye(3,3)...] # [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    
    _numpy_pixel_idx_forward_T : List[ np.ndarray ] = []
    _numpy_pixel_idx_backward_T: List[ np.ndarray ] = []

    save_results_to_meta:bool = False
    input_mats: List[ImageMat] = []
    out_mats: List[ImageMat] = []
    meta:dict = {}
    
    num_devices:List[str] = ['cpu']
    num_gpus:int = 0
    _enable:bool = True
    _eye:List = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    _mat_funcs:List[MatOps] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def init_common_utility_methods(self,idx,is_ndarray=True):
        if idx<len(self._mat_funcs):
            self._mat_funcs[idx] = NumpyMatOps() if is_ndarray else TorchMatOps()
        else:            
            self._mat_funcs.append( NumpyMatOps() if is_ndarray else TorchMatOps() )

    def print(self,*args):
        print(f'##############[{self.uuid}]#################')
        print(f'[{self.uuid}]',*args)
        print(f'############################################')

    def model_post_init(self, context: Any, /) -> None:
        if len(self.title)==0:
            self.title = self.__class__.__name__
        if len(self.uuid)==0:
            self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'
        return super().model_post_init(context)
    
    def is_enable(self):
        return self._enable

    def on(self):
        self._enable = True
        return self

    def off(self):
        self._enable = False
        return self
        
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
    
    def validate_img(self, img_idx:int, img:ImageMat):
        pass

    def validate(self, imgs: List[ImageMat], meta: Dict = {}, run=True):
        input_mats = [None]*len(imgs)
        for i,img in enumerate(imgs):
            self.validate_img(i,img)
            if img.shmIO_mode==False:
                input_mats[i]=img
            elif img.shmIO_mode=='writer':
                img = img.model_copy()
                img.to_shmIO_reader()
            else:
                raise ValueError(f"input_mats shmIO_mode must be None or writer. Got {img.shmIO_mode}")
            input_mats[i]=img

        self.input_mats = input_mats
        if run: return self(self.input_mats, meta)
        return self.input_mats,meta
    
    def build_out_mats(self,validated_imgs: List[ImageMat],converted_raw_imgs,color_type=None):
        if color_type is None:
            # 1:1 mapping
            self.out_mats = [ImageMat(color_type=old.info.color_type if color_type is None else color_type
                ).build(img)
                for old,img in zip(validated_imgs,converted_raw_imgs)]
        else:
            self.out_mats = [ImageMat(color_type=color_type).build(img) for img in converted_raw_imgs]
        return self.out_mats
    
    def build_pixel_transform_matrix(self, imgs_info: List[ImageMatInfo]=[]):
        self.pixel_idx_forward_T = [self._eye for _ in range(len(imgs_info))]
        self.pixel_idx_backward_T = [self._eye for _ in range(len(imgs_info))]

    def forward_transform_matrix(self, proc: 'ImageMatProcessor'):
        original_boxes = proc.bounding_box_xyxy        
        transformed_boxes = []
        if not self._numpy_pixel_idx_forward_T:
            self._numpy_pixel_idx_forward_T  = [np.asarray(i) for i in self.pixel_idx_forward_T ]

        if len(original_boxes)!=len(self._numpy_pixel_idx_forward_T):pass
            # raise ValueError(f"original_boxes and numpy_pixel_idx_forward_T are not same size. Got {len(original_boxes)} and {len(self._numpy_pixel_idx_forward_T)}")
        
        for i,(boxes, T) in enumerate(zip(original_boxes, self._numpy_pixel_idx_forward_T)):
            if boxes.size == 0:
                transformed_boxes.append(boxes)
                continue
            # Prepare homogeneous coordinates for top-left and bottom-right corners
            ones = np.ones((boxes.shape[0], 1), dtype=boxes.dtype)
            top_left = np.hstack((boxes[:, :2], ones))      # [x1, y1, 1]
            bottom_right = np.hstack((boxes[:, 2:4], ones)) # [x2, y2, 1]

            # Apply transformation
            tl_trans = top_left @ T.T  # Shape: (N, 3)
            br_trans = bottom_right @ T.T

            # Normalize if using homography (perspective)
            if tl_trans.shape[1] == 3 and not np.allclose(tl_trans[:, 2], 1):
                tl_trans /= tl_trans[:, 2:3]
                br_trans /= br_trans[:, 2:3]

            # Concatenate transformed coordinates and any remaining box fields
            transformed = np.hstack((
                tl_trans[:, :2],
                br_trans[:, :2],
                boxes[:, 4:]  # Preserve any extra box data
            ))
            ## !!!!!!!! overwrite original !!!!!!!
            original_boxes[i][:] = transformed
            transformed_boxes.append(original_boxes)

        return transformed_boxes


    def forward_raw(self, imgs: List[Any], imgs_info: List[ImageMatInfo]=[], meta={}) -> List[Any]:
        raise NotImplementedError()

    def forward(self, imgs: List[ImageMat], meta: Dict) -> Tuple[List[ImageMat],Dict]:
        if imgs and imgs[0].shmIO_mode==False:
            imgs = imgs
        elif self.input_mats:
            imgs = self.input_mats

        input_infos = [img.info for img in imgs]
        forwarded_imgs = [img.data() for img in imgs]

        if self._enable:
            forwarded_imgs = self.forward_raw(forwarded_imgs,input_infos,meta)
            if self.bounding_box_owner_uuid:
                self.forward_transform_matrix(meta[self.bounding_box_owner_uuid])       
        
        if len(self.out_mats)==len(forwarded_imgs):
            output_imgs = [self.out_mats[i].unsafe_update_mat(forwarded_imgs[i]) for i in range(len(forwarded_imgs))]
        else:
            # Create new ImageMat instances for output
            output_imgs = self.build_out_mats(self.input_mats,forwarded_imgs)
            shmIO_mode = any([i.shmIO_mode for i in imgs])          
            if shmIO_mode:  
                for o in output_imgs:
                    o.shmIO_mode='writer'
                    o.build_shmIO_writer()

            if len(self.pixel_idx_forward_T)==0:
                self.build_pixel_transform_matrix(input_infos)

        self.out_mats = output_imgs
            
        if self.save_results_to_meta:
            meta[self.uuid] = self

        return output_imgs, meta
    
    def __call__(self, imgs: List[ImageMat], meta: dict = {}):
        return self.forward(imgs, meta)

    def release(self):
        if hasattr(self,'input_mats'):
            for i in self.input_mats:i.release()
        if hasattr(self,'out_mats'):
            for i in self.out_mats:i.release()
    
    def __del__(self):
        self.release()

