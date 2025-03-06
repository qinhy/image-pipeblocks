
from concurrent.futures import ThreadPoolExecutor
import enum
import random
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import ultralytics
from ultralytics.utils import ops
import time
import numpy as np
from typing import List, Any, Dict, Tuple, Union
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, List, Optional, Any

class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)

class Debayer5x5(torch.nn.Module):
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

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer5x5, self).__init__()
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
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
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

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
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
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)

class DummyDebayer5x5(torch.nn.Module):
    def __init__(self):
        super(DummyDebayer5x5, self).__init__()

    def forward(self, x):
        # Repeat the single channel 3 times to create an RGB image
        return x.repeat(1, 3, 1, 1)  # Shape: (N, 3, H, W)

class ImageVerifier:
    """A class for verifying NumPy and Torch image types, shapes, and value ranges."""

    def _raise_error(self, message: str, strict: bool) -> bool:
        """Handles error raising or returns False based on strict mode."""
        if strict:
            raise ValueError(message)
        return False

    # ------------------------- TYPE CHECKS -------------------------
    def verify_type_numpy(self, img: np.ndarray, idx: int, strict: bool = True) -> bool:
        """Verify that a NumPy image is of type uint8."""
        if not isinstance(img, np.ndarray):
            return self._raise_error(f"Expected a NumPy array at index {idx}, but got {type(img)}.", strict)
        if img.dtype != np.uint8:
            return self._raise_error(f"NumPy image at index {idx} has invalid dtype {img.dtype}. Expected uint8.", strict)

    def verify_type_torch(self, img: torch.Tensor, idx: int, strict: bool = True) -> bool:
        """Verify that a Torch image is of type float32 or float64."""
        if not isinstance(img, torch.Tensor):
            return self._raise_error(f"Expected a Torch tensor at index {idx}, but got {type(img)}.", strict)
        if img.dtype not in [torch.float32, torch.float64]:
            return self._raise_error(f"Torch image at index {idx} has invalid dtype {img.dtype}. Expected float32 or float64.", strict)

    # ------------------------- SHAPE CHECKS -------------------------
    def _validate_shape(self, img, expected_shapes, img_type: str, strict: bool) -> bool:
        """Generic shape validation for images."""
        if img.shape not in expected_shapes:
            shape_str = " or ".join(map(str, expected_shapes))
            return self._raise_error(f"Invalid shape: {img.shape}. Expected {shape_str} for {img_type}.", strict)

    # ---- NumPy Shape Validation ----
    def verify_shape_bayer_numpy(self, img: np.ndarray, strict: bool = True) -> bool:
        """Check if NumPy image has Bayer shape (H, W)."""
        return self._validate_shape(img, expected_shapes=[(img.shape[0], img.shape[1])], img_type="Bayer NumPy image", strict=strict)

    def verify_shape_rgb_numpy(self, img: np.ndarray, strict: bool = True) -> bool:
        """Check if NumPy image has RGB shape (H, W, C) with 3 or 4 channels."""
        return self._validate_shape(img, expected_shapes=[(img.shape[0], img.shape[1], 3)], img_type="RGB NumPy image", strict=strict)

    # ---- Torch Shape Validation ----
    def verify_shape_rgb_torch(self, img: torch.Tensor, strict: bool = True) -> bool:
        """Check if Torch image has RGB shape (B, C, H, W) with 3 or 4 channels."""
        return self._validate_shape(img, expected_shapes=[(1, 3, img.shape[-2], img.shape[-1])], img_type="RGB Torch image", strict=strict)

    def verify_shape_batch_rgb_torch(self, img: torch.Tensor, strict: bool = True) -> bool:
        """Check if Torch batch image has RGB shape (B, C, H, W) with 3 or 4 channels."""
        return self._validate_shape(img, expected_shapes=[(img.shape[0], 3, img.shape[2], img.shape[3])], img_type="Batch RGB Torch image", strict=strict)

    def verify_shape_bayer_torch(self, img: torch.Tensor, strict: bool = True) -> bool:
        """Check if Torch batch image has Bayer shape (B, H, W)."""
        return self._validate_shape(img, expected_shapes=[(img.shape[0], img.shape[1], img.shape[2])], img_type="Bayer Torch image", strict=strict)

    # ------------------------- VALUE RANGE CHECKS -------------------------
    def verify_range_numpy(self, img: np.ndarray, idx: int, strict: bool = True) -> bool:
        """Verify that a NumPy image has values within [0, 255]."""
        if img.min() < 0 or img.max() > 255:
            return self._raise_error(f"NumPy image at index {idx} has values outside the range [0, 255].", strict)

    def verify_range_torch(self, img: torch.Tensor, idx: int, strict: bool = True) -> bool:
        """Verify that a Torch image has values within [0, 1]."""
        if img.min() < 0 or img.max() > 1:
            return self._raise_error(f"Torch image at index {idx} has values outside the range [0, 1].", strict)

class PipeBlock(ImageVerifier):
    def __init__(self, title: str = "null"):
        self.title = title

    def __call__(self, imgs: List[Any], meta: Dict = {}):
        # start = time.time()
        imgs, meta = self.forward(imgs, meta)
        # elapsed_time = time.time() - start
        # fps = 1 / (elapsed_time + 1e-5)
        # print(f"#### profile #### {fps:.2f} FPS - {self.title}")
        return imgs, meta    
    
    def gpu_info(self):
        # Detect available GPUs
        self.num_gpus = torch.cuda.device_count()
        print(f'[{self.__class__.__name__}] Number of GPUs available: {self.num_gpus}')
        # Ensure at least CPU is used
        self.num_gpus = max(self.num_gpus, 1)
        return self.num_gpus

    def _basic_test(self, imgs: List[Any], meta: Dict = {}):
        if not isinstance(imgs, list):
            raise TypeError(f"Expected 'imgs' to be a list, but got {type(imgs).__name__}.")
        if not all([hasattr(i,'shape') for i in imgs]):
            raise TypeError(f"Expected 'imgs' all has .shape, but {[type(i) for i in imgs]}.")
        if len(set([i.shape for i in imgs]))!=1:
            raise TypeError(f"Expected 'imgs' all has same shape, but {[i.shape for i in imgs]}.")
        if not isinstance(meta, dict):
            raise TypeError(f"Expected 'meta' to be a dict, but got {type(meta).__name__}.")
    
    def test_forward_numpy_rgb(self, imgs: List[Any], meta: Dict = {}):
        self._basic_test(imgs, meta)
        for idx, img in enumerate(imgs):
            self.verify_type_numpy(img, idx)
            self.verify_shape_rgb_numpy(img)
            self.verify_range_numpy(img,idx)
        return self.forward(imgs, meta)    
    
    def test_forward_torch_rgb(self, imgs: List[Any], meta: Dict = {}):
        self._basic_test(imgs, meta)
        for idx, img in enumerate(imgs):
            self.verify_type_torch(img, idx)
            self.verify_shape_rgb_torch(img)
            self.verify_range_torch(img,idx)
        return self.forward(imgs, meta)
    
    def test_forward_numpy_bayer(self, imgs: List[Any], meta: Dict = {}):
        self._basic_test(imgs, meta)
        for idx, img in enumerate(imgs):
            self.verify_type_numpy(img, idx)
            self.verify_shape_bayer_numpy(img)
        return self.forward(imgs, meta) 
    
    def test_forward_torch_bayer(self, imgs: List[Any], meta: Dict = {}):
        self._basic_test(imgs, meta)
        for idx, img in enumerate(imgs):
            self.verify_type_torch(img, idx)
            self.verify_shape_bayer_torch(img)
        return self.forward(imgs, meta) 
    
    def forward(self, imgs: List[Any], meta: Dict = {}):
        """ This function should be overridden by subclasses """
        return imgs, meta

class StaticWords(enum.Enum):
    yolo_results = 'yolo_results'
    yolo_input_imgs ='yolo_input_imgs'
    yolo_input_img_w_h ='yolo_input_img_w_h'
    sliding_window_input_imgs = 'input_imgs'
    sliding_window_output_offsets = 'sliding_window_output_offsets'

class LambdaBlock(PipeBlock):
    def __init__(self,forward_func=lambda imgs,meta:(imgs,meta),title='lambda'):
        super().__init__()
        self.title = title
        self.forward_func = forward_func

    def forward(self, imgs, meta={}):
        return self.forward_func(imgs,meta)

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.forward(imgs,meta)
    
class CvDebayerBlock(PipeBlock):
    def __init__(self,formart=cv2.COLOR_BAYER_BG2RGB):
        super().__init__()
        self.title = 'cv_debayer'
        self.formart = formart

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.test_forward_numpy_bayer(imgs,meta)
    
    def forward(self, imgs, meta={}):
        debayered_imgs = [cv2.cvtColor(img,self.formart) for img in imgs]
        return debayered_imgs, meta

class TorchDebayerBlock(PipeBlock):
    def __init__(self, dtype=torch.float32):
        """
        Converts a batched Bayer image (B, 1, H, W) into an RGB image (B, 3, H, W) using Torch's Debayer5x5.

        :param batched_img: PyTorch tensor of shape (B, 1, H, W) with dtype uint8.
        """
        super().__init__()
        self.title = 'torch_debayer'
        self.num_gpus = self.gpu_info()
        self.dtype = dtype
        
        self.debayer_models = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
            """
            Convert a Bayer raw image (B, 1, H, W) tensor to an RGB image (B, 3, H, W) using PyTorch's Debayer5x5.
            """
            model = Debayer5x5().to(device)
            self.debayer_models.append((model, device))

    def test_forward(self, imgs, meta={}):
        self.test_forward_torch_bayer(imgs, meta)
        return self.forward(imgs,meta)

    def forward(self, imgs: List[np.ndarray], meta={}):
        torch_imgs = []
        for i,img in enumerate(imgs):
            model,device = self.debayer_models[i%self.num_gpus]
            torch_imgs.append(model(img.unsqueeze(0)))
        return torch_imgs, meta
    

class TorchRGBToNumpyBGRBlock(PipeBlock):
    def __init__(self):
        super().__init__()
        self.title = 'torch_to_numpy'

    def forward(self, imgs: List[np.ndarray], meta={}):
        # [ [1,3,h,w], ...]
        imgs = [
            img.mul(255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            for i,img in enumerate(imgs)
        ]
        
        imgs = [
            img.cpu().numpy()[0][:,:,::-1]
            for i,img in enumerate(imgs)
        ]
        return imgs, meta

    def test_forward(self, imgs: List[np.ndarray], meta: Dict = {}):
        self.test_forward_torch_rgb(imgs, meta)
        return self.forward(imgs, meta)
    
class NumpyRGBToTorchBlock(PipeBlock):
    def __init__(self,dtype=torch.float32):
        super().__init__()
        self.title = "numpy_rgb_to_torch"
        self.num_gpus = self.gpu_info()
        self.dtype = dtype

        def get_model(device,dtype=dtype):
            return lambda img:torch.tensor(img, dtype=dtype, device=device).div(255.0).permute(2, 0, 1)
        
        self.tensor_models = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
            model = get_model(f"cuda:{i}")
            self.tensor_models.append((model, device))


    def forward(self, imgs: List[np.ndarray], meta={}):
        torch_imgs = [
            self.tensor_models[i%self.num_gpus][0](img)
            for i,img in enumerate(imgs)
        ]
        return torch_imgs, meta

    def test_forward(self, imgs: List[np.ndarray], meta: Dict = {}):
        for idx, img in enumerate(imgs):
            self.verify_type_numpy(img, idx)  # Ensure NumPy array
            self.verify_shape_rgb_numpy(img)  # Ensure (H, W, 3) for RGB
        return self.forward(imgs, meta)

class NumpyBayerToTorchBlock(PipeBlock):
    def __init__(self,dtype=torch.float32):
        super().__init__()
        self.title = "numpy_bayer_to_torch"
        self.num_gpus = self.gpu_info()
        self.dtype = dtype

        def get_model(device,dtype=dtype):
            return lambda img:torch.tensor(img, dtype=dtype, device=device).div(255.0).unsqueeze(0)
        
        self.tensor_models = []
        for i in range(self.num_gpus):
            device = f"cuda:{i}"
            model = get_model(f"cuda:{i}")
            self.tensor_models.append((model, device))

    def forward(self, imgs: List[np.ndarray], meta={}):
        torch_imgs = [
            self.tensor_models[i%self.num_gpus][0](img)
            # torch.tensor(img, dtype=torch.float32, device=self.device).div(255.0).unsqueeze(0)  # Normalize & add channel dim
            for i,img in enumerate(imgs)
        ]
        return torch_imgs, meta

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.test_forward_numpy_bayer(imgs,meta)

class TorchResizeBlock(PipeBlock):
    def __init__(self, output_size: tuple=(1280,1280), mode: str = 'bilinear', align_corners: bool = False):
        super().__init__()
        self.title = 'torch_resize'
        self.output_size = output_size  # (height, width)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, imgs: List[torch.Tensor], meta={}):
        resized_imgs: List[torch.Tensor] = []
        b,c,height,width = imgs[0].shape
        meta[StaticWords.yolo_input_img_w_h] = width,height
        for img_tensor in imgs:
            # img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            resized_tensor = F.interpolate(img_tensor,
                    size=self.output_size, mode=self.mode, align_corners=self.align_corners)
            resized_imgs.append(resized_tensor)
        return resized_imgs, meta    

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.test_forward_torch_rgb(imgs,meta) # B,C,H,W

class CVResizeBlock(PipeBlock):
    def __init__(self, output_size: tuple=(1280,1280), interpolation=cv2.INTER_LINEAR):
        super().__init__()
        self.title = 'cv_resize'
        self.output_size = output_size  # (width, height)
        self.interpolation = interpolation

    def forward(self, imgs, meta={}):
        resized_imgs = []
        # Store the original size
        width,height = imgs[0].shape[:2]        
        meta[StaticWords.yolo_input_img_w_h] = width,height

        for img in imgs:
            # Resize the image
            resized_img = cv2.resize(img, self.output_size, interpolation=self.interpolation)
            resized_imgs.append(resized_img)
        return resized_imgs, meta

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.test_forward_numpy_rgb(imgs,meta)

class TileImagesBlock(PipeBlock):
    def __init__(self, tile_width=2):
        super().__init__()
        self.title = 'tile_images'
        self.tile_width = tile_width
    
    def forward(self, imgs, meta={}):
        if not imgs:
            raise ValueError("The images list cannot be empty.")

        # Resize images if a target size is specified
        # if self.dsize:
        #     imgs = [cv2.resize(img, self.dsize) for img in imgs]

        # Get image dimensions
        h, w = imgs[0].shape[:2]
        num_images = len(imgs)
        tile_height = (num_images + self.tile_width - 1) // self.tile_width  # Ensures enough rows

        # Create a blank canvas
        tile = np.zeros((tile_height * h, self.tile_width * w, 3), dtype=imgs[0].dtype)

        # Place images in the tile
        for i, img in enumerate(imgs):
            r, c = divmod(i, self.tile_width)
            tile[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

        return [tile], meta
    

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.test_forward_numpy_rgb(imgs,meta)
  
class EncodeJpegBlock(PipeBlock):
    def __init__(self, quality=None):
        super().__init__()
        self.title = 'encode_jpeg'
        self.quality = quality  # JPEG encoding quality (0-100)

    def forward(self, imgs, meta={}):
        encoded_images = []
        for img in imgs:
            if self.quality is not None:
                success, encoded = cv2.imencode('.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
            else:
                success, encoded = cv2.imencode('.jpeg', img)

            if not success:
                raise ValueError("JPEG encoding failed.")

            encoded_images.append(encoded)

        return encoded_images, meta

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        return self.test_forward_numpy_rgb(imgs,meta)
 
class MergeYoloResultsBlock(PipeBlock):
    def __init__(self):
        super().__init__()
        self.title = 'merge_yolo_results'

    def forward(self, imgs, meta={}):
        results = meta.get(StaticWords.yolo_results)
        if not results:
            raise ValueError("yolo results list cannot get.")

        if len(results) == 1:
            return results[0], meta
        
        # Merge YOLO detection results        
        if hasattr(results[0],'boxes'):
            boxes = torch.cat([res.boxes.data.cpu() for res in results])
            # Create a new result object and update it with merged boxes
            result = results[0].new()
            result.update(boxes=boxes)
        elif isinstance(results[0],np.ndarray):
            result = np.vstack(results)
        meta[StaticWords.yolo_results] = result
        return imgs, meta
    
    def test_forward(self, imgs, meta = {}):
        return self.forward(imgs, meta)
    
class YOLOPredictor(PipeBlock):
    def __init__(
        self, model_path,  # Path to YOLO model (e.g., 'yolov8n.pt')
        imgsz=640,
        conf=0.25,
        class_names=None,
        non_notify_classes=None,
        conf_thres_per_class=None,
        conf_thres_per_class_rlfb=None
    ):
        super().__init__()
        self.title = 'yolo_predictor'
        # Default values
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names or []
        self.non_notify_classes = non_notify_classes or []
        self.conf_thres_per_class = conf_thres_per_class or {}
        self.conf_thres_per_class_rlfb = conf_thres_per_class_rlfb or [[], [], [], []]
        
        self.conf_thres_per_class = {k: v for k, v in conf_thres_per_class}
        self.conf_thres_per_class_rlfb = [
            {k: v for k, v in conf_thres} for conf_thres in conf_thres_per_class_rlfb
        ]

        self.gpu_info()
        # Load models onto separate GPUs
        self.yolo_models = []
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                device = f"cuda:{i}"
                model = YOLO(model_path).to(device)
                self.yolo_models.append((model, device))

            print("[YOLOPredictor] Models loaded on available GPUs:", [device for _, device in self.yolo_models])
        else: 
            self.yolo_models = [YOLO(model_path).to('cpu')]
    
    def forward(self, imgs, meta={}):
        meta[StaticWords.yolo_input_imgs] = imgs
        res = self.predict(imgs)
        meta[self.title] = self.postprocess(res)
        meta[StaticWords.yolo_results] = meta[self.title]
        return imgs, meta
    
    def test_forward(self, imgs, meta = {}):
        return self.forward(imgs,meta)
    
    def predict(self, imgs):
        raise NotImplemented('this is a interface class!')

    def postprocess(self, results):
        # results = [OriginalDetectionResults(res) for res in results]
        # Apply confidence threshold filtering
        results = self.remove_classes(results)
        results = [self.filter_thres(res, self.conf_thres_per_class) for res in results]
        # for res in results:
        #     for i, label_name in enumerate(self.class_names):
        #         res.names[i] = label_name
        return results

    def remove_classes(self,results):
        if len(self.non_notify_classes)==0:  return results
        results_filtered = []
        for res in results:
            keep = [cls not in self.non_notify_classes for cls in res.boxes.cls]
            # res_new = ultralytics.engine.results.Results(res)
            res.update(boxes=res.boxes.data[keep])
            results_filtered.append(res)
        return results_filtered
    
    def filter_thres(self, result, conf_thres_per_class:dict):
        """Filters results based on class-specific confidence thresholds."""
        if not conf_thres_per_class: return result
        if hasattr(result,'boxes'):
            boxes = result.boxes.data
            confs = boxes[:, 4].tolist()
            labels = boxes[:, 5].int().tolist()
            keep = [(conf_thres_per_class.get(l, 0) < c) for c, l in zip(confs, labels)]
            result.update(boxes=boxes[keep])
            return result
        elif isinstance(result,np.ndarray):
            confs = result[:,4]
            labels = result[:,5]
            keep = [(conf_thres_per_class.get(int(l), 0) < c) for c, l in zip(confs, labels)]
            return result[keep]

class YOLOPredictorGPU(YOLOPredictor):

    def test_forward(self, imgs, meta = {}):
        for i in range(self.num_gpus):
            batch = [(i * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
                      for i in imgs[i::self.num_gpus]]
            (yolo_model,device) = self.yolo_models[i]
            # print([i.shape for i in batch])
            yolo_model(batch, imgsz=self.imgsz, conf=self.conf)
        return self.forward(imgs,meta)
    
    def predict(self, imgs):
        """
        Run YOLO inference in parallel across GPUs.
        """       
        predictions = [None] * len(imgs)  # Placeholder for ordered predictions

        for i in range(self.num_gpus):
            batch_indices = list(range(i, len(imgs), self.num_gpus))  # Track original indices
            batch = [imgs[idx] for idx in batch_indices]  # Assign correct images to batch

            (yolo_model, device) = self.yolo_models[i]
            
            with torch.no_grad():
                tmp = torch.vstack([img.to(device) for img in batch])  # Move batch to correct GPU
                b, c, h, w = tmp.shape

                preds, feature_maps = yolo_model.model(tmp)

                preds = ops.non_max_suppression(
                    preds,
                    yolo_model.predictor.args.conf,
                    yolo_model.predictor.args.iou,
                    yolo_model.predictor.args.classes,
                    yolo_model.predictor.args.agnostic_nms,
                    max_det=yolo_model.predictor.args.max_det,
                    nc=len(yolo_model.predictor.model.names),
                    end2end=getattr(yolo_model.predictor.model, "end2end", False),
                    rotated=yolo_model.predictor.args.task == "obb",
                )

                # Normalize predictions and store them in the correct order
                for j, pred in enumerate(preds):
                    if pred is not None and len(pred) > 0:
                        pred[:, 0] /= w  # Normalize x1
                        pred[:, 1] /= h  # Normalize y1
                        pred[:, 2] /= w  # Normalize x2
                        pred[:, 3] /= h  # Normalize y2
                    
                    predictions[batch_indices[j]] = pred  # Store at the original index
                    
        results = [r.cpu().numpy() for r in predictions]
        return results

class YOLOPredictorCPU(YOLOPredictor):
    def predict(self, imgs):
        results = [self.yolo_models[0][0](i,
                    imgsz=self.imgsz, conf=self.conf, verbose=False)[0] for i in imgs]
        for i,r in enumerate(results):
            boxes = r.boxes.xyxyn.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().reshape(-1, 1)
            conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
            results[i] = np.hstack([boxes,conf,cls])
        return results

# class SlidingWindowBlock(PipeBlock):
#     def __init__(self, window_size=(1280,1280), stride=None):
#         """
#         Splits an image into sliding windows.

#         :param batched_img: NumPy array (H, W, C) or PyTorch tensor (C, H, W)
#         :param window_size: Tuple (window_height, window_width)
#         :param stride: Tuple (stride_height, stride_width), default is window_size (non-overlapping)
#         """
#         self.title = "sliding_window"
#         self.window_size = window_size
#         self.stride = stride if stride else window_size

#     def test_forward(self, imgs, meta):
#         self.test_forward_one_img(imgs[0],meta)
#         return self.forward(imgs,meta)
    
#     def forward(self, imgs, meta={}):        
#         meta[StaticWords.sliding_window_input_imgs] = imgs
#         output_offsets = []
#         new_imgs = []
        
#         for img in imgs:
#             out_imgs,offs = self.forward_one_img(img)
#             new_imgs += out_imgs
#             output_offsets.append(offs)

#         meta[StaticWords.sliding_window_output_offsets] = output_offsets
#         return new_imgs,meta
    
#     def test_forward_one_img(self, batched_img, meta):
#         """
#         Validate input image format.
#         """
#         if isinstance(batched_img, np.ndarray):
#             if batched_img.ndim not in [3, 4]:
#                 raise ValueError("NumPy input must have shape (H, W, C) or (B, H, W, C).")
#         elif isinstance(batched_img, torch.Tensor):
#             if batched_img.dim() not in [3, 4]:
#                 raise ValueError("PyTorch input must have shape (C, H, W) or (B, C, H, W).")
#         else:
#             raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
        
#         return self.forward_one_img(batched_img)

#     def forward_one_img(self, batched_img):
#         """
#         Splits an image into sliding windows.

#         Returns:
#             windows: NumPy array or PyTorch tensor with shape:
#                      - NumPy: (num_windows, window_height, window_width, C)
#                      - PyTorch: (num_windows, C, window_height, window_width)
#             offsets_xyxy: List of bounding boxes (x1, y1, x2, y2) for each window.
#         """
#         if isinstance(batched_img, np.ndarray):
#             imgs,offs = self._split_numpy(batched_img)
#         elif isinstance(batched_img, torch.Tensor):
#             imgs,offs = self._split_torch(batched_img)
#         else:
#             raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
#         return imgs,offs

#     def _split_numpy(self, data):
#         """
#         Split a NumPy array into sliding windows.
#         """
#         if data.ndim == 4:  # If batched, process each image
#             batch_windows = []
#             batch_offsets = []
#             for i in range(data.shape[0]):
#                 windows, offsets = self._split_numpy_single(data[i])
#                 batch_windows.append(windows)
#                 batch_offsets.append(offsets)
#             batch_windows, batch_offsets = np.concatenate(batch_windows, axis=0), batch_offsets
#             return batch_windows, batch_offsets
#         else:
#             windows, offsets = self._split_numpy_single(data)
#         return windows, offsets


#     def _split_torch(self, data):
#         """
#         Split a PyTorch tensor into sliding windows.
#         """
#         if data.dim() == 4:  # If batched, process each image
#             batch_windows = []
#             batch_offsets = []
#             for i in range(data.shape[0]):
#                 windows, offsets = self._split_torch_single(data[i])
#                 batch_windows.append(windows)
#                 batch_offsets.append(offsets)
#             return torch.cat(batch_windows, dim=0), batch_offsets
#         else:
#             return self._split_torch_single(data)

#     def _split_numpy_single(self, data):
#         """
#         Split a single NumPy image (H, W, C) into sliding windows.
#         """
#         H, W, C = data.shape
#         wH, wW = self.window_size
#         sH, sW = self.stride

#         if wH > H or wW > W:
#             raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

#         windows_list = []
#         offsets_xyxy = []

#         for row_start in range(0, H - wH + 1, sH):
#             for col_start in range(0, W - wW + 1, sW):
#                 window = data[row_start: row_start + wH, col_start: col_start + wW, :]
#                 windows_list.append(window)
#                 # offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))
#                 offsets_xyxy.append((col_start, row_start, col_start, row_start))

#         return windows_list, offsets_xyxy
    
#     def _split_torch_single(self, data):
#         """
#         Split a single PyTorch image (C, H, W) into sliding windows.
#         """
#         C, H, W = data.shape
#         wH, wW = self.window_size
#         sH, sW = self.stride

#         if wH > H or wW > W:
#             raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

#         windows_list = []
#         offsets_xyxy = []

#         for row_start in range(0, H - wH + 1, sH):
#             for col_start in range(0, W - wW + 1, sW):
#                 window = data[:, row_start: row_start + wH, col_start: col_start + wW]
#                 windows_list.append(window)
#                 # offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))
#                 offsets_xyxy.append((col_start, row_start, col_start, row_start))

#         return torch.cat(windows_list, dim=0), offsets_xyxy 

# class SlidingWindowMergeBlock(PipeBlock):
#     def __init__(self):
#         """
#         Merges YOLO detections from sliding windows back into the original image coordinates.
#         """
#         self.title = "sliding_window_merge"

#     def _extract_preds(self, preds):
#         """
#         Extracts YOLO-style detections from various possible data structures.

#         :param preds: Detection results, potentially in different formats.
#         :return: NumPy array of shape (N, 6) -> [x1, y1, x2, y2, confidence, class]
#         """
#         if hasattr(preds, 'pred'):
#             preds = preds.pred
#             preds = np.vstack([d.cpu().numpy() for d in preds])
#         elif hasattr(preds, 'boxes'):
#             # Extract boxes, confidence scores, and class labels
#             bs = preds.boxes
#             xyxy = bs.xyxy.cpu().numpy()
#             conf = bs.conf.cpu().numpy()
#             cls = bs.cls.cpu().numpy()
#             preds = np.hstack([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)])
#         return preds

#     def test_forward(self, batched_img, meta):
#         return self.forward(batched_img, meta)
    
#     def forward(self, imgs, meta:dict={}):
#         """
#         Merges sliding window detections back into the original image space.

#         :param imgs: List of split image windows.
#         :param meta: Metadata dictionary containing detection results and window offsets.
#         :return: Tuple (original_images, updated_meta)
#         """
#         # Retrieve detections and transformations from metadata
#         multi_dets = meta[StaticWords.yolo_results]
#         yolo_results = []
#         trans = meta[StaticWords.sliding_window_output_offsets]        
#         splits = len(trans[0])
#         multi_dets = [multi_dets[i:i+splits] for i in range(0,len(multi_dets),splits)]
#         for ii, (img, preds) in enumerate(zip(imgs, multi_dets)):
#             if len(preds) == 0: continue
#             trans_xyxy = trans[ii]
#             preds = [self._extract_preds(p) for p in preds]
#             # Adjust detection bounding boxes based on window offsets
#             for i, p in enumerate(preds):
#                 if len(p) == 0: continue
#                 if isinstance(p, torch.Tensor):
#                     p = p.cpu().numpy()
#                 p[:,:4] += trans_xyxy[i]  # Shift bounding boxes back to original image space
#                 preds[i] = p

#             # Stack detections into a single array
#             preds = np.vstack(preds)

#             # ---------- Apply Non-Maximum Suppression (NMS) ----------
#             boxes = torch.tensor(preds[:, :4])  # (x1, y1, x2, y2)
#             scores = torch.tensor(preds[:, 4])  # Confidence scores
#             class_ids = torch.tensor(preds[:, 5])  # Class labels

#             # Perform NMS using PyTorch
#             keep_indices = torch.ops.torchvision.nms(boxes, scores, 0.15)
#             preds = preds[keep_indices.numpy()]  # Keep only high-confidence, non-overlapping detections

#             yolo_results.append(preds)

#         meta[StaticWords.yolo_results] = yolo_results
#         raw_imgs = meta[StaticWords.sliding_window_input_imgs]
#         meta[StaticWords.yolo_input_imgs] = raw_imgs
#         return raw_imgs,meta
    
# class OriginalDetectionResults(ultralytics.engine.results.Results):
#   def __init__(self, results):
#     super().__init__(
#         results.orig_img,
#         path=results.path,
#         names=results.names,
#         boxes=results.boxes.data,
#         )

#   # 検出結果を描画する
#   # shape は出力画像のサイズ
#   def plot(self, font_size=None, colors=[], rect_padding=0, shape=(480, 640, 3)):
#     names = self.names
#     pred_boxes = self.boxes
#     if shape == self.orig_img.shape:
#       orig_img = self.orig_img.copy()
#     else:
#       orig_img = cv2.resize(self.orig_img, shape[1::-1])
    
#     start = time.time()
#     annotator = ultralytics.utils.plotting.Annotator(
#         orig_img,
#         font_size=font_size,
#         example=names,
#         )
#     for d in reversed(pred_boxes):
#       c, conf = int(d.cls), float(d.conf)
#       name = names[c]
#       label = f"{name} {conf:.2f}"
#       box = ((d.xyxyn.reshape(-1, 2))
#              * torch.tensor(shape[1::-1], device=d.xyxyn.device)).reshape(4)
#       box[:2] -= rect_padding
#       box[2:] += rect_padding
#       ultralytics.utils.ops.clip_boxes(box, orig_img.shape)
#       if c < len(colors):
#         color = colors[c]
#       else:
#         color = ultralytics.utils.plotting.colors(c, True)
#       annotator.box_label(box, label, color=color)
#     frame = annotator.result()
#     print(f"#### profile #### {1/(time.time()-start+1e-5):.2f} FPS - annotator.result")
#     return frame
  
def hex2rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA format."""
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    alpha = int(hex_color[6:8], 16) if len(hex_color) == 8 else 255
    return rgba + (alpha,)

class NumpyImageMask:
    def __init__(
        self,
        mask_image_path: Optional[str],
        mask_color: str = "#00000080",
        mask_split: Tuple[int, int] = (1, 1),
    ):
        """Initialize the NumpyImageMask class."""
        self.title = "numpy_image_mask"
        self._masks = self._make_mask_images(mask_image_path, mask_split, mask_color)
        if self._masks is None:
            print('Warning: no mask image loaded. This NumpyImageMask bloack will do nothing.')

    def gray2rgba_mask_image(self, gray_mask_img: np.ndarray, hex_color: str) -> Image.Image:
        """Convert a grayscale mask to an RGBA image with the specified color."""
        select_color = np.array(hex2rgba(hex_color), dtype=np.uint8)
        background = np.array([255, 255, 255, 0], dtype=np.uint8)

        condition = gray_mask_img == 0
        condition = condition[..., None]  # Expand dims for broadcasting
        color_mask_img = np.where(condition, select_color, background)

        return Image.fromarray(cv2.cvtColor(color_mask_img, cv2.COLOR_BGRA2RGBA))

    def _make_mask_images(
        self, mask_image_path: Optional[str], mask_split: Tuple[int, int], preview_color: str
    ) -> Optional[Dict[str, List[Any]]]:
        """Load and process the mask images."""
        if mask_image_path is None:
            return None

        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
        if mask_image is None:
            raise ValueError(f"Error: Unable to read mask image from {mask_image_path}")

        # Split mask into sub-masks
        try:
            mask_images = [
                sub_mask
                for row in np.split(mask_image, mask_split[0], axis=0)
                for sub_mask in np.split(row, mask_split[1], axis=1)
            ]
        except ValueError:
            print("Error: Invalid mask split configuration.")
            return None

        # Convert to grayscale and apply preview color
        gray_masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in mask_images]
        preview_masks = [self.gray2rgba_mask_image(gray, preview_color) for gray in gray_masks]

        return {"original": mask_images, "preview": preview_masks}

    def __call__(self, imgs: List[np.ndarray], meta: Optional[Dict] = None) -> Tuple[List[np.ndarray], Dict]:
        """Call forward method."""
        return self.forward(imgs, meta)

    def forward(self, imgs: List[np.ndarray], meta: Optional[Dict] = None) -> Tuple[List[np.ndarray], Dict]:
        """Forward pass applying masks."""
        return self.mask_images(imgs), meta or {}

    def test_forward(self, imgs: List[np.ndarray], meta: Optional[Dict] = None) -> Tuple[List[np.ndarray], Dict]:
        """Test forward pass applying masks."""
        if self._masks is None:
            return imgs, meta or {}
        
        if len(imgs) != len(self._masks["original"]):
            raise ValueError("Number of imgs and masks do not match.")
        
        return self.mask_images(imgs), meta or {}

    def mask_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply mask images to input images."""
        if self._masks is None:
            return images
        if self._masks.get("resized_masks") is None:
            self._masks["resized_masks"] = [
            cv2.resize(mask, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            for image, mask in zip(images, self._masks["original"])
        ]
        resized_masks = self._masks["resized_masks"]
        return [
            cv2.bitwise_and(image, mask) for image, mask in zip(images, resized_masks)
        ]

    def alpha_composite_mask_image(self, img: np.ndarray, mask_img: Image.Image) -> np.ndarray:
        """Apply alpha compositing to blend the mask with the image."""
        source_img = Image.fromarray(img)
        source_img.putalpha(255)

        mask = mask_img.resize(source_img.size, Image.Resampling.NEAREST)
        blended_img = Image.alpha_composite(source_img, mask).convert("RGB")

        return np.array(blended_img)

    def mask_previews(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply preview masks to images."""
        if self._masks is None:
            return images

        if len(images) != len(self._masks["preview"]):
            print("Error: Number of images and masks do not match.")
            return images

        return [
            self.alpha_composite_mask_image(image, mask.resize(image.shape[1::-1], Image.Resampling.NEAREST))
            for image, mask in zip(images, self._masks["preview"])
        ]

    def mask_yolo_result(self, results: List[Any]) -> List[Any]:
        """Apply mask filtering to results."""
        if self._masks is None:
            return results
        for res, mask_image in zip(results, self._masks["original"]):
            mask_array = mask_image.max(2)
            shape = mask_array.shape
            yx = (res.boxes.xywhn[:, :2].cpu().numpy()[:, ::-1] * shape).astype(int)
            pos = yx[:, 0].clip(0, shape[0] - 1), yx[:, 1].clip(0, shape[1] - 1)
            keep = mask_array[pos] > 0
            res.update(boxes=res.boxes.data[keep])

        return results


class DrawPredictionsBlock(PipeBlock):
    def __init__(self, class_names=None, class_colors=None):
        super().__init__()
        self.title = 'draw_predictions'
        
        # Assign class names
        self.class_names = class_names or list(range(100000))
        self.class_names = {i: k for i, k in enumerate(self.class_names)}

        # Assign or generate colors
        if class_colors is None:
            self.class_colors = self._generate_class_colors(len(self.class_names))
        elif type(class_colors)==list:
            self.class_colors = {i:v for i,v in enumerate(class_colors)}
        else:
            self.class_colors = class_colors

    def forward(self, imgs, meta={}):
        preds = meta.get(StaticWords.yolo_results, [])
        imgs = [self.draw_predictions(i, self._extract_preds(p)) for i, p in zip(imgs, preds)]
        return imgs, meta

    def test_forward(self, imgs: List[Any], meta: Dict = {}):
        if StaticWords.yolo_input_imgs not in meta:
            raise ValueError('yolo input raw imgs is needed.')
        if StaticWords.yolo_results not in meta:
            raise ValueError('yolo_results is needed.')
        if len(meta[StaticWords.yolo_input_imgs]) != len(meta[StaticWords.yolo_results]):
            raise ValueError('yolo input raw imgs and yolo_results must be of the same size.')
        return self.test_forward_numpy_rgb(imgs, meta)

    def draw_predictions(self, image, predictions):
        """Draw bounding boxes with class-specific colors."""
        image_with_boxes = image.copy()
        img_height, img_width = image.shape[:2]
        # Dynamically scale font size based on image size
        font_scale = max(0.5, min(img_width, img_height) / 600)  # Scales with image size
        thickness = max(1, int(font_scale * 2))  # Adjust thickness based on scale

        for pred in predictions:
            if len(predictions) == 0: continue

            x1, y1, x2, y2, confidence, class_id = pred[:6]
            x1, y1, x2, y2 = map(int, [x1*img_width, y1*img_width, x2*img_width, y2*img_width])
            
            # Get color for the class
            color = self.class_colors.get(int(class_id), (255, 255, 255))  # Default to white
            
            
            # Draw rectangle
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

            # Label text
            label = f"{self.class_names.get(int(class_id), 'null')}: {confidence:.2f}"
            
            # Get text size and adjust padding
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_w, text_h = text_size
            text_x, text_y = x1, max(0, y1 - 10)  # Adjust y position to avoid going out of frame
            
            # Draw filled rectangle for text background
            cv2.rectangle(image_with_boxes, 
                        (text_x, text_y - text_h - 5), 
                        (text_x + text_w + 5, text_y), 
                        color, 
                        -1)
            
            # Put text
            cv2.putText(image_with_boxes, label, 
                        (text_x + 2, text_y - 2),  # Slight offset for better visibility
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                        (0, 0, 0), thickness)

        return image_with_boxes

    def _extract_preds(self, preds):
        """Extracts predictions from YOLO output."""
        if hasattr(preds, 'pred'):
            preds = preds.pred
            preds = np.vstack([d.cpu().numpy() for d in preds])
        elif hasattr(preds, 'boxes'):
            bs = preds.boxes
            xyxy = bs.xyxy.cpu().numpy()
            conf = bs.conf.cpu().numpy()
            cls = bs.cls.cpu().numpy()
            preds = np.hstack([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)])
        return preds

    def _generate_class_colors(self, num_classes):
        """Generates distinct colors for each class."""
        return {i: tuple(random.randint(0, 255) for _ in range(3)) for i in range(num_classes)}
