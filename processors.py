
# Standard Library Imports
import enum
import math
import random
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Library Imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops


class Processors:
    torch_img_dtype = torch.float16
    numpy_img_dtype = np.uint8

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

        class Layout(enum.Enum):
            """Possible Bayer color filter array layouts.

            The value of each entry is the color index (R=0,G=1,B=2)
            within a 2x2 Bayer block.
            """

            RGGB = (0, 1, 1, 2)
            GRBG = (1, 0, 2, 1)
            GBRG = (1, 2, 0, 1)
            BGGR = (2, 1, 1, 0)

        def __init__(self, layout = None):
            if layout is None : layout = self.__class__.Layout.RGGB
            super(self.__class__, self).__init__()
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
                self.__class__.Layout.RGGB: rggb,
                self.__class__.Layout.GRBG: torch.roll(rggb, 1, -1),
                self.__class__.Layout.GBRG: torch.roll(rggb, 1, -2),
                self.__class__.Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
            }.get(layout)

    class DummyDebayer5x5(torch.nn.Module):
        def __init__(self):
            super(self.__class__, self).__init__()

        def forward(self, x):
            # Repeat the single channel 3 times to create an RGB image
            return x.repeat(1, 3, 1, 1)  # Shape: (N, 3, H, W)

    class ImageVerifier:
        def _img_w_h_c(self,im):
            
            if isinstance(im, np.ndarray):
                if len(im.shape)==2:
                    height,width = im.shape
                    channel = 1
                elif len(im.shape)==3:
                    height,width,channel = im.shape
                else:
                    raise ValueError('numpy image shape must be H,W or H,W,C')

            if isinstance(im, torch.Tensor):
                if len(im.shape)==3:
                    # C, H, W = x.shape
                    channel,height,width = im.shape
                elif len(im.shape)==4:
                    # B, C, H, W = x.shape
                    b,channel,height,width = im.shape
                else:
                    raise ValueError('torch image shape must be C,H,W or B,C,H,W')
            if channel not in [1,3]:
                raise ValueError(f'image channel is not 1 or 3, it is {channel}')

            return width,height,channel

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
            if img.dtype != Processors.numpy_img_dtype:
                return self._raise_error(f"NumPy image at index {idx} has invalid dtype {img.dtype}. Expected uint8.", strict)

        def verify_type_torch(self, img: torch.Tensor, idx: int, strict: bool = True) -> bool:
            """Verify that a Torch image is of type."""
            if not isinstance(img, torch.Tensor):
                return self._raise_error(f"Expected a Torch tensor at index {idx}, but got {type(img)}.", strict)
            if img.dtype not in [Processors.torch_img_dtype]:#[torch.float16, torch.float32, torch.float64]:
                return self._raise_error(f"Torch image at index {idx} has invalid dtype {img.dtype}. Expected {Processors.torch_img_dtype}.", strict)

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
            self._enable = True

        def on(self):
            self._enable = True

        def off(self):
            self._enable = False

        def __call__(self, imgs: List[Any], meta: Dict = {}):
            # start = time.time()
            if self._enable:
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
        
        before_resize_img_w_h ='before_resize_img_w_h'

        red_mask_imgs ='red_mask_imgs'

        sliding_window_input_img_w_h ='sliding_window_input_img_w_h'
        sliding_window_size = 'sliding_window_size'
        sliding_window_input_imgs = 'sliding_window_input_imgs'
        sliding_window_imgs_idx = 'sliding_window_imgs_idx'
        sliding_window_output_offsets = 'sliding_window_output_offsets'

    class LambdaBlock(PipeBlock):
        def __init__(self,forward_func=lambda imgs,meta:(imgs,meta),title='lambda'):
            super().__init__(title)
            self.forward_func = forward_func

        def forward(self, imgs, meta={}):
            return self.forward_func(imgs,meta)

        def test_forward(self, imgs: List[Any], meta: Dict = {}):
            return self.forward(imgs,meta)
        
    class CvDebayerBlock(PipeBlock):
        def __init__(self,formart=cv2.COLOR_BAYER_BG2RGB):
            super().__init__('cv_debayer')
            self.formart = formart

        def test_forward(self, imgs: List[Any], meta: Dict = {}):
            return self.test_forward_numpy_bayer(imgs,meta)
        
        def forward(self, imgs, meta={}):
            debayered_imgs = [cv2.cvtColor(img,self.formart) for img in imgs]
            return debayered_imgs, meta

    class TorchDebayerBlock(PipeBlock):
        def __init__(self, dtype=None):
            if dtype is None : dtype=Processors.torch_img_dtype
            """
            Converts a batched Bayer image (B, 1, H, W) into an RGB image (B, 3, H, W) using Torch's Debayer5x5.

            :param batched_img: PyTorch tensor of shape (B, 1, H, W) with dtype uint8.
            """
            super().__init__('torch_debayer')
            self.num_gpus = self.gpu_info()
            self.dtype = dtype
            
            self.debayer_models = []
            for i in range(self.num_gpus):
                device = f"cuda:{i}"
                """
                Convert a Bayer raw image (B, 1, H, W) tensor to an RGB image (B, 3, H, W) using PyTorch's Debayer5x5.
                """
                model = Processors.Debayer5x5().to(device).to(self.dtype)
                self.debayer_models.append((model, device))

        def test_forward(self, imgs, meta={}):
            self.test_forward_torch_bayer(imgs, meta)
            return self.forward(imgs,meta)

        def forward(self, imgs: List[torch.Tensor], meta={}):
            torch_imgs = []
            for i,img in enumerate(imgs):
                model,device = self.debayer_models[i%self.num_gpus]
                torch_imgs.append(model(img.unsqueeze(0)))
            return torch_imgs, meta

    class TorchRGBToNumpyBGRBlock(PipeBlock):
        def __init__(self):
            super().__init__('torch_to_numpy')

        def forward(self, imgs: List[np.ndarray], meta={}):
            # [ [1,3,h,w], ...]
            imgs = [
                img.mul(255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                for i,img in enumerate(imgs)
            ]
            
            imgs = [
                img.cpu().numpy()[0][:,:,::-1] # to BGR
                for i,img in enumerate(imgs)
            ]
            return imgs, meta

        def test_forward(self, imgs: List[np.ndarray], meta: Dict = {}):
            self.test_forward_torch_rgb(imgs, meta)
            return self.forward(imgs, meta)
        
    class NumpyBGRToTorchRGBBlock(PipeBlock):
        def __init__(self,dtype=None):
            if dtype is None : dtype=Processors.torch_img_dtype
            super().__init__("numpy_rgb_to_torch")
            self.num_gpus = self.gpu_info()
            self.dtype = dtype

            def get_model(device,dtype=dtype):
                return lambda img:torch.tensor(img[:,:,::-1].copy(),
                            dtype=dtype, device=device).div(255.0).permute(2, 0, 1).unsqueeze(0)
            
            self.tensor_models = []
            for i in range(self.num_gpus):
                device = f"cuda:{i}"
                model = get_model(f"cuda:{i}")
                self.tensor_models.append((model, device))


        def forward(self, imgs: List[np.ndarray], meta={}):
            # print([i.shape for i in imgs])
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
        def __init__(self,dtype=None):
            if dtype is None : dtype=Processors.torch_img_dtype
            super().__init__("numpy_bayer_to_torch")
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
                for i,img in enumerate(imgs)
            ]
            return torch_imgs, meta

        def test_forward(self, imgs: List[Any], meta: Dict = {}):
            return self.test_forward_numpy_bayer(imgs,meta)

    class TorchResizeBlock(PipeBlock):
        def __init__(self, output_size: tuple=(1280,1280), mode: str = 'bilinear', 
                    align_corners: bool = False,
                    target = None):
            super().__init__('torch_resize')
            self.target = target
            self.output_size = output_size  # (height, width)
            self.mode = mode
            self.align_corners = align_corners

        def forward(self, imgs: List[torch.Tensor], meta={}):
            if self.target and self.target in meta:
                imgs = meta[self.target]
            resized_imgs: List[torch.Tensor] = []
            for img_tensor in imgs:
                resized_tensor = F.interpolate(img_tensor,
                        size=self.output_size, mode=self.mode, align_corners=self.align_corners)
                resized_imgs.append(resized_tensor)
            return resized_imgs, meta    

        def test_forward(self, imgs: List[Any], meta: Dict = {}):
            width,height,c = self._img_w_h_c(imgs[0])
            meta[Processors.StaticWords.before_resize_img_w_h] = width,height
            return self.test_forward_torch_rgb(imgs,meta) # B,C,H,W

    class CVResizeBlock(PipeBlock):
        def __init__(self, output_size: tuple=(1280,1280), interpolation=cv2.INTER_LINEAR):
            super().__init__('cv_resize')
            self.output_size = output_size  # (width, height)
            self.interpolation = interpolation

        def forward(self, imgs, meta={}):
            resized_imgs = []
            for img in imgs:
                resized_img = cv2.resize(
                    img, self.output_size, interpolation=self.interpolation)
                resized_imgs.append(resized_img)
            return resized_imgs, meta

        def test_forward(self, imgs: List[Any], meta: Dict = {}):        
            width,height,c = self._img_w_h_c(imgs[0])
            meta[Processors.StaticWords.before_resize_img_w_h] = width,height
            return self.test_forward_numpy_rgb(imgs,meta)

    class TileImagesBlock(PipeBlock):
        def __init__(self, tile_width=2):
            super().__init__('tile_images')
            self.tile_width = tile_width
        
        def forward(self, imgs, meta={}):
            # Get image dimensions
            h, w = imgs[0].shape[:2]
            num_images = len(imgs)
            tile_height = (num_images + self.tile_width - 1) // self.tile_width  # Ensures enough rows

            # Create a blank canvas
            tile_width = math.ceil(num_images / tile_height)
            tile = np.zeros((tile_height * h, tile_width * w, 3), dtype=imgs[0].dtype)

            # Place images in the tile
            for i, img in enumerate(imgs):
                r, c = divmod(i, tile_width)
                tile[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

            return [tile], meta
        

        def test_forward(self, imgs: List[Any], meta: Dict = {}):
            return self.test_forward_numpy_rgb(imgs,meta)
    
    class EncodeJpegBlock(PipeBlock):
        def __init__(self, quality=None):
            super().__init__('encode_jpeg')
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
            super().__init__('merge_yolo_results')

        def forward(self, imgs, meta={}):
            results = meta.get(Processors.StaticWords.yolo_results,[])

            if len(results) == 1:
                result = results[0]        
            # Merge YOLO detection results        
            elif hasattr(results[0],'boxes'):
                boxes = torch.cat([res.boxes.data.cpu() for res in results])
                # Create a new result object and update it with merged boxes
                result = results[0].new()
                result.update(boxes=boxes)
            elif isinstance(results[0],np.ndarray):
                result = np.vstack(results)

            meta[Processors.StaticWords.yolo_results] = results
            return imgs, meta
        
        def test_forward(self, imgs, meta = {}):
            results = meta.get(Processors.StaticWords.yolo_results)
            if not results:
                raise ValueError("yolo results list cannot get.")
            return self.forward(imgs, meta)
        
    class YOLOPredictor(PipeBlock):
        def __init__(
            self, model_path,  # Path to YOLO model (e.g., 'yolov8n.pt')
            imgsz=640,
            conf=0.25,
            class_names=None,
            non_notify_classes=None,
            conf_thres_per_class=None,
            conf_thres_per_class_rlfb=None,
            dtype=None):
            if dtype is None : dtype=Processors.torch_img_dtype
            super().__init__('yolo_predictor')
            self.dtype = dtype
            self.model_path = model_path
            self.imgsz = imgsz
            self.conf = conf
            self.iou = -1
            self.class_names = class_names or []
            self.non_notify_classes = non_notify_classes or []
            self.conf_thres_per_class = conf_thres_per_class or {}
            self.conf_thres_per_class_rlfb = conf_thres_per_class_rlfb or [[], [], [], []]
            
            conf_thres_per_class = [i.split('=') for i in conf_thres_per_class]
            self.conf_thres_per_class = {k: v for k, v in conf_thres_per_class}
            self.conf_thres_per_class_rlfb = [
                {k: v for k, v in conf_thres} for conf_thres in conf_thres_per_class_rlfb
            ]
            self.yolo_models = [(YOLO(model_path).to('cpu'),'cpu')]
        
        def forward(self, imgs, meta={}):
            res = self.predict(imgs)
            res = self.postprocess(res)

            meta[Processors.StaticWords.yolo_input_imgs] = imgs
            meta[Processors.StaticWords.yolo_results] = res
            meta[self.title] = res
            return imgs, meta
        
        def test_forward(self, imgs, meta = {}):
            return self.forward(imgs,meta)
        
        def predict(self, imgs)-> list[np.ndarray]: # [(N,5) ... ] x1,x2,y1,y2,conf,id : x,y is persentages
            raise NotImplemented('this is a interface class!')

        def postprocess(self, results):
            # Apply confidence threshold filtering
            results = self.remove_classes(results, self.non_notify_classes)
            results = self.filter_conf_per_class(results, self.conf_thres_per_class) 
            return results

        def remove_classes(self,results,non_notify_classes):
            if len(self.non_notify_classes)==0:  return results
            for i,res in enumerate(results):
                if len(res)==0:continue
                labels = res[:,5]
                keep = [int(l) not in non_notify_classes for l in labels]
                results[i]=res[keep]                
            return results
        
        def filter_conf_per_class(self, results, conf_thres_per_class:dict):
            """Filters results based on class-specific confidence thresholds."""
            if len(conf_thres_per_class)==0: return results
            for i,result in enumerate(results):
                if len(result)==0:continue
                confs = result[:,4]
                labels = result[:,5]
                keep = [(conf_thres_per_class.get(int(l), 0) < c) for c, l in zip(confs, labels)]
                results[i]=result[keep]
            return results

    class YOLOPredictorGPU(YOLOPredictor):
        def test_forward(self, imgs, meta = {}):
            self.gpu_info()
            # Load models onto separate GPUs
            if torch.cuda.is_available():
                self.yolo_models = []
                for i in range(self.num_gpus):
                    device = f"cuda:{i}"
                    model = YOLO(self.model_path).to(device)
                    self.yolo_models.append((model, device))
                print("[YOLOPredictor] Models loaded on available GPUs:", [device for _, device in self.yolo_models])

            for i in range(self.num_gpus):
                batch = [(i * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
                        for i in imgs[i::self.num_gpus]]
                (yolo_model,device) = self.yolo_models[i]

                batch = [i.astype(np.uint8) for i in batch]
                if len(batch)==0:continue
                half = self.dtype == torch.float16
                # run one time and init
                yolo_model(batch, imgsz=self.imgsz, conf=self.conf, half=half)

                self.iou = yolo_model.predictor.args.iou
                self.classes = yolo_model.predictor.args.classes
                self.agnostic_nms = yolo_model.predictor.args.agnostic_nms
                self.max_det = yolo_model.predictor.args.max_det
                self.nc = len(yolo_model.predictor.model.names)
                self.end2end = getattr(yolo_model.predictor.model, "end2end", False)
                self.rotated = yolo_model.predictor.args.task == "obb"
                    
            self.imgs_device_dict = {d:[] for d in set([i.device.index for i in imgs])}
            for i,img in enumerate(imgs):
                self.imgs_device_dict[img.device.index].append(i)
            print(self.imgs_device_dict)

            return self.forward(imgs,meta)
        
        def predict(self, imgs):
            """
            Run YOLO inference in parallel across GPUs.
            """       
            predictions = [None] * len(imgs)  # Placeholder for ordered predictions
            res = []

            for i in range(min(self.num_gpus,len(imgs))):
                batch_indices = np.asarray(self.imgs_device_dict[i])  # Track original indices
                batch = [imgs[idx] for idx in batch_indices]  # Assign correct images to batch
                if len(batch)==0:continue

                (yolo_model, device) = self.yolo_models[i]
                
                with torch.no_grad():
                    tmp = torch.vstack([img for img in batch])  # Move batch to correct GPU
                    b, c, h, w = tmp.shape
                    res.append([yolo_model.model(tmp),h, w])

            # do it later for parallel GPU infer
            for i,r in enumerate(res):
                batch_indices = np.asarray(self.imgs_device_dict[i])
                (preds, feature_maps), h, w = r
                preds = ops.non_max_suppression(
                    preds,
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic_nms,
                    
                    max_det = self.max_det,
                    nc =      self.nc,
                    end2end = self.end2end,
                    rotated = self.rotated,
                )
                
                # Normalize predictions and store them in the correct order
                for j, pred in enumerate(preds):
                    if pred is not None and len(pred) > 0:
                        pred[:, 0] /= w  # Normalize x1
                        pred[:, 1] /= h  # Normalize y1
                        pred[:, 2] /= w  # Normalize x2
                        pred[:, 3] /= h  # Normalize y2
                    
                    predictions[batch_indices[j]] = pred  # Store at the original index
                

            results = [r.cpu().numpy() if r is not None else [] for r in predictions]
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

    class SlidingWindowBlock(PipeBlock):
        def __init__(self, window_size=(1280,1280), stride=None):
            """
            Splits an image into sliding windows.

            :param batched_img: NumPy array (H, W, C) or PyTorch tensor (C, H, W)
            :param window_size: Tuple (window_height, window_width)
            :param stride: Tuple (stride_height, stride_width), default is window_size (non-overlapping)
            """
            super().__init__("sliding_window")
            self.window_size = window_size
            self.stride = stride if stride else window_size

        def test_forward(self, imgs, meta):
            self.test_forward_one_img(imgs[0],meta)
            return self.forward(imgs,meta)
        
        def forward(self, imgs, meta={}):
            output_offsets = []
            new_imgs = []       
            imgs_idx = {}
            for i,img in enumerate(imgs):
                out_imgs,offs = self.forward_one_img(img,meta)
                imgs_idx[i] = list(range(len(new_imgs), len(new_imgs)+len(out_imgs)))
                new_imgs += out_imgs
                output_offsets.append(offs)
            meta[Processors.StaticWords.sliding_window_size] = self.window_size
            meta[Processors.StaticWords.sliding_window_imgs_idx] = imgs_idx
            meta[Processors.StaticWords.sliding_window_input_imgs] = imgs
            meta[Processors.StaticWords.sliding_window_output_offsets] = output_offsets
            return new_imgs,meta
        
        def test_forward_one_img(self, batched_img, meta):
            """
            Validate input image format.
            """
            if isinstance(batched_img, np.ndarray):
                if batched_img.ndim not in [3, 4]:
                    raise ValueError("NumPy input must have shape (H, W, C) or (B, H, W, C).")
            elif isinstance(batched_img, torch.Tensor):
                if batched_img.dim() not in [3, 4]:
                    raise ValueError("PyTorch input must have shape (C, H, W) or (B, C, H, W).")
            else:
                raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
            
            return self.forward_one_img(batched_img,meta)

        def forward_one_img(self, batched_img, meta={}):
            """
            Splits an image into sliding windows.

            Returns:
                windows: NumPy array or PyTorch tensor with shape:
                        - NumPy: (num_windows, window_height, window_width, C)
                        - PyTorch: (num_windows, C, window_height, window_width)
                offsets_xyxy: List of bounding boxes (x1, y1, x2, y2) for each window.
            """
            if isinstance(batched_img, np.ndarray):
                imgs,offs = self._split_numpy(batched_img,meta)
            elif isinstance(batched_img, torch.Tensor):
                imgs,offs = self._split_torch(batched_img,meta)
            else:
                raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
            
            res = []
            for i in imgs:
                res += i
            return res,offs

        def _split_numpy(self, data, meta={}):
            """
            Split a NumPy array into sliding windows.
            """
            if data.ndim == 4:  # If batched, process each image
                batch_windows = []
                batch_offsets = []
                for i in range(data.shape[0]):
                    windows, offsets = self._split_numpy_single(data[i],meta)
                    batch_windows.append(windows)
                    batch_offsets.append(offsets)
                batch_windows, batch_offsets = np.concatenate(batch_windows, axis=0), batch_offsets
                return batch_windows, batch_offsets
            else:
                windows, offsets = self._split_numpy_single(data,meta)
            return windows, offsets


        def _split_torch(self, data, meta={}):
            """
            Split a PyTorch tensor into sliding windows.
            """
            if data.dim() == 4:  # If batched, process each image
                batch_windows = []
                batch_offsets = []
                for i in range(data.shape[0]):
                    windows, offsets = self._split_torch_single(data[i],meta)
                    batch_windows.append(windows)
                    batch_offsets.append(offsets)
                return batch_windows, batch_offsets
            else:
                return self._split_torch_single(data,meta)

        def _split_numpy_single(self, data, meta={}):
            """
            Split a single NumPy image (H, W, C) into sliding windows.
            """
            H, W, C = data.shape
            wH, wW = self.window_size
            sH, sW = self.stride
            
            meta[Processors.StaticWords.sliding_window_input_img_w_h] = (W,H)
            if wH > H or wW > W:
                raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

            windows_list = []
            offsets_xyxy = []

            for row_start in range(0, H - wH + 1, sH):
                for col_start in range(0, W - wW + 1, sW):
                    window = data[row_start: row_start + wH, col_start: col_start + wW, :]
                    windows_list.append(window)
                    # offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))
                    offsets_xyxy.append((col_start, row_start, col_start, row_start))

            return windows_list, offsets_xyxy
        
        def _split_torch_single(self, data, meta={}):
            """
            Split a single PyTorch image (C, H, W) into sliding windows.
            """
            C, H, W = data.shape
            wH, wW = self.window_size
            sH, sW = self.stride
            
            meta[Processors.StaticWords.sliding_window_input_img_w_h] = (W,H)
            if wH > H or wW > W:
                raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

            windows_list = []
            offsets_xyxy = []

            for row_start in range(0, H - wH + 1, sH):
                for col_start in range(0, W - wW + 1, sW):
                    window = data[:, row_start: row_start + wH, col_start: col_start + wW]
                    windows_list.append(window.unsqueeze(0))
                    # offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))
                    offsets_xyxy.append((col_start, row_start, col_start, row_start))
            return windows_list, offsets_xyxy 

    class SlidingWindowMergeBlock(PipeBlock):
        def __init__(self):
            """
            Merges YOLO detections from sliding windows back into the original image coordinates.
            """
            super().__init__("sliding_window_merge")

        def _extract_preds(self, preds):
            """
            Extracts YOLO-style detections from various possible data structures.

            :param preds: Detection results, potentially in different formats.
            :return: NumPy array of shape (N, 6) -> [x1, y1, x2, y2, confidence, class]
            """
            if hasattr(preds, 'pred'):
                preds = preds.pred
                preds = np.vstack([d.cpu().numpy() for d in preds])
            elif hasattr(preds, 'boxes'):
                # Extract boxes, confidence scores, and class labels
                bs = preds.boxes
                xyxy = bs.xyxy.cpu().numpy()
                conf = bs.conf.cpu().numpy()
                cls = bs.cls.cpu().numpy()
                preds = np.hstack([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)])
            return preds

        def test_forward(self, batched_img, meta):
            return self.forward(batched_img, meta)
        
        def forward(self, imgs, meta:dict={}):
            """
            Merges sliding window detections back into the original image space.

            :param imgs: List of split image windows.
            :param meta: Metadata dictionary containing detection results and window offsets.
            :return: Tuple (original_images, updated_meta)
            """
            # Retrieve detections and transformations from metadata
            
            raw_imgs_idx = meta[Processors.StaticWords.sliding_window_imgs_idx]
            multi_dets = meta[Processors.StaticWords.yolo_results]
            trans = meta[Processors.StaticWords.sliding_window_output_offsets]
            raw_imgs = meta[Processors.StaticWords.sliding_window_input_imgs]        
            window_size = meta[Processors.StaticWords.sliding_window_size]
            W, H = meta[Processors.StaticWords.sliding_window_input_img_w_h]
            wH, wW = window_size

            if raw_imgs[0].dim()==4 and raw_imgs[0].shape[0]==1 and len(trans[0])==1:
                trans = [t[0] for t in trans]

            splits = len(raw_imgs_idx)
            yolo_results = []        
            multi_dets = [[np.asarray(multi_dets[ii]).reshape(-1,6) for ii in raw_imgs_idx[i]] for i in range(splits)]

            for ii, (img, preds) in enumerate(zip(raw_imgs, multi_dets)):
                if len(preds) == 0:
                    yolo_results.append([])
                    continue

                trans_xyxy = trans[ii]
                preds = [self._extract_preds(p) for p in preds]
                # Adjust detection bounding boxes based on window offsets
                for i, p in enumerate(preds):
                    if len(p) == 0: continue              
                    # (x1, y1, x2, y2)  
                    p[:, 0] = (p[:, 0]*(wW/W) + trans_xyxy[i][0]/W)
                    p[:, 1] = (p[:, 1]*(wH/H) + trans_xyxy[i][1]/H)
                    p[:, 2] = (p[:, 2]*(wW/W) + trans_xyxy[i][2]/W)
                    p[:, 3] = (p[:, 3]*(wH/H) + trans_xyxy[i][3]/H)
                    preds[i] = p

                # Stack detections into a single array
                preds = np.vstack(preds)
                # # ---------- Apply Non-Maximum Suppression (NMS) ----------
                boxes = torch.tensor(preds[:, :4])  # (x1, y1, x2, y2)
                scores = torch.tensor(preds[:, 4])  # Confidence scores
                class_ids = torch.tensor(preds[:, 5])  # Class labels

                # Perform NMS using PyTorch
                keep_indices = torch.ops.torchvision.nms(boxes, scores, 0.15)
                preds = preds[keep_indices.numpy()]  # Keep only high-confidence, non-overlapping detections

                yolo_results.append(preds)

            meta[Processors.StaticWords.yolo_results] = yolo_results
            
            raw_imgs = meta[Processors.StaticWords.sliding_window_input_imgs]
            meta[Processors.StaticWords.yolo_input_imgs] = raw_imgs
            return raw_imgs,meta
        
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

    class NumpyImageMask(PipeBlock):
        def __init__(
            self,
            mask_image_path: Optional[str],
            mask_color: str = "#00000080",
            mask_split: Tuple[int, int] = (2, 2),
        ):
            """Initialize the NumpyImageMask class."""
            super().__init__("numpy_image_mask")
            self.mask_split=mask_split
            
            if type(mask_image_path) is str:
                data = self._read_mask_image_file(mask_image_path)

            if type(mask_image_path) is np.ndarray:
                data = mask_image_path

            self._masks = self.set_mask_image_data(data)
            self._resized_masks = None
            self.need_adjust_mask = True
            if self._masks is None:
                print('[NumpyImageMask] Warning: no mask image loaded. This NumpyImageMask bloack will do nothing.')

        def set_mask_image_data(self,mask_image):                
            mask_split = self.mask_split
            # Split mask into sub-masks
            try:                
                if len(mask_image.shape) == 1:
                    num_imgs = len(mask_image)                
                    # mask_image = np.array([0,0,0,0])
                    mask_image = np.asarray(mask_image[:num_imgs])
                    mask_image = mask_image.reshape(int(num_imgs**0.5),int(num_imgs**0.5))
                    # mask_image = np.array([[0,1,],[1,0]])

                mask_images = [
                    sub_mask
                    for row in np.split(mask_image, mask_split[0], axis=0)
                    for sub_mask in np.split(row, mask_split[1], axis=1)
                ]
            except Exception as e:
                print(e)
                self._masks = []
                return self._masks 

            # Convert to grayscale and apply preview color
            self._masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
                    if len(mask.shape)>2 and mask.shape[-1]>1 else mask
                    for mask in mask_images]
            return self._masks 

        def _read_mask_image_file(self, mask_image_path: Optional[str]):
            """Load and process the mask images."""
            if mask_image_path is None:
                return None

            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
            if mask_image is None:
                raise ValueError(f"Error: Unable to read mask image from {mask_image_path}")
            
            return mask_image


        def _adjust_mask(self, images: List[np.ndarray]):
            w,h,c = self._img_w_h_c(images[0])
                
            self._resized_masks = [
                cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
                for mask in self._masks
            ]
            # print([i.shape for i in self._resized_masks])

            self._resized_masks = [
                mask if len(mask.shape) >= len(image.shape) else
                np.expand_dims(mask, axis=-1)
                for image, mask in zip(images, self._resized_masks)
            ]
            # print([i.shape for i in self._resized_masks])
            
            self._resized_masks = [
                mask if mask.shape[-1] == c else
                mask.repeat(c, axis=-1) 
                for mask in self._resized_masks
            ]
            
            self._resized_masks = [
                ((mask > 0) * 255).astype(np.uint8)
                for mask in self._resized_masks
            ]

        def clip(self,im,l=0,h=255):
            return np.clip(im,l,h).astype(np.uint8)

        def forward(self, imgs: List[np.ndarray], meta: dict = None, channel=2, h=255):
            if self.need_adjust_mask:
                self._adjust_mask(imgs)
                self.need_adjust_mask = False

            if len(self._resized_masks)==0:
                return imgs,meta
            
            """Apply mask images to input images."""
            if Processors.StaticWords.red_mask_imgs not in meta:
                meta[Processors.StaticWords.red_mask_imgs] = [i.copy() if hasattr(i,'copy') else i.clone() for i in imgs]
            
            for i,m in zip(meta[Processors.StaticWords.red_mask_imgs],self._resized_masks):
                i[:,channel,:,:] = self.clip(i[:,channel,:,:] + (h*0.2) *(h - m[:,channel,:,:]))

            imgs = [i * mask for i, mask in zip(imgs, self._resized_masks)]
            # imgs = [ i - i * (0.75 - 0.75 * mask) for i, mask in zip(imgs, self._resized_masks)]

            return imgs,meta
        
        def test_forward(self, imgs: List[np.ndarray], meta: dict = None):
            if len(imgs)!=len(self._masks) and len(self._masks)>0:
                raise ValueError('mask images numbers is not the same. len(imgs)!=len(masks)')
            return self.forward(imgs,meta)

    class TorchImageMask(NumpyImageMask):
        def __init__(
            self,
            mask_image_path: Optional[str],
            mask_color: str = "#00000080",
            mask_split: Tuple[int, int] = (2, 2),
            title = "torch_image_mask"
        ):
            """Initialize the TorchImageMask class."""
            super().__init__(mask_image_path,mask_color,mask_split)
            self.title = title
            self._torch_masks = [
                torch.from_numpy(i).unsqueeze(0).unsqueeze(0
                    ).to(Processors.torch_img_dtype)/255.0 for i in self._masks]
            self._torch_resized_masks = None
        
        def forward(self, imgs: List[np.ndarray], meta: dict = None, channel=0, h=1.0):
            return super().forward(imgs,meta,channel,h)
            
        def clip(self,im,l=0.0,h=1.0):
            return torch.clip(im,l,h)
        
        def _adjust_mask(self, images: List[torch.Tensor]):
            # print([i.shape for i in images])
            super()._adjust_mask(images)
            
            w,h,c = self._img_w_h_c(images[0])

            # print([i.shape for i in self._resized_masks])
            self._torch_resized_masks = [
                torch.from_numpy(mask.copy())
                for image, mask in zip(images, self._resized_masks)
            ]
            # print([i.shape for i in self._torch_resized_masks])
            self._torch_resized_masks = [
                mask.permute(2, 0, 1).to(Processors.torch_img_dtype).to(image.device).div(255.0)
                for image, mask in zip(images, self._torch_resized_masks)
            ]

            if c>1:
                self._torch_resized_masks = [
                    mask.unsqueeze(0)
                    for image, mask in zip(images, self._torch_resized_masks)
                ]

            self._resized_masks = self._torch_resized_masks
            
        
    class DrawPredictionsBlock(PipeBlock):
        def __init__(self, class_names=None, class_colors=None):
            super().__init__('draw_predictions')
            
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
                  
            self.start = time.time()
            self.count = 0
            self.fps = 0

        def forward(self, imgs, meta={}):
            preds = meta.get(Processors.StaticWords.yolo_results, [])
            imgs = [self.draw_predictions(
                i, self._extract_preds(p),f"FPS: {self.fps:.1f}, No.{ii}"
                ) for ii,(i, p) in enumerate(zip(imgs, preds))]            
            self.count += 1
            self.fps = self.count/(time.time()-self.start)
            return imgs, meta

        def test_forward(self, imgs: List[Any], meta: Dict = {}):
            if Processors.StaticWords.yolo_input_imgs not in meta:
                raise ValueError('yolo input raw imgs is needed.')
            if Processors.StaticWords.yolo_results not in meta:
                raise ValueError('yolo_results is needed.')
            if len(meta[Processors.StaticWords.yolo_input_imgs]) != len(meta[Processors.StaticWords.yolo_results]):
                raise ValueError('yolo input raw imgs and yolo_results must be of the same size.')
            return self.test_forward_numpy_rgb(imgs, meta)

        def draw_predictions(self, image, predictions, info_text=None):
            """Draw bounding boxes with class-specific colors."""
            image_with_boxes = image.copy()
            img_height, img_width = image.shape[:2]

            # Dynamically scale font size based on image size
            font_scale = max(0.5, min(img_width, img_height) / 600)
            thickness = max(1, int(font_scale * 2))

            if info_text:
                ### ---- Draw info at Bottom-Left ---- ###
                info_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                info_w, info_h = info_size

                # Position info at the bottom-left corner
                info_x = 10
                info_y = img_height - 10  # Place it slightly above the bottom edge

                # Background for info text
                cv2.rectangle(image_with_boxes, 
                            (info_x - 5, info_y - info_h - 5), 
                            (info_x + info_w + 5, info_y), 
                            (0, 0, 0), -1)  # Black background

                # Put info text
                cv2.putText(image_with_boxes, info_text, 
                            (info_x, info_y - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                            (255, 255, 255), thickness)  # White text
            
            if len(predictions) == 0:
                return image_with_boxes  # No predictions, return original image

            for pred in predictions:
                x1, y1, x2, y2, confidence, class_id = pred[:6]

                # Corrected scaling
                x1, y1, x2, y2 = map(int, [x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height])

                # Get color for the class (default white if class not found)
                color = self.class_colors.get(int(class_id), (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

                # Class label
                class_label = f"{self.class_names.get(int(class_id), 'null')}"
                score_label = f"{confidence:.2f}"

                # Get text sizes
                class_size, _ = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                score_size, _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                class_w, class_h = class_size
                score_w, score_h = score_size

                text_x = x1
                text_y = max(class_h + 5, y1)  # Prevent out-of-bounds issues

                # Background for class label
                cv2.rectangle(image_with_boxes, 
                            (text_x, text_y - class_h - 5), 
                            (text_x + class_w + 5, text_y), 
                            color, -1)

                # Put class name
                cv2.putText(image_with_boxes, class_label, 
                            (text_x + 2, text_y - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                            (0, 0, 0), thickness)

                # Background for confidence score (below class label)
                score_y = text_y + score_h + 5
                cv2.rectangle(image_with_boxes, 
                            (text_x, score_y - score_h - 5), 
                            (text_x + score_w + 5, score_y), 
                            color, -1)

                # Put confidence score
                cv2.putText(image_with_boxes, score_label, 
                            (text_x + 2, score_y - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                            (0, 0, 0), thickness)
                
            return image_with_boxes

        
        def _extract_preds(self, preds):
            """Extracts predictions from YOLO output."""
            if hasattr(preds, 'boxes'):
                bs = preds.boxes
                xyxyn = bs.xyxyn.cpu().numpy()
                conf = bs.conf.cpu().numpy()
                cls = bs.cls.cpu().numpy()
                preds = np.hstack([xyxyn, conf.reshape(-1, 1), cls.reshape(-1, 1)])
            return preds

        def _generate_class_colors(self, num_classes):
            """Generates distinct colors for each class."""
            return {i: tuple(random.randint(0, 255) for _ in range(3)) for i in range(num_classes)}









