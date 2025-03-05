import os
import time
import torch
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort
import enum

from ultralytics import YOLO

from GeneralTensorRTModel import GeneralTensorRTInferenceModel

def show_bayer(bayer_8bit):
    Image.fromarray(cv2.cvtColor(bayer_8bit.astype(np.uint8), cv2.COLOR_BAYER_BG2RGB)).show()

def show_rgb(rgb_8bit):    
    Image.fromarray(rgb_8bit).show()

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

class PipeBlock:
    def __init__(self, batched_img):
        self.title = 'null'
        self.is_tensor = isinstance(batched_img, torch.Tensor)
        self.is_numpy = isinstance(batched_img, np.ndarray)
        
        assert self.is_tensor or self.is_numpy, "batched_img must be either a PyTorch tensor or a NumPy array"
        
        if self.is_tensor:
            assert len(batched_img.shape) == 4, "Expected a 4D tensor of shape (B, C, H, W)"
            assert batched_img.shape[1] in [1, 3], "Expected channel dimension (C) to be 1 (grayscale) or 3 (RGB)"
        elif self.is_numpy:
            assert len(batched_img.shape) == 4, "Expected a 4D NumPy array of shape (B, H, W, C)"
            assert batched_img.shape[-1] in [1, 3], "Expected last dimension (C) to be 1 (grayscale) or 3 (RGB)"
    
    def __call__(self):
        raise NotImplementedError()

    def __check_input(self,batched_img):
        pass

    def __call__tensor_float(self, batched_img, meta={}):        
        """
        batched_img : BxCxHxW tensor, C = 1 or 3 (RGB only)
        """
        return batched_img, meta

    def __call__numpy_uint8(self, batched_img, meta={}):        
        """
        batched_img : BxHxWxC NumPy array, C = 1 or 3 (RGB only)
        """
        return batched_img, meta

class NumpyToTorchBlock(PipeBlock):
    def __init__(self, batched_img):
        """
        Convert a batched NumPy uint8 image to a normalized PyTorch float32 tensor.

        :param batched_img: NumPy array of shape (B, H, W, C), dtype=np.uint8.
        """
        super().__init__(batched_img)
        self.title = 'numpy_to_torch'

        if not self.is_numpy:
            raise ValueError("NumpyToTorchBlock only supports NumPy arrays!")

        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__numpy_uint8(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input NumPy array.
        """
        assert isinstance(batched_img, np.ndarray), "Input must be a NumPy array"
        assert batched_img.ndim == 4, "Expected 4D NumPy array of shape (B, H, W, C)"
        assert batched_img.shape[-1] in [1, 3], "Expected last dimension (C) to be 1 (grayscale) or 3 (RGB)"
        assert batched_img.dtype == np.uint8, "Expected np.uint8 input (0-255 range)"

    def __call__numpy_uint8(self, batched_img, meta={}):
        """
        Convert NumPy uint8 (B, H, W, C) to PyTorch float32 (B, C, H, W).

        :return: PyTorch tensor (B, C, H, W), dtype=torch.float32, values in range [0,1].
        """
        batched_img = torch.from_numpy(batched_img).permute(0, 3, 1, 2).float() / 255.0  # Convert to float32 and normalize
        return batched_img.cuda(), meta

class TorchToNumpyBlock(PipeBlock):
    def __init__(self, batched_img):
        """
        Convert a batched PyTorch float32 tensor to a NumPy uint8 image.

        :param batched_img: PyTorch tensor of shape (B, C, H, W), dtype=torch.float32, values in range [0,1].
        """
        super().__init__(batched_img)
        self.title = 'torch_to_numpy'

        if not self.is_tensor:
            raise ValueError("TorchToNumpyBlock only supports PyTorch tensors!")

        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__tensor_float(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input PyTorch tensor.
        """
        assert isinstance(batched_img, torch.Tensor), "Input must be a PyTorch tensor"
        assert batched_img.ndim == 4, "Expected 4D tensor of shape (B, C, H, W)"
        assert batched_img.shape[1] in [1, 3], "Expected channel dimension (C) to be 1 (grayscale) or 3 (RGB)"
        assert batched_img.dtype == torch.float32, "Expected torch.float32 input"

        if not (0.0 <= batched_img.min() and batched_img.max() <= 1.0):
            raise ValueError("Input tensor values must be in the range [0, 1]")

    def __call__tensor_float(self, batched_img, meta={}):
        """
        Convert PyTorch float32 (B, C, H, W) to NumPy uint8 (B, H, W, C).

        :return: NumPy array (B, H, W, C), dtype=np.uint8.
        """
        batched_img = (batched_img * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        return batched_img, meta

class CvDebayerBlock(PipeBlock):
    def __init__(self, batched_img, bayer_pattern=cv2.COLOR_BAYER_RG2RGB):
        """
        bayer_pattern: OpenCV Bayer conversion code (e.g., cv2.COLOR_BAYER_RG2RGB)
        """
        super().__init__(batched_img)
        self.title = 'cv_debayer'
        self.bayer_pattern = bayer_pattern  # Default to RGGB Bayer pattern
        if not self.is_numpy:
            raise ValueError('not support!')
        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__numpy_uint8(batched_img, meta)
    
    def __check_input(self,batched_img):
        assert isinstance(batched_img, np.ndarray), "Input must be a NumPy array"
        assert batched_img.shape[-1] == 1, "Expected single-channel input for Bayer images"
        assert batched_img.dtype == np.uint8, "Expected np.uint8 input for Bayer images"
        
    def __call__numpy_uint8(self, batched_img, meta={}):
        """
        Convert a Bayer raw image (B, H, W, 1) NumPy array to an RGB image (B, H, W, 3).
        """
        # Squeeze last channel to get (B, H, W)
        batched_img = batched_img.squeeze(-1)

        debayered_imgs = []
        for img in batched_img:
            img = cv2.cvtColor(img, self.bayer_pattern)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            debayered_imgs.append(img)

        # Stack to get (B, H, W, 3)
        debayered_imgs_np = np.stack(debayered_imgs)

        return debayered_imgs_np, meta

class TorchDebayerBlock(PipeBlock):
    def __init__(self, batched_img, debayer_model=Debayer5x5().cuda()):
        """
        Converts a batched Bayer image (B, 1, H, W) into an RGB image (B, 3, H, W) using Torch's Debayer5x5.

        :param batched_img: PyTorch tensor of shape (B, 1, H, W) with dtype uint8.
        """
        super().__init__(batched_img)
        self.title = 'torch_debayer'
        self.debayer_model = debayer_model

        if not self.is_tensor:
            raise ValueError("TorchDebayerBlock only supports PyTorch tensors!")

        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__tensor_float(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input PyTorch tensor.
        """
        if not isinstance(batched_img, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        if batched_img.dtype != torch.float32:
            raise TypeError(f"Expected torch.float32, but got {batched_img.dtype}")

        if batched_img.shape[1] != 1:
            raise ValueError(f"Expected single-channel input (B, 1, H, W), but got shape {batched_img.shape}")

        if not (0.0 <= batched_img.min() and batched_img.max() <= 1.0):
            raise ValueError("Input tensor values must be in the range [0, 1]")


    def __call__tensor_float(self, batched_img, meta={}):
        """
        Convert a Bayer raw image (B, 1, H, W) tensor to an RGB image (B, 3, H, W) using PyTorch's Debayer5x5.
        """
        rgb_tensor = self.debayer_model(batched_img)

        return rgb_tensor, meta

class TensorRTDebayerBlock(TorchDebayerBlock):
    def __init__(self, batched_img, debayer_model_path="debayer5x5.trt"):
        """
        Applies Bayer to RGB conversion using a TensorRT inference model.

        :param batched_img: PyTorch tensor of shape (B, 1, H, W) with dtype torch.float32 in range [0,1].
        :param debayer_model_path: Path to the TensorRT model file.
        """
        super().__init__(batched_img)
        self.title = 'tensorrt_debayer'
        self.debayer_model = GeneralTensorRTInferenceModel(debayer_model_path)
        self.input_shape = self.debayer_model.input_shape  # Ensure correct shap        
        
        if not self.is_tensor:
            raise ValueError("TensorRTDebayerBlock only supports PyTorch tensors!")
        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__tensor_float(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input PyTorch tensor.
        """
        if not isinstance(batched_img, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        if batched_img.dtype != torch.float32:
            raise TypeError(f"Expected torch.float32, but got {batched_img.dtype}")

        if batched_img.shape[1] != 1:
            raise ValueError(f"Expected single-channel input (B, 1, H, W), but got shape {batched_img.shape}")

        if not (0.0 <= batched_img.min() and batched_img.max() <= 1.0):
            raise ValueError("Input tensor values must be in the range [0, 1]")

        # Ensure input shape matches model expectation
        if batched_img.shape[1:] != self.input_shape[1:]:
            raise ValueError(f"Expected input shape {self.input_shape}, but got {batched_img.shape}")

    def __call__tensor_float(self, batched_img, meta={}):
        """
        Convert a Bayer raw image (B, 1, H, W) tensor to an RGB image (B, 3, H, W) using TensorRT inference.[0, 1.0]
        """
        # Perform inference using TensorRT model
        rgb_tensor = self.debayer_model(batched_img)
        return rgb_tensor, meta

class TorchResizeBlock(PipeBlock):
    def __init__(self, batched_img, target_size, interpolation="bilinear"):
        """
        Resizes a batch of images using PyTorch's torchvision.

        :param batched_img: PyTorch tensor of shape (B, C, H, W) with dtype torch.float32 in range [0,1].
        :param target_size: Tuple (new_H, new_W) specifying the new image size.
        :param interpolation: Interpolation mode ('nearest', 'bilinear', 'bicubic', etc.).
        """
        super().__init__(batched_img)
        self.title = 'torch_resize'
        self.target_size = target_size
        self.interpolation = interpolation

        if not self.is_tensor:
            raise ValueError("TorchResizeBlock only supports PyTorch tensors!")

        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__tensor_float(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input PyTorch tensor.
        """
        if not isinstance(batched_img, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        if batched_img.dtype != torch.float32:
            raise TypeError(f"Expected torch.float32, but got {batched_img.dtype}")

        if len(batched_img.shape) != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), but got {batched_img.shape}")

        if not (0.0 <= batched_img.min() and batched_img.max() <= 1.0):
            raise ValueError("Input tensor values must be in the range [0, 1]")

    def __call__tensor_float(self, batched_img, meta={}):
        """
        Resize a batch of images using torchvision.

        :param batched_img: PyTorch tensor of shape (B, C, H, W)
        :return: Resized images (B, C, new_H, new_W)
        """
        resized_imgs = F.interpolate(batched_img, size=self.target_size, mode=self.interpolation)
        return resized_imgs, meta
    
class CVResizeBlock(PipeBlock):
    def __init__(self, batched_img, target_size, interpolation=cv2.INTER_LINEAR):
        """
        Resize a batched image using OpenCV.

        :param batched_img: NumPy array of shape (B, H, W, C), dtype=np.uint8.
        :param target_size: Tuple (new_width, new_height) specifying the target size.
        :param interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR).
        """
        super().__init__(batched_img)
        self.title = 'cv_resize'
        self.target_size = target_size
        self.interpolation = interpolation

        if not self.is_numpy:
            raise ValueError("CVResizeBlock only supports NumPy arrays!")

        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__numpy_uint8(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input NumPy array.
        """
        assert isinstance(batched_img, np.ndarray), "Input must be a NumPy array"
        assert batched_img.ndim == 4, "Expected 4D NumPy array of shape (B, H, W, C)"
        assert batched_img.shape[-1] in [1, 3], "Expected last dimension (C) to be 1 (grayscale) or 3 (RGB)"
        assert batched_img.dtype == np.uint8, "Expected np.uint8 input (0-255 range)"

    def __call__numpy_uint8(self, batched_img, meta={}):
        """
        Resize a batch of images using OpenCV.

        :param batched_img: NumPy array (B, H, W, C) with dtype=np.uint8.
        :return: Resized NumPy array of shape (B, new_H, new_W, C).
        """
        resized_imgs = [
            cv2.resize(img, self.target_size, interpolation=self.interpolation)
            for img in batched_img
        ]

        # Stack resized images to maintain batch dimension (B, new_H, new_W, C)
        resized_imgs_np = np.stack(resized_imgs)

        return resized_imgs_np, meta

class YOLOBlock(PipeBlock):
    def __init__(self, batched_img,  modelname: str = 'yolov5s6u.pt', conf: float = 0.6, cpu: bool = False, names={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}):
        super().__init__(batched_img)
        self.title = 'yolov5_detections'
        self.conf = conf
        self.model = YOLO(modelname,task='detect')
                
        if not hasattr(self.model,'names'):
            self.names = names
        else:
            self.names = self.model.names

        self.__check_input(batched_img)

    def __call__(self,batched_img, meta={}):
        return self.__call__tensor_float(batched_img, meta)

    def __check_input(self, batched_img):
        """
        Validate the input PyTorch tensor.
        """
        if not self.is_tensor:
            raise ValueError("YOLOv5Block only supports PyTorch tensors!")

        if batched_img.dtype != torch.float32:
            raise TypeError(f"Expected torch.float32, but got {batched_img.dtype}")

        if batched_img.shape[1] != 3:
            raise ValueError(f"Expected RGB input (B, 3, H, W), but got shape {batched_img.shape}")

        if not (0.0 <= batched_img.min() and batched_img.max() <= 1.0):
            raise ValueError("Input tensor values must be in the range [0, 1]")
        
    def __call__tensor_float(self, batched_img, meta={}):
        """
        Perform YOLOv5 inference on a batch of images.

        :param batched_img: PyTorch tensor (B, 3, H, W) with dtype torch.float32 in range [0,1].
        :return: Dictionary with detected bounding boxes, class labels, and confidence scores.
        """
        meta['class_names'] = self.names
        with torch.no_grad():
            meta[self.title] = self.model(batched_img, conf=self.conf,verbose=False)
        return batched_img, meta

class ByteTrackBlock:
    def __init__(self, batch_size, modelname='yolov5s6u.pt', conf_thres=0.6,
                 track_thresh=0.6, match_thresh=0.8, frame_rate=30, track_buffer=30):
        """
        Initialize the ByteTrackBlock for multi-object tracking using YOLO for detection.

        :param modelname: Name of the YOLO model to use for detection.
        :param conf_thres: Confidence threshold for YOLO detection.
        :param track_thresh: Tracking confidence threshold for ByteTrack.
        :param match_thresh: Matching threshold for associating detections across frames.
        :param frame_rate: Frame rate of the video input for tracking.
        """
        self.title = 'bytetrack_trackings'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class ByteTrackData:
            def __init__(self,track_thresh,match_thresh,frame_rate,track_buffer):
                self.track_thresh=track_thresh
                self.match_thresh=match_thresh
                self.frame_rate=frame_rate
                self.track_buffer=track_buffer
                self.mot20 = False

        args=ByteTrackData(track_thresh,match_thresh,frame_rate,track_buffer)
        # Initialize ByteTrack tracker
        self.trackers = [BYTETracker(args) for _ in range(batch_size)]

    def __call__(self, batched_img, meta={}):
        """
        Perform tracking on a batch of images.

        :param batched_img: Tensor (B, 3, H, W) with dtype torch.float32 in range [0,1].
        :return: Tuple with the original image and tracking results.
        """
        if not isinstance(batched_img, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        _,detections = [(i,v) for i,v in meta.items() if 'detections' in i][0]
        if len(batched_img) != len(detections):
            raise ValueError("batched_img and detections must be same len.")
        
        tracking_outputs = []
        for i, (img, detection) in enumerate(zip(batched_img,detections)):
            if detection is None or len(detection)==0:continue

            tracker = self.trackers[i]
            if hasattr(detection, 'boxes'):
                # Extract boxes, confidence scores, and class labels
                bs = detection.boxes
                xyxy = bs.xyxy.cpu().numpy()
                conf = bs.conf.cpu().numpy()
                cls = bs.cls.cpu().numpy()
                dets = np.hstack([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)])
            
            # Track objects
            _, H,W = img.shape
            tracked_objects = tracker.update(torch.Tensor(dets), img_info=(H,W), img_size=(H,W))

            # Collect results
            frame_results = []
            for obj in tracked_objects:
                track_id = obj.track_id
                # x, y, w, h = obj.tlwh
                # cls_id = int(obj.class_id)
                # conf = obj.score

                frame_results.append(track_id)

            tracking_outputs.append(frame_results)

        meta[self.title] = tracking_outputs
        return batched_img, meta

class SlidingWindowBlock:
    def __init__(self, batched_img, window_size, stride=None):
        """
        Splits an image into sliding windows.

        :param batched_img: NumPy array (H, W, C) or PyTorch tensor (C, H, W)
        :param window_size: Tuple (window_height, window_width)
        :param stride: Tuple (stride_height, stride_width), default is window_size (non-overlapping)
        """
        self.title = "sliding_window"
        self.window_size = window_size
        self.stride = stride if stride else window_size
        self.__check_input(batched_img)

    def __check_input(self, batched_img):
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

    def __call__(self, batched_img, meta={}):
        """
        Splits an image into sliding windows.

        Returns:
            windows: NumPy array or PyTorch tensor with shape:
                     - NumPy: (num_windows, window_height, window_width, C)
                     - PyTorch: (num_windows, C, window_height, window_width)
            offsets_xyxy: List of bounding boxes (x1, y1, x2, y2) for each window.
        """
        meta['sliding_window_raw'] = batched_img
        if isinstance(batched_img, np.ndarray):
            imgs,offs = self._split_numpy(batched_img)
        elif isinstance(batched_img, torch.Tensor):
            imgs,offs = self._split_torch(batched_img)
        else:
            raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
        meta['sliding_window_offsets'] = offs
        return imgs,meta

    def _split_numpy(self, data):
        """
        Split a NumPy array into sliding windows.
        """
        if data.ndim == 4:  # If batched, process each image
            batch_windows = []
            batch_offsets = []
            for i in range(data.shape[0]):
                windows, offsets = self._split_numpy_single(data[i])
                batch_windows.append(windows)
                batch_offsets.append(offsets)
            batch_windows, batch_offsets = np.concatenate(batch_windows, axis=0), batch_offsets
            return batch_windows, batch_offsets
        else:
            windows, offsets = self._split_numpy_single(data[i])
        return windows, offsets

    def _split_numpy_single(self, data):
        """
        Split a single NumPy image (H, W, C) into sliding windows.
        """
        H, W, C = data.shape
        wH, wW = self.window_size
        sH, sW = self.stride

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

        return np.stack(windows_list, axis=0), offsets_xyxy

    def _split_torch(self, data):
        """
        Split a PyTorch tensor into sliding windows.
        """
        if data.dim() == 4:  # If batched, process each image
            batch_windows = []
            batch_offsets = []
            for i in range(data.shape[0]):
                windows, offsets = self._split_torch_single(data[i])
                batch_windows.append(windows)
                batch_offsets.append(offsets)
            return torch.cat(batch_windows, dim=0), batch_offsets
        else:
            return self._split_torch_single(data)

    def _split_torch_single(self, data):
        """
        Split a single PyTorch image (C, H, W) into sliding windows.
        """
        C, H, W = data.shape
        wH, wW = self.window_size
        sH, sW = self.stride

        if wH > H or wW > W:
            raise ValueError(f"Window size ({wH}, {wW}) must be <= image size ({H}, {W}).")

        windows_list = []
        offsets_xyxy = []

        for row_start in range(0, H - wH + 1, sH):
            for col_start in range(0, W - wW + 1, sW):
                window = data[:, row_start: row_start + wH, col_start: col_start + wW]
                windows_list.append(window)
                # offsets_xyxy.append((col_start, row_start, col_start + wW, row_start + wH))
                offsets_xyxy.append((col_start, row_start, col_start, row_start))

        return torch.stack(windows_list, dim=0), offsets_xyxy 

class SlidingWindowMergeBlock:
    def __init__(self):
        """
        Merges YOLO detections from sliding windows back into the original image coordinates.
        """
        self.title = "sliding_window_merge"

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

    def __call__(self, imgs, meta={}):
        """
        Merges sliding window detections back into the original image space.

        :param imgs: List of split image windows.
        :param meta: Metadata dictionary containing detection results and window offsets.
        :return: Tuple (original_images, updated_meta)
        """
        # Retrieve detections and transformations from metadata
        key, multi_dets = [(i, v) for i, v in meta.items() if "detections" in i][0]
        meta[key] = []
        trans = meta["sliding_window_offsets"]        
        splits = len(trans[0])
        multi_dets = [multi_dets[i:i+splits] for i in range(0,len(multi_dets),splits)]
        for ii, (img, preds) in enumerate(zip(imgs, multi_dets)):
            if len(preds) == 0: continue
            trans_xyxy = trans[ii]
            preds = [self._extract_preds(p) for p in preds]
            # Adjust detection bounding boxes based on window offsets
            for i, p in enumerate(preds):
                if len(p) == 0: continue
                if isinstance(p, torch.Tensor):
                    p = p.cpu().numpy()
                p[:,:4] += trans_xyxy[i]  # Shift bounding boxes back to original image space
                preds[i] = p

            # Stack detections into a single array
            preds = np.vstack(preds)

            # ---------- Apply Non-Maximum Suppression (NMS) ----------
            boxes = torch.tensor(preds[:, :4])  # (x1, y1, x2, y2)
            scores = torch.tensor(preds[:, 4])  # Confidence scores
            class_ids = torch.tensor(preds[:, 5])  # Class labels

            # Perform NMS using PyTorch
            keep_indices = torch.ops.torchvision.nms(boxes, scores, 0.15)
            preds = preds[keep_indices.numpy()]  # Keep only high-confidence, non-overlapping detections

            meta[key].append(preds)

        return meta["sliding_window_raw"], meta

class DrawPredictionsBlock:
    def __init__(self, class_names=None):
        self.title = 'draw_predictions'
        self.class_names = class_names
        self.color = (0, 255, 0)

    def draw_predictions(self, image, predictions, class_names=None):
        if isinstance(image,torch.Tensor):
            image_with_boxes = image.cpu().numpy()
        else:
            image_with_boxes = image.copy()
        for pred in predictions:
            if len(predictions) == 0:continue
            x1, y1, x2, y2, confidence, class_id = pred[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), self.color, 2)
            label = f"{class_names[int(class_id)] if class_names else 'Class'}: {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x, text_y = x1, y1 - 5
            cv2.rectangle(image_with_boxes, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), self.color, -1)
            cv2.putText(image_with_boxes, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image_with_boxes

    def _extract_preds(self,preds):        
        if hasattr(preds,'pred'):
            preds = preds.pred
            preds = np.vstack([d.cpu().numpy() for d in preds])
        elif hasattr(preds,'boxes'):
            bs = preds.boxes
            xyxy = bs.xyxy.cpu().numpy()
            conf = bs.conf.cpu().numpy()
            cls = bs.cls.cpu().numpy()
            preds = np.hstack([xyxy,conf.reshape(1,-1).T,cls.reshape(1,-1).T])
        return preds    

    def __call__(self, imgs, meta={}):
        # print(imgs.shape, meta.keys())
        self.class_names = meta.get('class_names',self.class_names)
        _,multi_predictions = [(i,v) for i,v in meta.items() if 'detections' in i][0]
        drawn_imgs = []
        if (len(imgs)==1 and len(imgs[0].shape)==4) and len(multi_predictions)==len(imgs[0]):
            imgs = [*imgs[0]]

        for img, preds in zip(imgs, multi_predictions):
            # Draw predictions on the image
            if (type(img) is list or len(img.shape)==4) and type(preds) is list:
                preds = [self._extract_preds(p) for p in preds]
                img = [*img]
                drawn_img = [self.draw_predictions(i, p, self.class_names)
                                    for i,p in zip(img, preds)]
                drawn_imgs.append(drawn_img)
            else:
                preds = self._extract_preds(preds)
                drawn_img = self.draw_predictions(img, preds, self.class_names)
                drawn_imgs.append(drawn_img)

        return drawn_imgs, meta

def video_gen(input_path):    
    cap = cv2.VideoCapture(input_path)    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def video2video(input_path, output_path, process=lambda im:im, verbose=False):
    # Initialize video capture and writer
    if verbose:
        print(f"Opening video file {input_path}...")
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:raise ValueError('Can not open video file!')

    frame = process(frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

    if verbose:
        print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
        print("Starting video processing...")

    try:
        frame_count = 0
        start_time = time.time()
        while cap.isOpened():
            # Calculate and display progress
            if verbose and frame_count % 10 == 0 and frame_count>0:
                elapsed_time = time.time() - start_time
                frames_remaining = total_frames - frame_count
                estimated_time_remaining = (elapsed_time / frame_count) * frames_remaining
                print(f"Processed {frame_count}/{total_frames} frames "
                    f"({(frame_count / total_frames) * 100:.2f}%), "
                    f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # if frame_count % 10 !=0 :continue

            frame = process(frame)

            # Write the annotated frame to output video
            out.write(frame)
    except Exception as e:
        print(e)

    finally:
        # Release resources
        cap.release()
        out.release()
        if verbose:
            print(f"Processed video saved to {output_path}")

def memo_gen(fs=[],num=4):
    ns = [np.load(f) for f in fs]
    ns = np.vstack(ns)

    while True:
        idx = np.random.randint(0,len(ns),num)
        yield ns[idx]
        # time.sleep(delay)

def memo2memo(gen,processes=[],title='memo2memo',timeout=60):
    # Initialize FPS tracking
    frame_count = 0
    start_time = time.time()

    title = '::'.join([title]+[(b.title if hasattr(b,'title') else 'null') for b in processes])

    start_time = time.time()
    print()
    for ims in gen:
        # Increment frame count
        frame_count += 1
        
        # Apply the processing function to each item in the current tuple of inputs
        imgs,meta = processes[0](ims)
        for p in processes[1:]:
            imgs,meta = p(imgs,meta)
        
        # Calculate elapsed time and FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print(f'[{title}]'+' fps : {:.2f}'.format(fps), end='\r')

        if time.time()-start_time>timeout:break

    print(f'[{title}]'+' fps : {:.2f}'.format(fps))
    print()


def memo2memo_onetime(gen,processes=[],title='memo2memo',timeout=60):
    for ims in gen:        
        # Apply the processing function to each item in the current tuple of inputs
        imgs,meta = processes[0](ims)
        for p in processes[1:]:
            imgs,meta = p(imgs,meta)

        return imgs,meta

def show_rgb(rgb_8bit):    
    from PIL import Image
    Image.fromarray(rgb_8bit).show()

def show_float(rgb):
    from PIL import Image
    Image.fromarray((rgb*255).astype(np.uint8)).show()

def test1(timeout=60):
    def gen(fs=[f'./raw/{f}' for f in os.listdir('./raw')[:3]]):
        ns = [np.load(f) for f in fs]
        ns = np.vstack(ns)
        while True:
            idx = np.random.randint(0,len(ns))
            res = ns[idx]
            h,w = res.shape
            yield ns[idx].reshape(1,h,w,1)
            
    mgen1 = gen()
    test_name = 'memo_gen_1'
    img = next(mgen1)
    cd,cr = CvDebayerBlock(img),CVResizeBlock(img,(1280,1280))
    it = NumpyToTorchBlock(img)
    img_t = it(img)[0]
    ti = TorchToNumpyBlock(img_t)
    td,sw = TorchDebayerBlock(img_t),SlidingWindowBlock(cd(img)[0],(1280,1280))
    sm,dp = SlidingWindowMergeBlock(),DrawPredictionsBlock()
    img_tc = td(img_t)[0]
    y = YOLOBlock(img_tc,'yolov5s6u.pt')
    bt = ByteTrackBlock(1)
    rd,tr = TensorRTDebayerBlock(img_t,'debayer5x5.(1, 1, 2840, 2832).FP16.trt'),TorchResizeBlock(img_tc,(1280,1280))
                     
    memo2memo(mgen1,[lambda i:(i,{})],test_name,timeout)

    memo2memo(mgen1,[cd],test_name,timeout)
    memo2memo(mgen1,[it,td],test_name,timeout)
    memo2memo(mgen1,[it,rd],test_name,timeout)

    memo2memo(mgen1,[cd,cr,it,y],test_name,timeout)
    memo2memo(mgen1,[cd,cr,it,y,ti,dp],test_name,timeout)
    memo2memo(mgen1,[cd,cr,it,y,bt,ti,dp],test_name,timeout)

    memo2memo(mgen1,[it,td,tr,y],test_name,timeout)
    memo2memo(mgen1,[it,td,tr,y,ti,dp],test_name,timeout)

    memo2memo(mgen1,[it,td,sw,y,sm],test_name,timeout)
    memo2memo(mgen1,[it,td,sw,y,sm,ti,dp],test_name,timeout)
test1(60)

def test4(timeout=60):
    def gen(fs=[f'./raw/{f}' for f in os.listdir('./raw')[:3]]):
        ns = [np.load(f) for f in fs]
        ns = np.vstack(ns)
        while True:
            idx = np.random.randint(0,len(ns),4)
            res = ns[idx]
            h,w = res[0].shape
            yield ns[idx].reshape(4,h,w,1)
            
    mgen4 = gen()
    test_name = 'memo_gen_4'
    img = next(mgen4)
    cd,cr = CvDebayerBlock(img),CVResizeBlock(img,(1280,1280))
    it = NumpyToTorchBlock(img)
    img_t = it(img)[0]
    ti = TorchToNumpyBlock(img_t)
    td,sw,sm,dp = TorchDebayerBlock(img_t),SlidingWindowBlock(cd(img)[0],(1280,1280)),SlidingWindowMergeBlock(),DrawPredictionsBlock()
    img_tc = td(img_t)[0]
    y = YOLOBlock(img_tc,'yolov5s6u.(36, 3, 1280, 1280).FP16.engine')
    rd,tr = TensorRTDebayerBlock(img_t,'debayer5x5.(4, 1, 2840, 2832).FP16.trt'),TorchResizeBlock(img_tc,(1280,1280))
                     
    memo2memo(mgen4,[lambda i:(i,{})],test_name,timeout)

    memo2memo(mgen4,[cd],test_name,timeout)
    memo2memo(mgen4,[it,td],test_name,timeout)
    memo2memo(mgen4,[it,rd],test_name,timeout)

    memo2memo(mgen4,[cd,cr,it,y],test_name,timeout)
    memo2memo(mgen4,[cd,cr,it,y,ti,dp],test_name,timeout)

    memo2memo(mgen4,[it,td,tr,y],test_name,timeout)
    memo2memo(mgen4,[it,td,tr,y,ti,dp],test_name,timeout)

    memo2memo(mgen4,[it,td,sw,y,sm],test_name,timeout)
    memo2memo(mgen4,[it,td,sw,y,sm,ti,dp],test_name,timeout)









