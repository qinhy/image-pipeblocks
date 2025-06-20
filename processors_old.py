
import enum
import math
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops
import numpy as np
from typing import List, Any, Dict, Tuple
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, List, Optional, Any

# global setting
torch_img_dtype = torch.float16
numpy_img_dtype = np.uint8

class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)

class DummyDebayer5x5(torch.nn.Module):
    def __init__(self):
        super(DummyDebayer5x5, self).__init__()

    def forward(self, x):
        # Repeat the single channel 3 times to create an RGB image
        return x.repeat(1, 3, 1, 1)  # Shape: (N, 3, H, W)

class StaticWords(enum.Enum):
    yolo_results = 'yolo_results'
    yolo_input_imgs ='yolo_input_imgs'
    yolo_input_img_w_h ='yolo_input_img_w_h'

    sliding_window_input_img_w_h ='sliding_window_input_img_w_h'
    sliding_window_size = 'sliding_window_size'
    sliding_window_input_imgs = 'sliding_window_input_imgs'
    sliding_window_imgs_idx = 'sliding_window_imgs_idx'
    sliding_window_output_offsets = 'sliding_window_output_offsets'
    
class MergeYoloResults(PipeBlock):
    def __init__(self):
        super().__init__('merge_yolo_results')

    def forward(self, imgs, meta={}):
        results = meta.get(StaticWords.yolo_results,[])

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

        meta[StaticWords.yolo_results] = results
        return imgs, meta
    
    def test_forward(self, imgs, meta = {}):
        results = meta.get(StaticWords.yolo_results)
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
        dtype=torch_img_dtype
    ):
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
        
        self.conf_thres_per_class = {k: v for k, v in conf_thres_per_class}
        self.conf_thres_per_class_rlfb = [
            {k: v for k, v in conf_thres} for conf_thres in conf_thres_per_class_rlfb
        ]
        self.yolo_models = [(YOLO(model_path).to('cpu'),'cpu')]
    
    def forward(self, imgs, meta={}):
        res = self.predict(imgs)
        res = self.postprocess(res)

        meta[StaticWords.yolo_input_imgs] = imgs
        meta[StaticWords.yolo_results] = res
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

            half= self.dtype == torch_img_dtype
            
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

class SlidingWindow(PipeBlock):
    def __init__(self, window_size=(1280,1280), stride=None):
        """
        Splits an image into sliding windows.

        :param batched_img: NumPy array (H, W, C) or PyTorch tensor (C, H, W)
        :param window_size: Tuple (window_height, window_width)
        :param stride: Tuple (stride_height, stride_width), default is window_size (non-overlapping)
        """
        self.title = "sliding_window"
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
        meta[StaticWords.sliding_window_size] = self.window_size
        meta[StaticWords.sliding_window_imgs_idx] = imgs_idx
        meta[StaticWords.sliding_window_input_imgs] = imgs
        meta[StaticWords.sliding_window_output_offsets] = output_offsets
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
        
        meta[StaticWords.sliding_window_input_img_w_h] = (W,H)
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
        
        meta[StaticWords.sliding_window_input_img_w_h] = (W,H)
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

class SlidingWindowMerge(PipeBlock):
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
        
        raw_imgs_idx = meta[StaticWords.sliding_window_imgs_idx]
        multi_dets = meta[StaticWords.yolo_results]
        trans = meta[StaticWords.sliding_window_output_offsets]
        raw_imgs = meta[StaticWords.sliding_window_input_imgs]        
        window_size = meta[StaticWords.sliding_window_size]
        W, H = meta[StaticWords.sliding_window_input_img_w_h]
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

        meta[StaticWords.yolo_results] = yolo_results
        
        raw_imgs = meta[StaticWords.sliding_window_input_imgs]
        meta[StaticWords.yolo_input_imgs] = raw_imgs
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
  
class DrawPredictions(PipeBlock):
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
