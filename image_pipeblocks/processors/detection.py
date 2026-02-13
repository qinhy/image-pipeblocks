from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import cv2
import numpy as np
import torch
from pydantic import Field


from ultralytics import YOLO as ultralyticsYOLO
from ultralytics.utils import ops

from ..ImageMat import ColorType, ImageMat, ImageMatInfo, ImageMatProcessor
from .utils import logger

if TYPE_CHECKING:
    from .basic import GPS


GeneralTensorRTInferenceModel = None
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    class GeneralTensorRTInferenceModel:
        class CudaDeviceContext:
            def __init__(self, device_index):
                self.device = cuda.Device(device_index)
                self.context = None

            def __enter__(self):
                self.context = self.device.make_context()
                return self.context

            def __exit__(self, exc_type, exc_value, traceback):
                self.context.pop()

        np2torch_dtype = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
        }

        class HostDeviceMem:
            def __init__(self, host_mem, device_mem, name, shape, dtype, size):
                self.host = host_mem
                self.device = device_mem
                self.name = name
                self.shape = shape
                self.dtype = dtype
                self.size = size

            def __repr__(self):
                return (
                    f"{self.__class__.__name__}(host=0x{id(self.host):x}"
                    f",device=0x{id(self.device):x})"
                    f",name={self.name}"
                    f",shape={self.shape}"
                    f",dtype={self.dtype}"
                    f",size={self.size})"
                )

        def __init__(self, engine_path, device, input_name='input', output_name='output'):
            self.engine_path = engine_path
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = None
            self.context = None
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.stream = cuda.Stream()
            self.input_shape = None
            self.output_shape = None
            self.dtype = None
            self.device = torch.device(device)
            if self.device.index > 0:
                with GeneralTensorRTInferenceModel.CudaDeviceContext(self.device.index):
                    self.load_trt(engine_path, input_name, output_name)
                    self(torch.rand(*self.input_shape, device=self.device, dtype=self.np2torch_dtype[self.dtype]))
                    self.load_trt(engine_path, input_name, output_name)
            else:
                self.load_trt(engine_path, input_name, output_name)
                self(torch.rand(*self.input_shape, device=self.device, dtype=self.np2torch_dtype[self.dtype]))
                self.load_trt(engine_path, input_name, output_name)

        def load_trt(self, engine_path, input_name='input', output_name='output', verb=True):
            """Load a TensorRT engine file and prepare context and buffers."""
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self.input_shape = [*self.engine.get_tensor_shape(input_name)]
            self.output_shape = [*self.engine.get_tensor_shape(output_name)]
            self.dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))

            self.inputs, self.outputs, self.bindings = self._allocate_buffers()
            if verb:
                print(f"[TensorRT] Loaded engine: {engine_path}, dtype: {self.dtype}")
                print(f"  Input shape: {self.input_shape}")
                print(f"  Output shape: {self.output_shape}")

        def _allocate_buffers(self):
            inputs, outputs, bindings = [], [], []
            num_io = self.engine.num_io_tensors
            for i in range(num_io):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                hdm = self.HostDeviceMem(host_mem, device_mem, name, shape, dtype, size)

                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    inputs.append(hdm)
                else:
                    outputs.append(hdm)
            return inputs, outputs, bindings

        def _transfer_torch2cuda(self, tensor: torch.Tensor, device_mem: HostDeviceMem):
            num_bytes = tensor.element_size() * tensor.nelement()
            cuda.memcpy_dtod_async(
                dest=int(device_mem.device),
                src=int(tensor.data_ptr()),
                size=num_bytes,
                stream=self.stream,
            )

        def _transfer_cuda2torch(self, device_mem: HostDeviceMem):
            torch_dtype = self.np2torch_dtype[self.dtype]
            out_tensor = torch.empty(self.output_shape, device=self.device, dtype=torch_dtype)

            num_bytes = out_tensor.element_size() * out_tensor.nelement()
            cuda.memcpy_dtod_async(
                dest=int(out_tensor.data_ptr()),
                src=int(device_mem.device),
                size=num_bytes,
                stream=self.stream,
            )
            return out_tensor

        def infer(self, inputs: List[torch.Tensor]):
            x = inputs[0]
            assert torch.is_tensor(x), "Input must be a torch.Tensor!"
            assert x.is_cuda, "Torch input must be on CUDA!"
            assert x.dtype == self.np2torch_dtype[self.dtype], (
                f"Expected dtype {self.np2torch_dtype[self.dtype]}, got {x.dtype}"
            )
            assert x.device == self.device, "Torch input must be on same CUDA!"
            return self.raw_infer(x)

        def raw_infer(self, x: torch.Tensor):
            [self._transfer_torch2cuda(x, mem) for x, mem in zip([x], self.inputs)]
            self.context.execute_v2(bindings=self.bindings)
            self.stream.synchronize()
            return []

        def __call__(self, input_data):
            outputs = self.infer([input_data])
            return outputs[0] if len(outputs) == 1 else outputs
except Exception:
    pass


class YOLO(ImageMatProcessor):
    title: str = 'YOLO_detections'
    gpu: bool = True
    multi_gpu: int = -1
    _torch_dtype: Any = ImageMatInfo.torch_img_dtype()

    modelname: str = 'yolov5s6u.pt'
    imgsz: int = -1
    conf: Union[float, Dict[int, float]] = 0.6
    min_conf: float = 0.6
    max_det: int = 300
    class_names: Optional[Dict[int, str]] = None
    save_results_to_meta: bool = True

    plot_imgs: bool = True
    use_official_predict: bool = True

    yolo_verbose: bool = False

    nms_iou: float = 0.7
    _models: dict = {}
    devices: List[str] = []

    def change_model(self, modelname: str):
        self.modelname = modelname
        self.model_post_init(None)
        self.build_models()

    def model_post_init(self, context):
        self.num_devices = self.devices_info(gpu=self.gpu, multi_gpu=self.multi_gpu)
        default_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush',
        }
        self.class_names = self.class_names if self.class_names is not None else default_names
        if isinstance(self.conf, dict):
            self.min_conf = min(list(self.conf.values()))
        else:
            self.min_conf = self.conf
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        if img.is_ndarray():
            img.require_ndarray()
            img.require_HWC()
            img.require_RGB()
        elif img.is_torch_tensor():
            img.require_torch_tensor()
            img.require_BCHW()
            img.require_RGB()
        else:
            raise TypeError("Unsupported image type for YOLO")

        if self.imgsz < 0:
            self.imgsz = img.info.W

        self.devices.append(img.info.device)

    def forward_raw(
        self,
        imgs_data: List[Union[np.ndarray, torch.Tensor]],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List["Any"]:
        if len(self._models) == 0:
            self.build_models()
        if self.use_official_predict:
            imgs, yolo_results_xyxycc = self.official_predict(imgs_data, imgs_info)
        else:
            imgs, yolo_results_xyxycc = self.predict(imgs_data, imgs_info)

        self.bounding_box_xyxy = []
        for xyxycc in yolo_results_xyxycc:
            confs = xyxycc[:, 4]
            ids = xyxycc[:, 5]
            thresholds = [self.conf[int(i)] if isinstance(self.conf, dict) else self.conf for i in ids]
            self.bounding_box_xyxy.append(xyxycc[confs > thresholds])

        res = imgs if len(imgs) > 0 else imgs_data
        return res

    def build_out_mats(self, validated_imgs, converted_raw_imgs, color_type=ColorType.RGB):
        return super().build_out_mats(validated_imgs, converted_raw_imgs, color_type)

    def build_models(self):
        for d in self.num_devices:
            if d not in self._models:
                model = ultralyticsYOLO(model=self.modelname, task='detect').to(d)
                if hasattr(model, 'names'):
                    self.class_names = model.names

                if not self.use_official_predict:
                    model = model.type(self._torch_dtype)

                    def model_predict(img, model=model):
                        res = model.model(img)
                        return res

                    self._models[d] = model_predict
                else:

                    def model_predict(
                        img,
                        model=model,
                        device=d,
                        conf=self.min_conf,
                        verbose=self.yolo_verbose,
                        imgsz=self.imgsz,
                        half=(self._torch_dtype == torch.float16),
                    ):
                        res = model.predict(source=img, conf=conf, verbose=verbose, half=half)
                        return res

                    self._models[d] = model_predict

    def official_predict(
        self,
        imgs_data: List[Union[np.ndarray, torch.Tensor]],
        imgs_info: List[ImageMatInfo] = [],
    ):
        imgs = []
        yolo_results = []
        for i, img in enumerate(imgs_data):
            device = self.num_devices[i % self.num_gpus]
            yolo_result = self._models[device](img)
            if isinstance(yolo_result, list):
                yolo_results += yolo_result

        yolo_results_xyxycc = [None] * len(yolo_results)
        for i, yolo_result in enumerate(yolo_results):
            if self.plot_imgs:
                imgs.append(yolo_result.plot())

            if hasattr(yolo_result, 'boxes'):
                boxes = yolo_result.boxes
                xyxycc = (
                    torch.cat(
                        [
                            boxes.xyxy,
                            boxes.conf.view(-1, 1),
                            boxes.cls.view(-1, 1),
                        ],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
            else:
                xyxycc = np.zeros((0, 6), dtype=np.float32)

            yolo_results_xyxycc[i] = xyxycc
        return imgs, yolo_results_xyxycc

    def predict(
        self,
        imgs_data: List[Union[np.ndarray, torch.Tensor]],
        imgs_info: List[ImageMatInfo] = [],
    ):
        yolo_results = []
        for i, img in enumerate(imgs_data):
            device = self.num_devices[i % self.num_gpus]
            yolo_model: YOLO = self._models[device]
            preds, feature_maps = yolo_model(img)
            preds = ops.non_max_suppression(
                preds,
                self.min_conf,
                self.nms_iou,
                classes=None,
                agnostic=False,
                max_det=self.max_det,
                nc=len(self.class_names),
                end2end=False,
                rotated=False,
            )
            if isinstance(preds, list):
                yolo_results += preds
            else:
                yolo_results.append(preds)
        yolo_results_xyxycc: List[np.ndarray] = [r.cpu().numpy() for r in yolo_results]
        return [], yolo_results_xyxycc


class SimpleTracking(ImageMatProcessor):
    title: str = 'simple_tracking'
    detect_frames_thre: int = 10
    ignore_frames_thre: int = 3
    queue_len: int = 20

    detector_uuid: str
    frame_cnt: int = 0
    frame_recs: List = Field(default_factory=list, exclude=True)
    frame_save_queue: queue.Queue[Tuple[str, List[ImageMat]]] = Field(
        default_factory=queue.Queue, exclude=True
    )
    frame_save_queue_len: int = 1000
    class_names: Dict[str, str] = {}
    all_cls_id: Set[int] = set()

    save_jpeg: bool = False
    save_mp4: bool = False
    save_jpeg_gps: bool = False
    save_dir: str = ''
    gps_uuid: str = ''

    timezone: int = 9
    callback: Optional[ImageMatProcessor] = None

    _det_processor: 'YOLO' = None
    _gps_processor: Optional['GPS'] = None

    def model_post_init(self, context):
        self.queue_len = self.detect_frames_thre + self.ignore_frames_thre + 1
        if self.save_jpeg or self.save_mp4 or self.save_jpeg_gps:
            if not os.path.isdir(self.save_dir):
                raise ValueError(f'save_dir {self.save_dir} not exist')
            self.save_dir = os.path.abspath(self.save_dir)
            self.frame_save_queue = queue.Queue(maxsize=self.frame_save_queue_len)
            save_thread = threading.Thread(target=self._save_worker, daemon=True)
            save_thread.start()
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        img.require_ndarray()
        img.require_HWC()
        img.require_BGR()

    def validate(self, imgs, meta=..., run=True):
        if self.detector_uuid and self.detector_uuid not in meta:
            raise ValueError(f"detector_uuid {self.detector_uuid} not found in meta")
        if self.gps_uuid and self.gps_uuid not in meta:
            raise ValueError(f"gps_uuid {self.gps_uuid} not found in meta")
        return super().validate(imgs, meta, run)

    def _init_frame_recs(self, size=0):
        self.class_names = self._det_processor.class_names
        self.all_cls_id = set(list(self.class_names.keys()))

        for _ in range(size):
            frame_rec = {k: deque(maxlen=self.queue_len) for k in self.class_names.keys()}
            for k, v in frame_rec.items():
                for i in range(self.queue_len):
                    v.append((0, None))
            self.frame_recs.append(frame_rec)

    def stop(self):
        self.frame_save_queue.put_nowait((None, None))

    def _save_worker(self):
        while True:
            try:
                class_name, imagemat_list = self.frame_save_queue.get()
            except queue.Empty:
                self.frame_save_queue.task_done()
                continue
            if imagemat_list is None:
                self.frame_save_queue.task_done()
                break

            timestamp = imagemat_list[len(imagemat_list) // 2].timestamp
            dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            dt_jst = dt_utc + timedelta(hours=self.timezone)
            timestamp = dt_jst.strftime("%Y%m%d_%H%M%S")
            filename = f'{self.save_dir}/{timestamp}_{class_name}'

            os.makedirs(filename, exist_ok=True)

            print('save_jpeg')
            for i, img in enumerate(imagemat_list):
                fn = f'{filename}/{i}.jpeg'
                cv2.imwrite(fn, img.data())
                if self.save_jpeg_gps:
                    try:
                        self._gps_processor._gps.set_jpeg_gps_location(
                            fn, img.info.latlon[0], img.info.latlon[1], fn
                        )
                    except Exception:
                        pass

            if self.save_mp4:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                writer = cv2.VideoWriter(f'{filename}/video.mp4', fourcc, 1, (img.info.W, img.info.H))
                for i, img in enumerate(imagemat_list):
                    writer.write(img.data())
                writer.release()

            self.callback(imagemat_list)

            self.frame_save_queue.task_done()

    def forward_raw(self, imgs_data: List[np.ndarray], imgs_info=..., meta=...):
        self.frame_cnt += 1
        if self.detector_uuid:
            self._det_processor = meta[self.detector_uuid]
        if self.gps_uuid:
            self._gps_processor = meta[self.gps_uuid]
        if not self.frame_recs:
            self._init_frame_recs(len(imgs_data))

        for idx, img in enumerate(imgs_data):
            frame_rec = self.frame_recs[idx]
            detections = self._det_processor.bounding_box_xyxy[idx]

            cls_id = set()
            if len(detections) > 0:
                cls_id = set(detections[:, 5].flatten().tolist())

            for i in cls_id:
                imgc = None
                if self.save_jpeg or self.save_mp4:
                    info = imgs_info[idx]
                    imgc = ImageMat(color_type=ColorType.BGR, info=info).unsafe_update_mat(img.copy())
                frame_rec[i].append((frame_rec[i][-1][0] + 1, imgc))

            for i in self.all_cls_id - cls_id:
                if frame_rec[i][-1][0] > 0:
                    frame_rec[i].append((frame_rec[i][-1][0] - 1, None))

            for k, q in frame_rec.items():
                res = q[-self.ignore_frames_thre - 1][0] > self.detect_frames_thre
                if not res:
                    continue

                for i in range(-self.ignore_frames_thre, 0, -1):
                    res = res and q[i - 1][0] > q[i][0]

                if res:
                    latlon = None
                    if self.save_jpeg or self.save_mp4:
                        save_imgs = [q.pop()[1] for _ in range(self.queue_len)][::-1]
                        save_imgs: List[ImageMat] = [i for i in save_imgs if i is not None]
                        if self.save_jpeg_gps:
                            for i in save_imgs:
                                latlon = self._gps_processor.get_latlon()
                                if latlon:
                                    i.info.latlon = (latlon[0], latlon[1])
                        self.frame_save_queue.put((self.class_names[k], save_imgs))

                    for _ in range(self.queue_len):
                        q.append((0, None))
        return imgs_data

    def release(self):
        if hasattr(self, 'frame_save_queue'):
            self.frame_save_queue.put((None, None))
        return super().release()


class DrawYOLO(ImageMatProcessor):
    title: str = 'draw_yolo'
    draw_box_color: Union[Tuple[int, int, int], List[int]] = Field(
        (0, 255, 0), description="Bounding box color (B, G, R)"
    )
    draw_text_color: Union[Tuple[int, int, int], List[int]] = Field(
        (255, 255, 255), description="Label text color (B, G, R)"
    )
    draw_font_scale: float = Field(0.5, description="Font scale for label text")
    draw_thickness: int = Field(2, description="Line thickness for box and text")
    class_names: Dict[str, str] = {}
    class_colors_code: dict = {
        0: "FF3838",
        1: "FF9D97",
        2: "FF701F",
        3: "FFB21D",
        4: "CFD231",
        5: "48F90A",
        6: "92CC17",
        7: "3DDB86",
        8: "1A9334",
        9: "00D4BB",
        10: "2C99A8",
        11: "00C2FF",
        12: "344593",
    }
    class_colors: Dict = {}
    yolo_uuid: str = ''
    _yolo_processor: 'YOLO' = None

    @staticmethod
    def jp2en(text):
        try:
            import pykakasi

            kks = pykakasi.kakasi()
            text = kks.convert(text)
        except Exception:
            pass
        return ''.join([i['hepburn'].replace('mono', 'butsu') for i in text])

    @staticmethod
    def hex_to_bgr(hex_str: str):
        hex_str = hex_str.lstrip('#')
        return (int(hex_str[4:6], 16), int(hex_str[2:4], 16), int(hex_str[0:2], 16))

    def model_post_init(self, context):
        self.class_colors = {k: self.hex_to_bgr(v) for k, v in self.class_colors_code.items()}
        return super().model_post_init(context)

    def validate_img(self, img_idx, img):
        img.require_np_uint()
        img.require_BGR()
        img.require_HWC()

    def forward_raw(self, imgs_data: List[np.ndarray], imgs_info=..., meta=...):
        res = []

        if self.yolo_uuid:
            self._yolo_processor = meta[self.yolo_uuid]
            if len(self.class_names) != len(self._yolo_processor.class_names):
                self.class_names = {k: self.jp2en(v) for k, v in self._yolo_processor.class_names.items()}

        for idx, img in enumerate(imgs_data):
            res.append(
                self.draw(
                    img,
                    self._yolo_processor.bounding_box_xyxy[idx],
                    self.class_names,
                    self.class_colors,
                )
            )
        return res

    def draw(
        self,
        img: np.ndarray,
        detections: np.ndarray,
        class_names: Dict[str, str] = [],
        class_colors: Dict[str, str] = [],
    ) -> np.ndarray:
        draw_font_scale = int(self.draw_font_scale * img.shape[1] / 320)
        draw_thickness = int(self.draw_thickness * img.shape[1] / 320)
        draw_thickness_d = int(2 * img.shape[1] / 320)
        position_d = int(5 * img.shape[1] / 320)

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = map(float, det[:6])
            cls_id = int(cls_id)

            if class_names and 0 <= cls_id < len(class_names):
                label = f"{class_names[cls_id]} {conf:.2f}"
            else:
                label = f"ID {cls_id} {conf:.2f}"

            box_color = class_colors[cls_id] if len(class_colors) > 0 else (0, 255, 0)

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                draw_thickness,
            )

            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - position_d),
                cv2.FONT_HERSHEY_SIMPLEX,
                draw_font_scale,
                (0, 0, 0),
                draw_thickness + draw_thickness_d,
                cv2.LINE_AA,
            )

            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - position_d),
                cv2.FONT_HERSHEY_SIMPLEX,
                draw_font_scale,
                box_color,
                draw_thickness,
                cv2.LINE_AA,
            )
        return img


class CvImageViewer(ImageMatProcessor):
    title: str = 'cv_image_viewer'
    window_name_prefix: str = Field(default='ImageViewer', description="Prefix for window name")
    resizable: bool = Field(default=False, description="Whether window is resizable")
    scale: Optional[float] = Field(default=None, description="Scale factor for displayed image")
    overlay_texts: List[str] = Field(default_factory=list, description="Text overlays for images")
    save_on_key: Optional[int] = Field(default=ord('s'), description="Key code to trigger image save")
    window_names: List[str] = []
    mouse_pos: Tuple[int, int] = (0, 0)

    yolo_uuid: Optional[str] = Field(default=None, description="UUID key to fetch YOLO results from meta")
    _yolo_processor: 'YOLO' = None
    draw_text_color: tuple = (255, 255, 255)
    draw_font_scale: float = 0.5
    draw_thickness: int = 2
    draw_yolo: Optional['DrawYOLO'] = None

    def model_post_init(self, context):
        self.draw_yolo = DrawYOLO()
        return super().model_post_init(context)

    def validate_img(self, img_idx, img: ImageMat):
        img.require_ndarray()
        img.require_np_uint()
        win_name = f'{self.window_name_prefix}:{self.uuid}:{img_idx}'
        self.window_names.append(win_name)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL if self.resizable else cv2.WINDOW_AUTOSIZE)

    def forward_raw(
        self,
        imgs_data: List[np.ndarray],
        imgs_info: List[ImageMatInfo] = [],
        meta={},
    ) -> List[np.ndarray]:
        if self.yolo_uuid:
            self._yolo_processor = meta[self.yolo_uuid]

        scale = self.scale
        overlay_texts = self.overlay_texts

        for idx, img in enumerate(imgs_data):
            img = img.copy()

            text = overlay_texts[idx] if idx < len(overlay_texts) else ""
            if text:
                cv2.putText(
                    img,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.draw_text_color,
                    self.draw_thickness,
                    cv2.LINE_AA,
                )

            if self._yolo_processor and idx < len(self._yolo_processor.bounding_box_xyxy):
                self.draw_yolo.draw(
                    img,
                    self._yolo_processor.bounding_box_xyxy[idx],
                    self._yolo_processor.class_names,
                )

            win_name = f'{self.window_name_prefix}:{self.uuid}:{idx}'

            if scale is not None:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            cv2.imshow(win_name, img)
            self.cv2_keys(idx, img)
        return imgs_data

    def cv2_keys(self, idx, img):
        key = cv2.waitKey(1) & 0xFF
        if self.save_on_key and key == self.save_on_key:
            filename = f'image_{idx}.png'
            cv2.imwrite(filename, img)
            logger(f'Saved {filename}')
        elif key == ord('e'):
            new_text = input(f"Enter new overlay text for image {idx}: ")
            if idx < len(self.overlay_texts):
                self.overlay_texts[idx] = new_text
            else:
                self.overlay_texts.append(new_text)

    def release(self):
        try:
            [cv2.destroyWindow(n) for n in self.window_names]
        except Exception:
            pass

    def __del__(self):
        self.release()


class YOLOTRT(YOLO):
    title: str = 'YOLO_TRT_detections'
    modelname: str = 'yolov5s6u.engine'
    use_official_predict: bool = False
    conf: float = 0.6

    def build_models(self):
        self._models = {}
        for i, d in enumerate(self.num_devices):
            with torch.cuda.device(d):
                modelname = self.modelname.replace('.trt', (f'_{d}.trt').replace(':', '@'))
                yolo = GeneralTensorRTInferenceModel(modelname, d, 'images', 'output0')
            self._models[d] = yolo
        self.use_official_predict = False

    def predict(self, imgs_data: List[torch.Tensor], imgs_info: List[ImageMatInfo] = []):
        yolo_results = []
        for i, img in enumerate(imgs_data):
            info = imgs_info[i]
            yolo_model = self._models[info.device]
            preds = yolo_model(img)
            preds = ops.non_max_suppression(
                preds,
                self.min_conf,
                self.nms_iou,
                classes=None,
                agnostic=False,
                max_det=self.max_det,
                nc=len(self.class_names),
                in_place=False,
                end2end=False,
                rotated=False,
            )
            if isinstance(preds, list):
                yolo_results += preds
            else:
                yolo_results.append(preds)
        yolo_results_xyxycc: List[np.ndarray] = [r.cpu().numpy() for r in yolo_results]
        return [], yolo_results_xyxycc


try:
    class SegmentationModelsPytorch(ImageMatProcessor):
        import pytorch_lightning as _pl

        class SegmentationModel(_pl.LightningModule):
            class DINOv3Seg(torch.nn.Module):
                def __init__(
                    self,
                    variant="vitb16",
                    hub_repo_dir: str = None,
                    hub_weights: str = None,
                    in_channels: int = 1,
                    out_channels: int = 1,
                    freeze_backbone: bool = True,
                    adapter_mode="repeat",
                ):
                    super().__init__()
                    assert hub_repo_dir and hub_weights, "Provide hub_repo_dir and hub_weights."
                    builder = f"dinov3_{variant}"
                    self.backbone = torch.hub.load(hub_repo_dir, builder, source='local', weights=hub_weights)

                    if in_channels == 3:
                        self.input_adapter = torch.nn.Identity()
                    else:
                        if adapter_mode == "repeat":
                            class RepeatGray3(torch.nn.Module):
                                def forward(self, x):
                                    if x.shape[1] == 1:
                                        return x.repeat(1, 3, 1, 1)
                                    return x

                            self.input_adapter = RepeatGray3()
                        elif adapter_mode == "fixed":
                            def fixed_gray_to_rgb_conv():
                                conv = torch.nn.Conv2d(1, 3, kernel_size=1, bias=True)
                                with torch.no_grad():
                                    conv.weight.fill_(1.0)
                                    conv.bias.zero_()
                                for p in conv.parameters():
                                    p.requires_grad = False
                                return conv

                            self.input_adapter = fixed_gray_to_rgb_conv()
                        elif adapter_mode == "conv":
                            self.input_adapter = torch.nn.Conv2d(in_channels, 3, 1)
                        else:
                            raise ValueError(f"Unknown adapter_mode: {adapter_mode}")

                    self.embed_dim = 768 if "vitb16" in variant.lower() else (1024 if "vitl16" in variant.lower() else 1280)
                    hidden = {768: 256, 1024: 384, 1280: 512}[self.embed_dim]
                    self.proj = torch.nn.Conv2d(self.embed_dim, hidden, 1)

                    def sepconv_block(ch: int) -> torch.nn.Sequential:
                        return torch.nn.Sequential(
                            torch.nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                            torch.nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                            torch.nn.BatchNorm2d(ch),
                            torch.nn.ReLU(inplace=True),
                        )

                    self.dec = torch.nn.Sequential(
                        torch.nn.Conv2d(hidden, hidden, 3, padding=1),
                        torch.nn.BatchNorm2d(hidden),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(hidden, hidden, 3, padding=1),
                        torch.nn.BatchNorm2d(hidden),
                        torch.nn.ReLU(inplace=True),
                        sepconv_block(hidden),
                        torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        sepconv_block(hidden),
                        torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        sepconv_block(hidden),
                        torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        sepconv_block(hidden),
                        torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        torch.nn.Conv2d(hidden, out_channels, 1),
                    )

                    self.is_backbone_frozen = False
                    self._auto_unfreeze_epoch = None
                    if freeze_backbone:
                        self.freeze_backbone()

                def freeze_backbone(self):
                    for p in self.backbone.parameters():
                        p.requires_grad = False
                    self.is_backbone_frozen = True
                    self.backbone.eval()
                    return self

                def unfreeze_backbone(self):
                    for p in self.backbone.parameters():
                        p.requires_grad = True
                    self.is_backbone_frozen = False
                    if self.training:
                        self.backbone.train()
                    return self

                def schedule_unfreeze(self, epoch: int):
                    self._auto_unfreeze_epoch = int(epoch)

                def maybe_auto_unfreeze(self, current_epoch: int):
                    if self._auto_unfreeze_epoch is not None and current_epoch >= self._auto_unfreeze_epoch:
                        if self.is_backbone_frozen:
                            self.unfreeze_backbone()

                def train(self, mode: bool = True):
                    super().train(mode)
                    if self.is_backbone_frozen:
                        self.backbone.eval()
                    return self

                def param_groups(self, lr_head: float, lr_backbone: float, weight_decay: float = 0.0):
                    head = list(self.input_adapter.parameters()) + list(self.proj.parameters()) + list(
                        self.dec.parameters()
                    )
                    bb = [p for p in self.backbone.parameters() if p.requires_grad]
                    groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
                    if bb:
                        groups.append({"params": bb, "lr": lr_backbone, "weight_decay": weight_decay})
                    return groups

                @staticmethod
                def _pad_to_multiple(x, multiple=16):
                    H, W = x.shape[-2:]
                    ph = (multiple - H % multiple) % multiple
                    pw = (multiple - W % multiple) % multiple
                    if ph or pw:
                        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
                    return x, (ph, pw)

                @staticmethod
                def _unpad(x, pad_hw):
                    ph, pw = pad_hw
                    return x[..., : x.shape[-2] - ph if ph else x.shape[-2], : x.shape[-1] - pw if pw else x.shape[-1]]

                def _tokens_to_grid(self, toks, H, W, patch=16):
                    if toks.dim() == 3 and toks.size(1) == (H // patch) * (W // patch) + 1:
                        toks = toks[:, 1:, :]
                    B, N, C = toks.shape
                    h, w = H // patch, W // patch
                    return toks.transpose(1, 2).reshape(B, C, h, w)

                def _forward_tokens(self, x):
                    out = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
                    if isinstance(out, dict):
                        for k in ("x", "x_norm_patchtokens", "x_prenorm", "tokens"):
                            if k in out:
                                return out[k]
                        raise RuntimeError("Backbone dict lacks token tensor.")
                    if torch.is_tensor(out) and out.dim() == 3:
                        return out
                    raise RuntimeError("Unexpected backbone output (expected [B,N,C] tokens or dict).")

                def forward(self, x):
                    x = self.input_adapter(x)
                    x, pad_hw = self._pad_to_multiple(x, 16)
                    H, W = x.shape[-2:]
                    toks = self._forward_tokens(x)
                    grid = self._tokens_to_grid(toks, H, W, 16)
                    logits = self.dec(self.proj(grid))
                    oB, oC, oH, oW = logits.shape
                    if oH != H and W != oW:
                        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                    return self._unpad(logits, pad_hw)

            class BCEDiceLoss(torch.nn.Module):
                def __init__(self, bce_weight=0.5, dice_weight=0.5):
                    super().__init__()
                    self.bce = torch.nn.BCEWithLogitsLoss()
                    self.bce_weight = bce_weight
                    self.dice_weight = dice_weight

                def forward(self, logits, targets, smooth=1e-6):
                    bce_loss = self.bce(logits, targets)
                    preds = torch.clamp(torch.sigmoid(logits), min=1e-7, max=1 - 1e-7)
                    intersection = (preds * targets).sum(dim=(1, 2, 3))
                    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
                    dice = (2.0 * intersection + smooth) / (union + smooth)
                    dice_loss = 1 - dice
                    return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()

            class FocalDiceLoss(torch.nn.Module):
                def __init__(self, alpha=0.8, gamma=2.0, dice_weight=0.5):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.dice_weight = dice_weight

                def focal_loss(self, logits, targets):
                    probs = torch.sigmoid(logits)
                    probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
                    pt = probs * targets + (1 - probs) * (1 - targets)
                    w = self.alpha * (1 - pt).pow(self.gamma)
                    return -(w * pt.log()).mean()

                def dice_loss(self, probs, targets, smooth=1e-6):
                    dims = tuple(range(1, probs.dim()))
                    intersection = (probs * targets).sum(dim=dims)
                    union = probs.sum(dim=dims) + targets.sum(dim=dims)
                    dice = (2.0 * intersection + smooth) / (union + smooth)
                    return 1 - dice.mean()

                def forward(self, logits, targets):
                    probs = torch.sigmoid(logits)
                    fl = self.focal_loss(logits, targets)
                    dl = self.dice_loss(probs, targets)
                    return (1 - self.dice_weight) * fl + self.dice_weight * dl

            class TverskyDiceLoss(torch.nn.Module):
                def __init__(self, alpha=0.3, beta=0.7, tversky_weight=0.5, smooth=1e-6):
                    super().__init__()
                    self.alpha = alpha
                    self.beta = beta
                    self.tversky_weight = tversky_weight
                    self.smooth = smooth

                def forward(self, logits, targets):
                    probs = torch.sigmoid(logits)
                    dims = tuple(range(1, logits.dim()))
                    tp = (probs * targets).sum(dim=dims)
                    fp = ((1 - targets) * probs).sum(dim=dims)
                    fn = (targets * (1 - probs)).sum(dim=dims)
                    tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
                    tversky_loss = 1 - tversky.mean()

                    intersection = (probs * targets).sum(dim=dims)
                    union = probs.sum(dim=dims) + targets.sum(dim=dims)
                    dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                    dice_loss = 1 - dice.mean()

                    return self.tversky_weight * tversky_loss + (1 - self.tversky_weight) * dice_loss

            def __init__(
                self,
                arch_name='DeepLabV3Plus',
                encoder_name='efficientnet-b7',
                encoder_weights='imagenet',
                in_channels=1,
                lr=1e-4,
                dinov3_hub_repo_dir: str = None,
                dinov3_hub_weights: str = None,
            ):
                super().__init__()
                self.loss_fn = self.BCEDiceLoss()
                self.arch_name = arch_name
                self.encoder_name = encoder_name
                self.encoder_weights = encoder_weights
                self.lr = lr
                self.in_channels = in_channels

                self.save_hyperparameters()

                if "dinov3" in str(arch_name).lower():
                    if dinov3_hub_repo_dir is None:
                        dinov3_hub_repo_dir = self.arch_name
                    if dinov3_hub_weights is None:
                        dinov3_hub_weights = self.encoder_weights
                    self.model = self.DINOv3Seg(
                        variant=encoder_name,
                        hub_repo_dir=dinov3_hub_repo_dir,
                        hub_weights=dinov3_hub_weights,
                        in_channels=in_channels,
                    )
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    if in_channels == 1:
                        mean = mean.mean(1, keepdim=True)
                        std = std.mean(1, keepdim=True)
                    self.register_buffer("mean", mean)
                    self.register_buffer("std", std)

                else:
                    import segmentation_models_pytorch as smp

                    self.model = smp.create_model(
                        self.arch_name,
                        self.encoder_name,
                        self.encoder_weights,
                        in_channels=in_channels,
                        classes=1,
                    )
                    params = smp.encoders.get_preprocessing_params(self.encoder_name)
                    std = torch.tensor(params["std"]).view(1, 3, 1, 1)
                    mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
                    if in_channels == 1:
                        std = std.mean(1, keepdim=True)
                        mean = mean.mean(1, keepdim=True)
                    self.register_buffer("std", std)
                    self.register_buffer("mean", mean)

            def forward(self, x: torch.Tensor):
                x = (x - self.mean) / self.std
                return self.model(x)

            def _shared_step(self, batch, stage):
                images, masks = batch
                logits = self(images)
                loss = self.loss_fn(logits, masks)
                preds_bin = (torch.sigmoid(logits) > 0.5).float()
                iou = self._iou_score(preds_bin, masks)
                self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
                self.log(f"{stage}_iou", iou, on_epoch=True, prog_bar=True)
                print()
                return loss

            def training_step(self, batch, batch_idx):
                return self._shared_step(batch, "train")

            def validation_step(self, batch, batch_idx):
                self._shared_step(batch, "val")

            def test_step(self, batch, batch_idx):
                self._shared_step(batch, "test")

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
                return optimizer

            def on_epoch_end(self):
                lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log('lr', lr, prog_bar=True)

            def _iou_score(self, preds, targets, eps=1e-6):
                intersection = (preds * targets).sum(dim=(1, 2, 3))
                union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
                iou = (intersection + eps) / (union + eps)
                return iou.mean()

        title: str = 'segmentation_models_pytorch'
        ckpt_path: str
        current_ckpt_path: str = ''
        device: str
        arch_name: str = 'DeepLabV3Plus'
        encoder_name: str = 'efficientnet-b7'
        encoder_weights: str = 'imagenet'
        in_channels: int = 1
        _model: Optional[SegmentationModel] = None

        def model_post_init(self, context):
            self.load_segmentation_model()
            return super().model_post_init(context)

        def load_segmentation_model(self):
            self._model = self.SegmentationModel.load_from_checkpoint(
                self.ckpt_path, map_location=torch.device(self.device)
            )
            self.current_ckpt_path = self.ckpt_path
            self._model.eval()

        def infer(self, imgs):
            with torch.no_grad():
                test_inputs = imgs.to(self._model.device)
                pred_logits = self._model(test_inputs)
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
                return pred_masks

        def validate_img(self, img_idx, img):
            img.require_BCHW()
            img.require_torch_float()

        def forward_raw(
            self,
            imgs_data: List[torch.Tensor],
            imgs_info: List[ImageMatInfo] = [],
            meta={},
        ) -> List[torch.Tensor]:
            if self.current_ckpt_path != self.ckpt_path:
                self.load_segmentation_model()
            imgs_data = [self.infer(i) for i in imgs_data]
            return imgs_data
except Exception:
    pass
