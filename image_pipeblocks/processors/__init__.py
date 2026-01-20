from __future__ import annotations

import json
import multiprocessing
from typing import List, Union

from pydantic import BaseModel

from ..ImageMat import ImageMatProcessor

from .utils import logger
from .basic import DoingNothing, BackUp, Lambda, GPS
from .image_ops import (
    CropImageToDivisibleByNum,
    GaussianBlur,
    CvDebayer,
    NumpyRGBToNumpyBGR,
    NumpyBGRToTorchRGB,
    NumpyPadImage,
    NumpyBayerToTorchBayer,
    NumpyGrayToTorchGray,
    TorchRGBToNumpyBGR,
    TorchGrayToNumpyGray,
    TorchResize,
    CVResize,
)
from .compose import TileNumpyImages, EncodeNumpyToJpeg, CvVideoRecorder
from .analysis import (
    NumpyImageMask,
    TorchImageMask,
    TorchDebayer,
    MockTorchDebayer,
    SlidingWindowSplitter,
    SlidingWindowMerger,
)
from .detection import (
    GeneralTensorRTInferenceModel,
    YOLO,
    SimpleTracking,
    DrawYOLO,
    CvImageViewer,
    YOLOTRT,
)

try:
    from .detection import SegmentationModelsPytorch
except Exception:
    SegmentationModelsPytorch = None


class Processors:
    class DoingNothing(DoingNothing):pass
    class BackUp(BackUp):pass
    class Lambda(Lambda):pass
    class CropImageToDivisibleByNum(CropImageToDivisibleByNum):pass
    class GaussianBlur(GaussianBlur):pass
    class CvDebayer(CvDebayer):pass
    class NumpyRGBToNumpyBGR(NumpyRGBToNumpyBGR):pass
    class NumpyBGRToTorchRGB(NumpyBGRToTorchRGB):pass
    class NumpyPadImage(NumpyPadImage):pass
    class NumpyBayerToTorchBayer(NumpyBayerToTorchBayer):pass
    class NumpyGrayToTorchGray(NumpyGrayToTorchGray):pass
    class TorchRGBToNumpyBGR(TorchRGBToNumpyBGR):pass
    class TorchGrayToNumpyGray(TorchGrayToNumpyGray):pass
    class TorchResize(TorchResize):pass
    class CVResize(CVResize):pass
    class TileNumpyImages(TileNumpyImages):pass
    class EncodeNumpyToJpeg(EncodeNumpyToJpeg):pass
    class CvVideoRecorder(CvVideoRecorder):pass
    class NumpyImageMask(NumpyImageMask):pass
    class TorchImageMask(TorchImageMask):pass
    class TorchDebayer(TorchDebayer):pass
    class MockTorchDebayer(MockTorchDebayer):pass
    class SlidingWindowSplitter(SlidingWindowSplitter):pass
    class SlidingWindowMerger(SlidingWindowMerger):pass
    class YOLO(YOLO):pass
    class SimpleTracking(SimpleTracking):pass
    class DrawYOLO(DrawYOLO):pass
    class CvImageViewer(CvImageViewer):pass
    class YOLOTRT(YOLOTRT):pass
    if GPS is not None:
        class GPS(GPS):pass
    if SegmentationModelsPytorch is not None:
        class SegmentationModelsPytorch(SegmentationModelsPytorch):pass

class ImageMatProcessors(BaseModel):
    @staticmethod
    def dumps(pipes: List[ImageMatProcessor]):
        return json.dumps([json.loads(p.model_dump_json()) for p in pipes])

    @staticmethod
    def loads(pipes_json: str) -> List[ImageMatProcessor]:
        processors = {k: v for k, v in Processors.__dict__.items() if '__' not in k}
        return [processors[f'{p["uuid"].split(":")[0]}'](**p) for p in json.loads(pipes_json)]

    @staticmethod
    def run_once(imgs, meta={}, pipes: List['ImageMatProcessor'] = [], validate=False):
        try:
            for fn in pipes:
                imgs, meta = (fn.validate if validate else fn)(imgs, meta)
        except Exception as e:
            logger(fn.uuid, e)
            raise e
        return imgs, meta

    @staticmethod
    def run(gen, pipes: List['ImageMatProcessor'] = [], meta={}, validate_once=False):
        if isinstance(pipes, str):
            pipes = ImageMatProcessors.loads(pipes)
        for imgs in gen:
            ImageMatProcessors.run_once(imgs, meta, pipes, validate_once)
            if validate_once:
                return

    @staticmethod
    def validate_once(gen, pipes: List['ImageMatProcessor'] = []):
        ImageMatProcessors.run(gen, pipes, validate_once=True)

    @staticmethod
    def worker(pipes_serialized):
        pipes = ImageMatProcessors.loads(pipes_serialized)
        imgs, meta = [], {}
        while True:
            for fn in pipes:
                imgs, meta = fn(imgs, meta)

    @staticmethod
    def run_async(pipes: Union[List[ImageMatProcessor], str]):
        if isinstance(pipes, str):
            pipes_serialized = pipes
        else:
            pipes_serialized = ImageMatProcessors.dumps(pipes)

        p = multiprocessing.Process(target=ImageMatProcessors.worker, args=(pipes_serialized,))
        p.start()
        return p
