import types
import sys
import importlib
import numpy as np

class FakeYOLO:
    def __init__(self, modelname, task='detect'):
        self.modelname = modelname
        self.task = task
        self.names = {0: 'fake'}
    def to(self, device):
        self.device = device
        return self
    def __call__(self, img, conf=0.6, verbose=False):
        return [f'{self.modelname}-{self.device}']

def test_multiple_images(monkeypatch):
    fake_module = types.ModuleType('ultralytics')
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, 'ultralytics', fake_module)
    import processors
    importlib.reload(processors)

    YOLOBlock = processors.YOLOBlock
    ImageMat = processors.ImageMat

    img1 = ImageMat(np.zeros((4,4,3), dtype=np.uint8), 'RGB')
    img2 = ImageMat(np.zeros((4,4,3), dtype=np.uint8), 'RGB')
    block = YOLO(modelname='fake')
    _, meta = block.validate([img1, img2])

    results = meta['YOLO_detections']
    assert isinstance(results, list)
    assert len(results) == 2
