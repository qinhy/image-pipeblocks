import unittest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from test import build_image_pipeline, build_image_pipeline_gpu

class TestPipelineFunctions(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH and HAS_CV2, "Requires torch and cv2")
    def test_build_image_pipeline(self):
        bayer_imgs = [np.random.randint(0,256,(32,32),dtype=np.uint8) for _ in range(4)]
        run_pipe = build_image_pipeline(bayer_imgs, debayer_backend='torch', resize_to=(16,16), tile_width=2, jpeg_quality=90)
        result = run_pipe(bayer_imgs)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)

    @unittest.skipUnless(HAS_TORCH and HAS_CV2, "Requires torch and cv2")
    def test_build_image_pipeline_gpu(self):
        bayer_imgs = [np.random.randint(0,256,(16,16),dtype=np.uint8) for _ in range(4)]
        run_pipe = build_image_pipeline_gpu(bayer_imgs, debayer_backend='torch', resize_to=(8,8), tile_width=2, jpeg_quality=80, cuda_device=0)
        result, meta = run_pipe(bayer_imgs)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(meta, dict)

if __name__ == "__main__":
    unittest.main()
