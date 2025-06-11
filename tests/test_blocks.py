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

from ImageMat import ImageMat, ColorType
from processors import (
    CvDebayerBlock,
    TorchResizeBlock,
    NumpyBGRToTorchRGBBlock,
    TileNumpyImagesBlock,
    CVResizeBlock,
    EncodeNumpyToJpegBlock,
    TorchRGBToNumpyBGRBlock,
    NumpyBayerToTorchBayerBlock,
    MergeYoloResultsBlock,
)

class TestImageMatAndBlocks(unittest.TestCase):
    def test_image_mat_none_raises(self):
        with self.assertRaises(ValueError):
            ImageMat(None, color_type="bayer")

    def test_invalid_color_type_raises(self):
        with self.assertRaises(ValueError):
            ImageMat(np.zeros((10,10), dtype=np.uint8), color_type="foo")

    def test_copy_numpy(self):
        arr = np.ones((5,5), dtype=np.uint8)
        img = ImageMat(arr, color_type="grayscale")
        copy = img.copy()
        self.assertTrue(np.array_equal(arr, copy.data()))

    @unittest.skipUnless(HAS_TORCH, "Requires torch")
    def test_copy_torch(self):
        import torch
        t = torch.ones(1,3,4,4)
        img = ImageMat(t, color_type="RGB")
        copy = img.copy()
        self.assertTrue(torch.equal(t, copy.data()))

    @unittest.skipUnless(HAS_CV2 and HAS_TORCH, "Requires cv2 and torch")
    def test_cv_debayer_and_resize(self):
        import torch
        bayer = np.random.randint(0,255,(10,10),dtype=np.uint8)
        img = ImageMat(bayer, color_type="bayer")
        debayer = CvDebayerBlock()
        out,_ = debayer.validate([img])
        self.assertEqual(out[0].data().shape, (10,10,3))

        tensor = torch.rand(1,3,10,10)
        img_t = ImageMat(tensor, color_type="RGB")
        resize = TorchResizeBlock(target_size=(5,5))
        out,_ = resize.validate([img_t])
        self.assertEqual(out[0].data().shape, (1,3,5,5))

    @unittest.skipUnless(HAS_TORCH, "Requires torch")
    def test_numpy_bgr_to_torch_rgb(self):
        bgr = np.random.randint(0,255,(10,10,3),dtype=np.uint8)
        img = ImageMat(bgr, color_type="BGR")
        block = NumpyBGRToTorchRGBBlock()
        out,_ = block.validate([img])
        self.assertEqual(out[0].data().shape, (1,3,10,10))

    def test_tile_numpy_images_block(self):
        imgs = [ImageMat(np.random.randint(0,255,(5,5,3),dtype=np.uint8), color_type="BGR") for _ in range(4)]
        block = TileNumpyImagesBlock(tile_width=2)
        out,_ = block.validate(imgs)
        expected_h = max(img.data().shape[0] for img in imgs)*2
        expected_w = max(img.data().shape[1] for img in imgs)*2
        self.assertEqual(out[0].data().shape[:2], (expected_h, expected_w))

    def test_tile_numpy_images_block_error_non_hwc(self):
        img = ImageMat(np.random.randint(0,255,(5,5),dtype=np.uint8), color_type="grayscale")
        block = TileNumpyImagesBlock(tile_width=2)
        with self.assertRaises(Exception):
            block.validate([img])

    @unittest.skipUnless(HAS_CV2, "Requires cv2")
    def test_cvresize_block(self):
        gray = ImageMat(np.random.randint(0,255,(8,8),dtype=np.uint8), color_type="grayscale")
        color = ImageMat(np.random.randint(0,255,(8,8,3),dtype=np.uint8), color_type="BGR")
        block = CVResizeBlock((4,4))
        out,_ = block.validate([gray,color])
        self.assertEqual(out[0].data().shape, (4,4))
        self.assertEqual(out[1].data().shape, (4,4,3))

    def test_encode_numpy_to_jpeg_block(self):
        img = ImageMat(np.random.randint(0,255,(8,8,3),dtype=np.uint8), color_type="BGR")
        block = EncodeNumpyToJpegBlock(quality=80)
        out,_ = block.validate([img])
        self.assertEqual(out[0].info.color_type, ColorType.JPEG)
        self.assertIsInstance(out[0].data(), np.ndarray)

        img_gray = ImageMat(np.random.randint(0,255,(8,8),dtype=np.uint8), color_type="grayscale")
        block = EncodeNumpyToJpegBlock()
        with self.assertRaises(Exception):
            block.validate([img_gray])

    @unittest.skipUnless(HAS_TORCH, "Requires torch")
    def test_torch_rgb_to_numpy_bgr_block(self):
        import torch
        tensor = torch.rand(1,3,20,20)
        img = ImageMat(tensor, "RGB")
        block = TorchRGBToNumpyBGRBlock()
        out,_ = block.validate([img])
        self.assertEqual(out[0].data().shape, (20,20,3))
        with self.assertRaises(TypeError):
            block.validate([ImageMat(np.zeros((20,20,3),dtype=np.uint8), "BGR")])

    @unittest.skipUnless(HAS_TORCH, "Requires torch")
    def test_numpy_bayer_to_torch_bayer_block(self):
        import torch
        bayer = ImageMat(np.random.randint(0,255,(10,10),dtype=np.uint8), "bayer")
        block = NumpyBayerToTorchBayerBlock()
        out,_ = block.validate([bayer])
        self.assertIsInstance(out[0].data(), torch.Tensor)
        with self.assertRaises(Exception):
            block.validate([ImageMat(np.zeros((10,10,3),dtype=np.uint8), "BGR")])

    def test_merge_yolo_results_block_no_results(self):
        block = MergeYoloResultsBlock("missing")
        img = ImageMat(np.zeros((10,10,3),dtype=np.uint8), "BGR")
        out,meta = block([img],{})
        self.assertEqual(out,[img])

if __name__ == "__main__":
    unittest.main()
