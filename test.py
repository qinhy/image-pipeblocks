import json
from generator import ImageMatGenerators, VideoFrameGenerator, XVSdkRGBDGenerator
import processors
from processors import *
from ImageMat import ImageMat, ColorType

def test1():
    # Test ImageMat creation for a Bayer numpy image
    bayer_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img_mat = ImageMat(color_type="bayer").build(bayer_image)
    print("ImageMat created:", img_mat.info)

    # Test CvDebayerBlock
    debayer_block = CvDebayerBlock()
    debayered_imgs, _ = debayer_block.validate([img_mat])
    print("Debayered image shape:", debayered_imgs[0].data().shape)

    # Test TorchResizeBlock
    torch_image = torch.rand(1, 3, 100, 100)
    img_mat_torch = ImageMat(color_type="RGB").build(torch_image)
    resize_block = TorchResizeBlock(target_size=(50, 50))
    resized_imgs, _ = resize_block.validate([img_mat_torch])
    print("Resized torch image shape:", resized_imgs[0].data().shape)

    # Test NumpyBGRToTorchRGBBlock
    bgr_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_mat_bgr = ImageMat(color_type="BGR").build(bgr_image)
    bgr_to_rgb_block = NumpyBGRToTorchRGBBlock()
    rgb_imgs, _ = bgr_to_rgb_block.validate([img_mat_bgr])
    print("Converted torch RGB image shape:", rgb_imgs[0].data().shape)

    # Test TileNumpyImagesBlock
    tile_block = TileNumpyImagesBlock(tile_width=2)
    images = [ImageMat(color_type="BGR").build(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)) for _ in range(4)]
    tiled_imgs, _ = tile_block.validate(images)
    print("Tiled image shape:", tiled_imgs[0].data().shape)
    print("All tests passed.")

    print("/n--- Additional Unit Tests ---")
    # 1. Test error on None input for ImageMat
    try:
        ImageMat(color_type="bayer").build(None)
        print("ERROR: None input did not raise.")
    except ValueError as e:
        print("Pass: ImageMat(...) raised ValueError:", str(e))

    # 2. Test error for unsupported color type in ImageMatInfo
    try:
        ImageMat(color_type="foo").build(np.zeros((10, 10), dtype=np.uint8))
        print("ERROR: Unsupported color_type did not raise.")
    except ValueError as e:
        print("Pass: Unsupported color_type raised ValueError:", str(e))

    # 3. Test copy for numpy and torch
    a = ImageMat(color_type="grayscale").build(np.ones((5, 5), dtype=np.uint8))
    b = a.copy()
    assert np.array_equal(a.data(), b.data()), "Copy failed for numpy"
    print("Pass: ImageMat.copy() for numpy array")

    torch_a = ImageMat(color_type="RGB").build(torch.ones(1, 3, 4, 4))
    torch_b = torch_a.copy()
    assert torch.equal(torch_a.data(), torch_b.data()), "Copy failed for torch"
    print("Pass: ImageMat.copy() for torch tensor")

    # 4. TorchResizeBlock error for wrong input shape
    try:
        TorchResizeBlock(target_size=(20, 20)).validate([ImageMat(np.zeros((10, 10), dtype=np.uint8), "grayscale")])
        print("ERROR: TorchResizeBlock accepted numpy input.")
    except TypeError:
        print("Pass: TorchResizeBlock rejects non-torch input.")

    # 5. NumpyBGRToTorchRGBBlock error for wrong color
    try:
        NumpyBGRToTorchRGBBlock().validate([ImageMat(np.zeros((10, 10, 3), dtype=np.uint8), "RGB")])
        print("ERROR: NumpyBGRToTorchRGBBlock accepted wrong color.")
    except Exception as e:
        print("Pass: NumpyBGRToTorchRGBBlock rejects non-BGR input.")

    # 6. Test CVResizeBlock on HW and HWC
    gray_img = ImageMat(color_type="grayscale").build(np.random.randint(0, 255, (8, 8), dtype=np.uint8))
    color_img = ImageMat(color_type="BGR").build(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
    cvresize = CVResizeBlock(target_size=(4, 4))
    imgs, _ = cvresize.validate([gray_img, color_img])
    assert imgs[0].data().shape == (4, 4) or imgs[0].data().shape == (4, 4, 3)
    print("Pass: CVResizeBlock resizes HW and HWC formats.")

    # 7. TileNumpyImagesBlock error for non-HWC shape
    try:
        TileNumpyImagesBlock(2).validate([gray_img])
        print("ERROR: TileNumpyImagesBlock accepted HW shape.")
    except Exception as e:
        print("Pass: TileNumpyImagesBlock rejects HW input.")

    # 8. EncodeNumpyToJpegBlock positive and error test
    encoder = EncodeNumpyToJpegBlock(quality=80)
    encoded, _ = encoder.validate([color_img])
    assert isinstance(encoded[0].data(), np.ndarray) and encoded[0].data().dtype == np.uint8
    print("Pass: EncodeNumpyToJpegBlock works on HWC.")

    try:
        EncodeNumpyToJpegBlock().validate([gray_img])
        print("Pass: EncodeNumpyToJpegBlock encodes grayscale (HW) images (should raise).")
    except Exception as e:
        print("Pass: EncodeNumpyToJpegBlock correctly rejects HW input.")

    # 9. TorchRGBToNumpyBGRBlock: accept torch RGB, error for wrong shape
    rgb_tensor = torch.rand(1, 3, 20, 20)
    rgb_img = ImageMat(color_type="RGB").build(rgb_tensor)
    out, _ = TorchRGBToNumpyBGRBlock().validate([rgb_img])
    assert out[0].data().shape == (20, 20, 3)
    print("Pass: TorchRGBToNumpyBGRBlock converts to numpy BGR.")

    try:
        TorchRGBToNumpyBGRBlock().validate([ImageMat(np.zeros((20, 20, 3), dtype=np.uint8), "BGR")])
        print("ERROR: TorchRGBToNumpyBGRBlock accepted numpy.")
    except TypeError:
        print("Pass: TorchRGBToNumpyBGRBlock rejects numpy input.")

    # 10. NumpyBayerToTorchBayerBlock works and errors
    bayer_np = ImageMat(color_type="bayer").build(np.random.randint(0, 255, (10, 10), dtype=np.uint8))
    result, _ = NumpyBayerToTorchBayerBlock().validate([bayer_np])
    assert isinstance(result[0].data(), torch.Tensor)
    print("Pass: NumpyBayerToTorchBayerBlock converts Bayer.")

    try:
        NumpyBayerToTorchBayerBlock().validate([ImageMat(np.zeros((10, 10, 3), dtype=np.uint8), "BGR")])
        print("ERROR: NumpyBayerToTorchBayerBlock accepted non-HW input.")
    except Exception as e:
        print("Pass: NumpyBayerToTorchBayerBlock rejects non-HW input.")

    # 11. MergeYoloResultsBlock: no result in meta, returns as is
    merge = MergeYoloResultsBlock(yolo_results_uuid="not-in-meta")
    input_imgs = [ImageMat(color_type="BGR").build(np.zeros((10, 10, 3), dtype=np.uint8))]
    out, meta = merge(input_imgs, {})
    assert out == input_imgs
    print("Pass: MergeYoloResultsBlock returns input if no YOLO results in meta.")

    print("All additional tests passed.")

def build_image_pipeline(
    bayer_images: List[np.ndarray],
    debayer_backend: str = 'torch',  # or 'cv2'
    resize_to: Tuple[int, int] = (256, 256),
    tile_width: int = 2,
    jpeg_quality: int = 90,
) -> Tuple[List[np.ndarray], dict]:
    """
    Build the CPU image pipeline and return a callable.

    The returned callable accepts a list of Bayer numpy arrays and
    processes them through the steps:
    Bayer (np) -> Torch -> Debayer -> Resize -> Numpy -> Tile -> JPEG.
    It yields JPEG-encoded numpy buffers for each processed frame.
    """
    def init(bayer_images):
        meta = {}
        # 1. Wrap Bayer images as ImageMat objects
        bayer_mats = [ImageMat(color_type="bayer").build(img) for img in bayer_images]
        return bayer_mats,meta
    
    bayer_mats,meta = init(bayer_images)

    # 2. Numpy Bayer -> Torch Bayer
    np2torch_block = NumpyBayerToTorchBayerBlock()
    torch_bayer_imgs, meta = np2torch_block.validate(bayer_mats, meta)

    # 3. Debayer (Demosaic) to RGB
    if debayer_backend == 'torch':
        debayer_block = TorchDebayerBlock()
    else:
        # cv2: use BG2BGR, change as needed for your Bayer pattern
        debayer_block = CvDebayerBlock(format=cv2.COLOR_BAYER_BG2BGR)
    torch_rgb_imgs, meta = debayer_block.validate(torch_bayer_imgs, meta)

    # 4. Resize
    resize_block = TorchResizeBlock(target_size=resize_to)
    torch_rgb_imgs, meta = resize_block.validate(torch_rgb_imgs, meta)

    # 5. Torch RGB -> Numpy BGR
    torch2np_block = TorchRGBToNumpyBGRBlock()
    bgr_imgs, meta = torch2np_block.validate(torch_rgb_imgs, meta)

    # 6. Tile images
    tile_block = TileNumpyImagesBlock(tile_width=tile_width)
    tiled_imgs, meta = tile_block.validate(bgr_imgs, meta)

    # 7. Encode to JPEG
    jpeg_block = EncodeNumpyToJpegBlock(quality=jpeg_quality)
    jpeg_imgs, meta = jpeg_block.validate(tiled_imgs, meta)
    
    def run(bayer_images: List[np.ndarray]):
        # raw data in
        imgs, meta = init(bayer_images)
        imgs, meta = np2torch_block(imgs, meta)
        imgs, meta = debayer_block(imgs, meta)
        imgs, meta = resize_block(imgs, meta)
        imgs, meta = torch2np_block(imgs, meta)
        imgs, meta = tile_block(imgs, meta)
        imgs, meta = jpeg_block(imgs, meta)
        # raw data out
        return [i.data() for i in imgs]

    return run
    # # Return JPEG data (np.ndarray, .img_data) and meta
    # return [img.img_data for img in jpeg_imgs], meta

def build_image_pipeline_gpu(
    bayer_images: List[np.ndarray],
    debayer_backend: str = 'torch',  # or 'cv2'
    resize_to: Tuple[int, int] = (256, 256),
    tile_width: int = 2,
    jpeg_quality: int = 90,
    torch_dtype=torch.float32,
    cuda_device: int = 0,  # Set to -1 for CPU fallback
) -> callable:
    """
    Returns a callable pipeline that:
    Bayer (np) -> Torch Bayer (GPU) -> Debayer (GPU) -> Resize (GPU)
    -> Numpy BGR (CPU) -> Tile (CPU) -> JPEG (CPU)
    """

    def init(bayer_images):
        meta = {}
        bayer_mats = [ImageMat(color_type="bayer").build(img) for img in bayer_images]
        return bayer_mats, meta

    bayer_mats, meta = init(bayer_images)

    # 2. Numpy Bayer -> Torch Bayer (on GPU)
    # The block will detect CUDA availability; we force device here.
    np2torch_block = NumpyBayerToTorchBayerBlock(dtype=torch_dtype,gpu=True)

    torch_bayer_imgs, meta = np2torch_block.validate(bayer_mats, meta)

    # 3. Debayer (GPU)
    if debayer_backend == 'torch':
        debayer_block = TorchDebayerBlock()
    else:
        debayer_block = CvDebayerBlock(format=cv2.COLOR_BAYER_BG2BGR)
    torch_rgb_imgs, meta = debayer_block.validate(torch_bayer_imgs, meta)

    # 4. Resize (GPU)
    resize_block = TorchResizeBlock(target_size=resize_to)
    torch_rgb_imgs, meta = resize_block.validate(torch_rgb_imgs, meta)

    # 5. Torch RGB -> Numpy BGR (CPU, so move to CPU here)
    torch2np_block = TorchRGBToNumpyBGRBlock()

    bgr_imgs, meta = torch2np_block.validate(torch_rgb_imgs, meta)

    # 6. Tile (CPU)
    tile_block = TileNumpyImagesBlock(tile_width=tile_width)
    tiled_imgs, meta = tile_block.validate(bgr_imgs, meta)

    # 7. JPEG (CPU)
    jpeg_block = EncodeNumpyToJpegBlock(quality=jpeg_quality)
    jpeg_imgs, meta = jpeg_block.validate(tiled_imgs, meta)

    def run(bayer_images: List[np.ndarray]) -> List[np.ndarray]:
        imgs, meta = init(bayer_images)
        imgs, meta = np2torch_block(imgs, meta)
        imgs, meta = debayer_block(imgs, meta)
        imgs, meta = resize_block(imgs, meta)
        imgs, meta = torch2np_block(imgs, meta)
        imgs, meta = tile_block(imgs, meta)
        imgs, meta = jpeg_block(imgs, meta)
        return [i.data() for i in imgs], meta

    return run

def test_build_image_pipeline():
    # Generate 4 random Bayer images (grayscale, uint8, HW shape)
    def generate_random_bayer_images(num_images=4, height=128, width=128):
        return [np.random.randint(0, 256, (height, width), dtype=np.uint8) for _ in range(num_images)]

    bayer_imgs = generate_random_bayer_images(num_images=4, height=128, width=128)

    # Now, run your pipeline!
    run_pipe = build_image_pipeline(
        bayer_images=bayer_imgs,
        debayer_backend='torch',    # or 'cv2'
        resize_to=(256, 256),
        tile_width=2,
        jpeg_quality=90,
    )
    
    encoded_jpegs = run_pipe(bayer_imgs)
    # Print result info
    print(f"Number of JPEG images: {len(encoded_jpegs)}")
    for i, jpeg_buf in enumerate(encoded_jpegs):
        print(f"JPEG {i}: shape={jpeg_buf.shape}, dtype={jpeg_buf.dtype}, first 10 bytes={jpeg_buf[:10].flatten()}")

def test_build_image_pipeline_gpu():
    bayer_images = [np.random.randint(0,256,(480,640),np.uint8) for _ in range(4)]

    pipeline = build_image_pipeline_gpu(bayer_images, debayer_backend='torch', resize_to=(512, 512), tile_width=2, jpeg_quality=95, cuda_device=0)
    encoded_jpegs, meta = pipeline(bayer_images)
    for i, jpeg_buf in enumerate(encoded_jpegs):
        print(f"JPEG {i}: shape={jpeg_buf.shape}, dtype={jpeg_buf.dtype}, first 10 bytes={jpeg_buf[:10].flatten()}")

def test_vid_show(mp4s=[]):
    if len(mp4s)==0:return print('Not mp4s.')
    pipes = []
    # Create multi-video generator
    gen = multi_video_gen = VideoFrameGenerator(
        sources=mp4s,
        color_types=[ColorType.BGR for _ in mp4s],  # Or ColorType.RGB, as needed
        # scale=0.5,                 # Resize frames for speed, optional
        # step=1,                    # No frame skipping
        # max_frames=10              # Limit to 10 frames for quick testing
    )
    # Create a simple viewer
    viewer = CvImageViewer(
        window_name_prefix="MultiVideoTest",
        resizable=False,
        scale=0.25,
        # overlay_texts=None
    )
    pipes.append(viewer)

    # validate
    for i, imgs in enumerate(multi_video_gen):
        ImageMatProcessors.run_once(imgs,pipes=pipes,validate=True)
        print("validate complete.")
        break

    # dumps
    gen_json = ImageMatGenerators.dumps(gen)
    pipes_json = ImageMatProcessors.dumps(pipes)
    del pipes,gen
    print(gen_json)
    print(pipes_json)

    # loads
    pipes = ImageMatProcessors.loads(pipes_json)
    gen = ImageMatGenerators.loads(gen_json)

    # return gen, pipes
    for imgs in gen:ImageMatProcessors.run_once(imgs,pipes=pipes)


def test_xvsdk_show(output_filename="./out.mp4"):
    pipes = [
        # Create a simple viewer
        TileNumpyImagesBlock(tile_width=1),        
        CvImageViewer(
            window_name_prefix="MultiVideoTest",
            resizable=False,
            scale=0.5,
            # overlay_texts=None
        ),]
    
    if output_filename:
        pipes += [
            CvVideoRecorder(
                output_filename=output_filename,
                fps=30,
            ),
        ]

    # Create multi-video generator
    gen =  XVSdkRGBDGenerator(
        color_resolution=XVSdkRGBDGenerator.RGBResolution.RGB_1280x720,
    )

    # validate
    for i, imgs in enumerate(gen):
        ImageMatProcessors.run_once(imgs,pipes=pipes,validate=True)
        print("validate complete.")
        break

    # return gen, pipes
    for imgs in gen:ImageMatProcessors.run_once(imgs,pipes=pipes)

# ========== Usage Example ==========
if __name__ == "__main__":
    # test_xvsdk_show()
    test_vid_show(['./data/Serene Valley Vista.avi',])
    test1()
    test_build_image_pipeline()
    test_build_image_pipeline_gpu()
