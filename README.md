# Image Pipeblocks

This repository contains a small framework for building image processing
pipelines composed of modular **blocks**. Each block consumes and produces
`ImageMat` objects which wrap image data and metadata. The provided blocks use
OpenCV and PyTorch to perform operations such as debayering, resizing,
color-space conversion, tiling, JPEG encoding and basic YOLO inference.

The `test.py` module includes sample functions for constructing CPU and GPU
pipelines (`build_image_pipeline` and `build_image_pipeline_gpu`). These
functions show how to combine the blocks to process Bayer images and produce
JPEG output.

## Installation

Install the minimal dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

GPU support and YOLO functionality require optional packages such as
`torch` and `ultralytics`.

## Usage Example

```python
from test import build_image_pipeline, build_image_pipeline_gpu

# Example list of Bayer numpy arrays
bayer_imgs = [ ... ]

# CPU pipeline
run_cpu = build_image_pipeline(bayer_imgs)
jpeg_buffers = run_cpu(bayer_imgs)

# GPU pipeline
run_gpu = build_image_pipeline_gpu(bayer_imgs)
jpeg_buffers, meta = run_gpu(bayer_imgs)
```

## Running Tests

Execute the unit tests with:

```bash
pytest
```

## License

This project is licensed under the terms of the MIT License. See the
[LICENSE](LICENSE) file for details.
