#!/usr/bin/env python3
"""
Step 3A: Snapshot + tile a full-resolution torch.cuda.Tensor produced by nvivafilter.

Flow:
  nvivafilter C++/CUDA writes full frame into `full_tensor`
  Python waits for GStreamer EOS / frame_count
  snapshot = full_tensor.detach().clone()
  Python tiles snapshot on GPU
  Print tile stats and save a small CPU preview image

Usage:
  python3 test_step3a_snapshot_tiling.py rgb_full_test.h264 3000 4000 fp16 rgba \
      --so ./libdepthai_cuda_preprocess.so \
      --tile 1024 \
      --overlap 256 \
      --batch-size 4
"""

import argparse
import ctypes
import os
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import torch


def tile_starts(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    last = length - tile

    if starts[-1] != last:
        starts.append(last)

    return starts


def iter_tile_batches(
    image_nchw: torch.Tensor,
    tile: int = 1024,
    overlap: int = 256,
    batch_size: int = 4,
):
    """
    image_nchw: shape (1, 3, H, W), CUDA
    yields:
      tiles:  shape (B, 3, tile, tile)
      coords: list[(x0, y0, x1, y1)]
    """
    assert image_nchw.ndim == 4, image_nchw.shape
    assert image_nchw.shape[0] == 1, image_nchw.shape
    assert image_nchw.shape[1] == 3, image_nchw.shape
    assert image_nchw.is_cuda

    _, c, h, w = image_nchw.shape
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")

    xs = tile_starts(w, tile, stride)
    ys = tile_starts(h, tile, stride)

    batch = []
    coords = []

    for y in ys:
        for x in xs:
            crop = image_nchw[:, :, y:y + tile, x:x + tile]

            if crop.shape[-2:] != (tile, tile):
                padded = torch.zeros(
                    (1, c, tile, tile),
                    device=image_nchw.device,
                    dtype=image_nchw.dtype,
                )
                padded[:, :, :crop.shape[-2], :crop.shape[-1]] = crop
                crop = padded

            batch.append(crop)
            coords.append((x, y, x + tile, y + tile))

            if len(batch) == batch_size:
                yield torch.cat(batch, dim=0), coords
                batch = []
                coords = []

    if batch:
        yield torch.cat(batch, dim=0), coords


def run_gstreamer_decode_to_tensor(
    *,
    h264_path: str,
    so_path: str,
    tensor: torch.Tensor,
    dtype_code: int,
    channel_order: str,
):
    Gst.init(None)

    lib = ctypes.CDLL(so_path)

    lib.set_torch_output_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.set_torch_output_buffer.restype = None

    # Optional symbols from the Step 2A channel-fix package.
    has_channel_order = hasattr(lib, "set_channel_order")
    if has_channel_order:
        lib.set_channel_order.argtypes = [ctypes.c_int]
        lib.set_channel_order.restype = None

    has_frame_count = hasattr(lib, "get_torch_output_frame_count")
    if has_frame_count:
        lib.get_torch_output_frame_count.argtypes = []
        lib.get_torch_output_frame_count.restype = ctypes.c_int

    order_map = {
        "auto": 0,
        "rgba": 1,
        "bgra": 2,
        "argb": 3,
        "abgr": 4,
    }

    if has_channel_order:
        lib.set_channel_order(order_map[channel_order])

    n, c, h, w = tensor.shape

    tensor.zero_()
    torch.cuda.synchronize()

    print("before mean:", tensor.mean())

    lib.set_torch_output_buffer(
        ctypes.c_void_p(tensor.data_ptr()),
        dtype_code,
        n,
        c,
        h,
        w,
    )

    pipeline_desc = f"""
filesrc location={h264_path}
!
h264parse
!
nvv4l2decoder enable-max-performance=true disable-dpb=true enable-full-frame=true
!
video/x-raw(memory:NVMM),format=NV12
!
nvivafilter cuda-process=true customer-lib-name={so_path} silent=false
!
video/x-raw(memory:NVMM),format=RGBA
!
fakesink sync=false
"""

    print("pipeline:")
    print(pipeline_desc)

    pipeline = Gst.parse_launch(pipeline_desc)
    bus = pipeline.get_bus()

    pipeline.set_state(Gst.State.PLAYING)

    while True:
        msg = bus.timed_pop_filtered(
            10 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )

        if msg is None:
            print("timeout waiting for EOS/error")
            break

        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print("GStreamer ERROR:", err)
            print("debug:", debug)
            break

        if msg.type == Gst.MessageType.EOS:
            print("EOS")
            break

    pipeline.set_state(Gst.State.NULL)
    torch.cuda.synchronize()

    if has_frame_count:
        print("frame_count:", lib.get_torch_output_frame_count())

    print("after mean:", tensor.mean())
    print("after min/max:", tensor.min(), tensor.max())


def save_preview_from_nchw(snapshot: torch.Tensor, out_path: str):
    """
    Save a small preview for sanity-checking.
    Expects snapshot as normalized RGB NCHW on CUDA.
    """
    import cv2
    import numpy as np

    _, _, h, w = snapshot.shape
    max_side = 1024
    scale = min(max_side / max(h, w), 1.0)
    ph = max(1, int(h * scale))
    pw = max(1, int(w * scale))

    preview = torch.nn.functional.interpolate(
        snapshot.float(),
        size=(ph, pw),
        mode="bilinear",
        align_corners=False,
    )

    img = (
        preview[0]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )

    # tensor is RGB, OpenCV writes BGR.
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_bgr)
    print("saved preview:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h264_path")
    parser.add_argument("height", type=int)
    parser.add_argument("width", type=int)
    parser.add_argument("dtype", choices=["fp16", "fp32"])
    parser.add_argument("channel_order", choices=["auto", "rgba", "bgra", "argb", "abgr"])
    parser.add_argument("--so", default="./libdepthai_cuda_preprocess.so")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 means process all tile batches")
    args = parser.parse_args()

    h264_path = os.path.abspath(args.h264_path)
    so_path = os.path.abspath(args.so)

    if not os.path.exists(h264_path):
        raise FileNotFoundError(h264_path)
    if not os.path.exists(so_path):
        raise FileNotFoundError(so_path)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dtype_code = 1 if args.dtype == "fp16" else 0

    full_tensor = torch.empty(
        (1, 3, args.height, args.width),
        device="cuda",
        dtype=torch_dtype,
    )

    print("SO_PATH:", so_path)
    print("H264_PATH:", h264_path)
    print("full_tensor:", tuple(full_tensor.shape), full_tensor.dtype, full_tensor.device)

    run_gstreamer_decode_to_tensor(
        h264_path=h264_path,
        so_path=so_path,
        tensor=full_tensor,
        dtype_code=dtype_code,
        channel_order=args.channel_order,
    )

    # Step 3A: stable GPU snapshot.
    print("cloning full_tensor -> snapshot ...")
    snapshot = full_tensor.detach().clone()
    torch.cuda.synchronize()

    print("snapshot:", tuple(snapshot.shape), snapshot.dtype, snapshot.device)
    print("snapshot mean/min/max:", snapshot.mean(), snapshot.min(), snapshot.max())

    save_preview_from_nchw(snapshot, "step3a_snapshot_preview.jpg")

    # Tile snapshot on GPU.
    print(
        f"tiling snapshot: tile={args.tile}, overlap={args.overlap}, "
        f"batch_size={args.batch_size}"
    )

    total_tiles = 0
    batch_count = 0

    for tiles, coords in iter_tile_batches(
        snapshot,
        tile=args.tile,
        overlap=args.overlap,
        batch_size=args.batch_size,
    ):
        batch_count += 1
        total_tiles += tiles.shape[0]

        # Dummy "model" work: stats stay on GPU until printed.
        batch_mean = tiles.mean()
        batch_min = tiles.min()
        batch_max = tiles.max()

        print(
            f"batch {batch_count}: tiles={tuple(tiles.shape)} "
            f"coords[0]={coords[0]} coords[-1]={coords[-1]} "
            f"mean={batch_mean.item():.5f} min={batch_min.item():.5f} max={batch_max.item():.5f}"
        )

        if args.max_batches and batch_count >= args.max_batches:
            break

    print("total tile batches:", batch_count)
    print("total tiles:", total_tiles)


if __name__ == "__main__":
    main()
