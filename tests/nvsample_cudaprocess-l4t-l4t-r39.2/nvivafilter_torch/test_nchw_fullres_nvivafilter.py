import ctypes
import os
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import torch

Gst.init(None)

SO_PATH = os.path.abspath("./libdepthai_cuda_preprocess.so")
H264_PATH = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "rgb_full_test.h264")

# Optional args: H W dtype channel_order
# channel_order: auto, rgba, bgra, argb, abgr
# Example: python3 test_nchw_fullres_nvivafilter.py rgb_full_test.h264 3000 4000 fp16 rgba
REQ_H = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
REQ_W = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
DTYPE_NAME = sys.argv[4].lower() if len(sys.argv) > 4 else "fp16"
CHANNEL_ORDER_NAME = sys.argv[5].lower() if len(sys.argv) > 5 else "auto"
CHANNEL_ORDER_MAP = {"auto": 0, "rgba": 1, "bgra": 2, "argb": 3, "abgr": 4}
if CHANNEL_ORDER_NAME not in CHANNEL_ORDER_MAP:
    raise SystemExit(f"unsupported channel_order {CHANNEL_ORDER_NAME}; use auto, rgba, bgra, argb, or abgr")
CHANNEL_ORDER_CODE = CHANNEL_ORDER_MAP[CHANNEL_ORDER_NAME]

if DTYPE_NAME in ("fp16", "float16", "half"):
    TORCH_DTYPE = torch.float16
    DTYPE_CODE = 1
elif DTYPE_NAME in ("fp32", "float32"):
    TORCH_DTYPE = torch.float32
    DTYPE_CODE = 0
else:
    raise SystemExit(f"unsupported dtype {DTYPE_NAME}; use fp16 or fp32")

lib = ctypes.CDLL(SO_PATH)
lib.set_torch_output_buffer.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.set_torch_output_buffer.restype = None
lib.set_channel_order.argtypes = [ctypes.c_int]
lib.set_channel_order.restype = None
lib.get_torch_output_frame_count.argtypes = []
lib.get_torch_output_frame_count.restype = ctypes.c_int
lib.get_last_frame_width.argtypes = []
lib.get_last_frame_width.restype = ctypes.c_int
lib.get_last_frame_height.argtypes = []
lib.get_last_frame_height.restype = ctypes.c_int
lib.get_last_color_format.argtypes = []
lib.get_last_color_format.restype = ctypes.c_int

def run_pipeline():
    pipeline_desc = f"""
filesrc location={H264_PATH}
!
h264parse
!
nvv4l2decoder enable-max-performance=true disable-dpb=true enable-full-frame=true
!
video/x-raw(memory:NVMM),format=NV12
!
nvivafilter cuda-process=true customer-lib-name={SO_PATH} silent=false
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
        msg = bus.timed_pop_filtered(10 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
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

print("SO_PATH:", SO_PATH)
print("H264_PATH:", H264_PATH)
print("requested tensor:", (1, 3, REQ_H, REQ_W), TORCH_DTYPE)
print("channel_order:", CHANNEL_ORDER_NAME, CHANNEL_ORDER_CODE)

tensor = torch.empty((1, 3, REQ_H, REQ_W), device="cuda", dtype=TORCH_DTYPE)
tensor.zero_()
torch.cuda.synchronize()

print("before mean:", tensor.mean())

lib.set_channel_order(CHANNEL_ORDER_CODE)
lib.set_torch_output_buffer(ctypes.c_void_p(tensor.data_ptr()), DTYPE_CODE, 1, 3, REQ_H, REQ_W)
run_pipeline()
torch.cuda.synchronize()

frame_count = lib.get_torch_output_frame_count()
last_w = lib.get_last_frame_width()
last_h = lib.get_last_frame_height()
last_fmt = lib.get_last_color_format()

print("frame_count:", frame_count)
print("last frame HxW:", last_h, last_w)
print("last color format:", last_fmt)

if frame_count <= 0:
    print("No tensor frame was written. If shape mismatch was printed above, rerun with:")
    print(f"  python3 {os.path.basename(__file__)} {H264_PATH} {last_h} {last_w} {DTYPE_NAME}")
    raise SystemExit(2)

print("after shape:", tuple(tensor.shape), tensor.dtype, tensor.device)
print("after mean:", tensor.mean())
print("after min:", tensor.min())
print("after max:", tensor.max())
print("channel means RGB:", tensor[0, 0].mean(), tensor[0, 1].mean(), tensor[0, 2].mean())

# Save a tiny CPU preview of the tensor for sanity checking if PIL is available.
try:
    import numpy as np
    from PIL import Image
    preview = tensor[0].float().clamp(0, 1).permute(1, 2, 0)
    # Downsample by slicing to avoid another dependency.  This is only a debug copy.
    step_y = max(1, preview.shape[0] // 720)
    step_x = max(1, preview.shape[1] // 960)
    preview_np = (preview[::step_y, ::step_x].detach().cpu().numpy() * 255.0).astype(np.uint8)
    out_name = f"torch_nchw_preview_{CHANNEL_ORDER_NAME}.jpg"
    Image.fromarray(preview_np, "RGB").save(out_name)
    print("saved", out_name)
except Exception as e:
    print("preview save skipped:", e)
