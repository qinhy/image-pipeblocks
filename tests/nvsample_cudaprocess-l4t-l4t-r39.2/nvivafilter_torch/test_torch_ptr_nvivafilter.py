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
lib.get_torch_output_frame_count.argtypes = []
lib.get_torch_output_frame_count.restype = ctypes.c_int

# dtype: 0 = fp32, 1 = fp16
DTYPE_CODE = 1
tensor = torch.empty((1, 3, 640, 640), device="cuda", dtype=torch.float16)
tensor.zero_()
torch.cuda.synchronize()

print("SO_PATH:", SO_PATH)
print("H264_PATH:", H264_PATH)
print("before mean:", tensor.mean())

lib.set_torch_output_buffer(
    ctypes.c_void_p(tensor.data_ptr()),
    DTYPE_CODE,
    1,
    3,
    640,
    640,
)

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

torch.cuda.synchronize()
print("frame_count:", lib.get_torch_output_frame_count())
print("after mean:", tensor.mean())
print("after min:", tensor.min())
print("after max:", tensor.max())

# Hard failure is useful for CI / quick shell tests.
if lib.get_torch_output_frame_count() <= 0:
    raise SystemExit("nvivafilter callback did not fill tensor")
if not torch.allclose(tensor.mean(), torch.tensor(0.5, device="cuda", dtype=torch.float16), atol=1e-3):
    raise SystemExit("tensor mean is not 0.5")
