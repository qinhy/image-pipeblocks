Step 2A fix: previous package registered gpu_process but did not call convert_egl_frame_to_torch() inside gpu_process. This fixed version maps the EGLImageKHR and calls convert_egl_frame_to_torch(&eglFrame, 0), writing normalized NCHW FP16/FP32 into the Torch-owned CUDA tensor.

Build:
  make clean && make -j$(nproc)

Test:
  python3 test_nchw_fullres_nvivafilter.py /path/to/rgb_full_test.h264 3000 4000 fp16

If it prints tensor shape mismatch with the actual decoded HxW, rerun with that H W.
