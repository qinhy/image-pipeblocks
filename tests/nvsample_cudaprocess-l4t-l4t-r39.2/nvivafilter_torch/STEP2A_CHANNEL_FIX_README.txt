Step 2A channel-order fix

Your R39.2 run produced colorFormat=14 and a too-red preview, which means the first mapping treated alpha as red.

This version adds:
  extern "C" void set_channel_order(int order);

Orders:
  0 auto  - default; for 4-byte EGL frames it now uses RGBA
  1 rgba  - bytes R,G,B,A
  2 bgra  - bytes B,G,R,A
  3 argb  - bytes A,R,G,B
  4 abgr  - bytes A,B,G,R

Run:
  make clean && make -j$(nproc)
  python3 test_nchw_fullres_nvivafilter.py ../../../rgb_full_test.h264 3000 4000 fp16 rgba

If the preview is still wrong, try bgra next:
  python3 test_nchw_fullres_nvivafilter.py ../../../rgb_full_test.h264 3000 4000 fp16 bgra

The script saves torch_nchw_preview_<order>.jpg.
