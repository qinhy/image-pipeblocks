/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "customer_functions.h"
#include "cudaEGL.h"
#include "iva_metadata.h"
#include "nvbufsurface.h"

#define BOX_W 32
#define BOX_H 32

#define CORD_X 64
#define CORD_Y 64
#define MAX_BUFFERS 30
static BBOX rect_data[MAX_BUFFERS];


// -----------------------------------------------------------------------------
// Step 2A Torch CUDA tensor output from nvivafilter frame
//
// Python owns a torch.cuda.Tensor and passes tensor.data_ptr() here through
// ctypes.  The nvivafilter CUDA callback converts the decoded frame to
// normalized RGB NCHW and writes it directly into that Torch CUDA memory.
//
// dtype: 0 = float32, 1 = float16
// shape is expected to be exactly (1, 3, input_h, input_w) for this Step 2A.
// No resize/letterbox is done here yet.
// -----------------------------------------------------------------------------
static void *g_torch_output_ptr = NULL;
static int g_torch_dtype = -1;
static int g_torch_n = 0;
static int g_torch_c = 0;
static int g_torch_h = 0;
static int g_torch_w = 0;
static int g_torch_frame_count = 0;
static int g_last_frame_w = 0;
static int g_last_frame_h = 0;
static int g_last_color_format = 0;
static int g_logged_size_mismatch = 0;
static int g_logged_color_format = 0;
// Channel order override for 4-byte CUDA/EGL frames.
// 0 = auto/use CUDA enum, 1 = RGBA, 2 = BGRA, 3 = ARGB, 4 = ABGR.
static int g_channel_order = 0;
static int g_logged_channel_order = 0;

extern "C" void set_torch_output_buffer(void *ptr, int dtype, int n, int c, int h, int w)
{
  g_torch_output_ptr = ptr;
  g_torch_dtype = dtype;
  g_torch_n = n;
  g_torch_c = c;
  g_torch_h = h;
  g_torch_w = w;
  g_torch_frame_count = 0;
  g_logged_size_mismatch = 0;
  g_logged_color_format = 0;
  g_logged_channel_order = 0;

  fprintf(stderr,
          "[depthai_cuda_preprocess] set_torch_output_buffer ptr=%p dtype=%d shape=(%d,%d,%d,%d)\n",
          ptr, dtype, n, c, h, w);
}

extern "C" void clear_torch_output_buffer(void)
{
  g_torch_output_ptr = NULL;
  g_torch_dtype = -1;
  g_torch_n = g_torch_c = g_torch_h = g_torch_w = 0;
  g_torch_frame_count = 0;
}


extern "C" void set_channel_order(int order)
{
  // 0=auto, 1=RGBA, 2=BGRA, 3=ARGB, 4=ABGR.
  if (order < 0 || order > 4) {
    fprintf(stderr, "[depthai_cuda_preprocess] invalid channel order %d; using auto\n", order);
    order = 0;
  }
  g_channel_order = order;
  g_logged_channel_order = 0;
  fprintf(stderr, "[depthai_cuda_preprocess] set_channel_order=%d (0=auto,1=RGBA,2=BGRA,3=ARGB,4=ABGR)\n", g_channel_order);
}

extern "C" int get_channel_order(void)
{
  return g_channel_order;
}

extern "C" int get_torch_output_frame_count(void)
{
  return g_torch_frame_count;
}

extern "C" int get_last_frame_width(void)
{
  return g_last_frame_w;
}

extern "C" int get_last_frame_height(void)
{
  return g_last_frame_h;
}

extern "C" int get_last_color_format(void)
{
  return g_last_color_format;
}

__device__ __forceinline__ unsigned char clamp_u8_int(int v)
{
  return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__device__ __forceinline__ void write_rgb_to_nchw(
    void *dst, int dtype, int plane_size, int i, float r, float g, float b)
{
  if (dtype == 0) {
    float *out = reinterpret_cast<float *>(dst);
    out[0 * plane_size + i] = r;
    out[1 * plane_size + i] = g;
    out[2 * plane_size + i] = b;
  } else {
    half *out = reinterpret_cast<half *>(dst);
    out[0 * plane_size + i] = __float2half(r);
    out[1 * plane_size + i] = __float2half(g);
    out[2 * plane_size + i] = __float2half(b);
  }
}

__global__ void abgr_to_nchw_kernel(
    const unsigned char *src,
    int width,
    int height,
    int pitch_bytes,
    void *dst,
    int dtype)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const unsigned char *p = src + y * pitch_bytes + x * 4;

  // CU_EGL_COLOR_FORMAT_ABGR is byte order A,B,G,R.
  // Convert to normalized RGB in NCHW.
  float r = ((float)p[3]) * (1.0f / 255.0f);
  float g = ((float)p[2]) * (1.0f / 255.0f);
  float b = ((float)p[1]) * (1.0f / 255.0f);

  int i = y * width + x;
  int plane_size = width * height;
  write_rgb_to_nchw(dst, dtype, plane_size, i, r, g, b);
}

__global__ void rgba_to_nchw_kernel(
    const unsigned char *src,
    int width,
    int height,
    int pitch_bytes,
    void *dst,
    int dtype)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const unsigned char *p = src + y * pitch_bytes + x * 4;

  // CU_EGL_COLOR_FORMAT_RGBA is byte order R,G,B,A.
  float r = ((float)p[0]) * (1.0f / 255.0f);
  float g = ((float)p[1]) * (1.0f / 255.0f);
  float b = ((float)p[2]) * (1.0f / 255.0f);

  int i = y * width + x;
  int plane_size = width * height;
  write_rgb_to_nchw(dst, dtype, plane_size, i, r, g, b);
}


__global__ void fourcc_to_nchw_kernel(
    const unsigned char *src,
    int width,
    int height,
    int pitch_bytes,
    void *dst,
    int dtype,
    int order)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const unsigned char *p = src + y * pitch_bytes + x * 4;

  unsigned char r8, g8, b8;
  if (order == 1) {
    // RGBA byte order: R,G,B,A
    r8 = p[0]; g8 = p[1]; b8 = p[2];
  } else if (order == 2) {
    // BGRA byte order: B,G,R,A
    r8 = p[2]; g8 = p[1]; b8 = p[0];
  } else if (order == 3) {
    // ARGB byte order: A,R,G,B
    r8 = p[1]; g8 = p[2]; b8 = p[3];
  } else {
    // ABGR byte order: A,B,G,R
    r8 = p[3]; g8 = p[2]; b8 = p[1];
  }

  float r = ((float)r8) * (1.0f / 255.0f);
  float g = ((float)g8) * (1.0f / 255.0f);
  float b = ((float)b8) * (1.0f / 255.0f);

  int i = y * width + x;
  int plane_size = width * height;
  write_rgb_to_nchw(dst, dtype, plane_size, i, r, g, b);
}

__global__ void nv12_to_nchw_kernel(
    const unsigned char *y_plane,
    const unsigned char *uv_plane,
    int width,
    int height,
    int pitch_y,
    int pitch_uv,
    int uv_is_vu,
    void *dst,
    int dtype)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  int yy = (int)y_plane[y * pitch_y + x];
  int uv_index = (y >> 1) * pitch_uv + ((x >> 1) << 1);

  int u;
  int v;
  if (uv_is_vu) {
    v = (int)uv_plane[uv_index + 0];
    u = (int)uv_plane[uv_index + 1];
  } else {
    u = (int)uv_plane[uv_index + 0];
    v = (int)uv_plane[uv_index + 1];
  }

  // BT.601 limited-range YUV -> RGB.  This is a good first pass for NV12 H264.
  int c = yy - 16;
  int d = u - 128;
  int e = v - 128;
  c = c < 0 ? 0 : c;

  unsigned char r8 = clamp_u8_int((298 * c + 409 * e + 128) >> 8);
  unsigned char g8 = clamp_u8_int((298 * c - 100 * d - 208 * e + 128) >> 8);
  unsigned char b8 = clamp_u8_int((298 * c + 516 * d + 128) >> 8);

  float r = ((float)r8) * (1.0f / 255.0f);
  float g = ((float)g8) * (1.0f / 255.0f);
  float b = ((float)b8) * (1.0f / 255.0f);

  int i = y * width + x;
  int plane_size = width * height;
  write_rgb_to_nchw(dst, dtype, plane_size, i, r, g, b);
}

static int validate_torch_shape_for_frame(int width, int height)
{
  if (!g_torch_output_ptr) {
    return 0;
  }
  if (g_torch_n != 1 || g_torch_c != 3 || (g_torch_dtype != 0 && g_torch_dtype != 1)) {
    fprintf(stderr,
            "[depthai_cuda_preprocess] invalid tensor config dtype=%d shape=(%d,%d,%d,%d); expected dtype 0/1 and shape (1,3,H,W)\n",
            g_torch_dtype, g_torch_n, g_torch_c, g_torch_h, g_torch_w);
    return 0;
  }
  if (g_torch_w != width || g_torch_h != height) {
    if (!g_logged_size_mismatch) {
      fprintf(stderr,
              "[depthai_cuda_preprocess] tensor shape mismatch: tensor HxW=%dx%d but frame HxW=%dx%d. For Step 2A allocate torch.empty((1,3,%d,%d)).\n",
              g_torch_h, g_torch_w, height, width, height, width);
      g_logged_size_mismatch = 1;
    }
    return 0;
  }
  return 1;
}

static void convert_egl_frame_to_torch(const CUeglFrame *eglFrame, cudaStream_t stream)
{
  if (!eglFrame || !g_torch_output_ptr) {
    return;
  }

  int width = (int)eglFrame->width;
  int height = (int)eglFrame->height;
  int pitch = (int)eglFrame->pitch;
  g_last_frame_w = width;
  g_last_frame_h = height;
  g_last_color_format = (int)eglFrame->eglColorFormat;

  if (!g_logged_color_format) {
    fprintf(stderr,
            "[depthai_cuda_preprocess] EGL frame width=%d height=%d pitch=%d frameType=%d colorFormat=%d\n",
            width, height, pitch, (int)eglFrame->frameType, (int)eglFrame->eglColorFormat);
    g_logged_color_format = 1;
  }

  if (eglFrame->frameType != CU_EGL_FRAME_TYPE_PITCH) {
    fprintf(stderr, "[depthai_cuda_preprocess] unsupported EGL frameType=%d; expected PITCH\n", (int)eglFrame->frameType);
    return;
  }
  if (!validate_torch_shape_for_frame(width, height)) {
    return;
  }

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

  if (eglFrame->eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR ||
      eglFrame->eglColorFormat == CU_EGL_COLOR_FORMAT_RGBA ||
      g_channel_order != 0) {
    int order = g_channel_order;
    if (order == 0) {
      // The enum name and the actual byte layout may differ across Jetson releases.
      // In your R39.2 test, colorFormat=14 produced alpha-as-red with the old ABGR
      // mapping, so prefer RGBA for this 4-byte frame path. Override from Python
      // with set_channel_order() if needed.
      order = 1;
    }
    if (!g_logged_channel_order) {
      fprintf(stderr,
              "[depthai_cuda_preprocess] using 4-byte channel order %d (1=RGBA,2=BGRA,3=ARGB,4=ABGR) for colorFormat=%d\n",
              order, (int)eglFrame->eglColorFormat);
      g_logged_channel_order = 1;
    }
    fourcc_to_nchw_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char *)eglFrame->frame.pPitch[0],
        width, height, pitch, g_torch_output_ptr, g_torch_dtype, order);
  } else if (eglFrame->eglColorFormat == CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR) {
    nv12_to_nchw_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char *)eglFrame->frame.pPitch[0],
        (const unsigned char *)eglFrame->frame.pPitch[1],
        width, height, pitch, pitch, 1, g_torch_output_ptr, g_torch_dtype);
  } else if (eglFrame->eglColorFormat == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR) {
    nv12_to_nchw_kernel<<<grid, block, 0, stream>>>(
        (const unsigned char *)eglFrame->frame.pPitch[0],
        (const unsigned char *)eglFrame->frame.pPitch[1],
        width, height, pitch, pitch, 0, g_torch_output_ptr, g_torch_dtype);
  } else {
    fprintf(stderr,
            "[depthai_cuda_preprocess] unsupported EGL colorFormat=%d. Try making nvivafilter output RGBA or NV12.\n",
            (int)eglFrame->eglColorFormat);
    return;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[depthai_cuda_preprocess] convert kernel launch error: %s\n", cudaGetErrorString(err));
    return;
  }

  // Debug only: deterministic handoff to Python.  Later replace with CUDA events
  // or double buffering for performance.
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "[depthai_cuda_preprocess] cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
    return;
  }

  ++g_torch_frame_count;
  if (g_torch_frame_count <= 5) {
    fprintf(stderr,
            "[depthai_cuda_preprocess] wrote NCHW normalized RGB tensor frame_count=%d shape=(1,3,%d,%d) dtype=%s\n",
            g_torch_frame_count, height, width, g_torch_dtype == 1 ? "fp16" : "fp32");
  }
}

/**
  * Dummy custom pre-process API implematation.
  * It just access mapped surface userspace pointer &
  * memset with specific pattern modifying pixel-data in-place.
  *
  * @param sBaseAddr  : Mapped Surfaces pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param nsurfcount : surfaces count
  */
static void
pre_process (void **sBaseAddr,
                unsigned int *smemsize,
                unsigned int *swidth,
                unsigned int *sheight,
                unsigned int *spitch,
                ColorFormat  *sformat,
                unsigned int nsurfcount,
                void ** usrptr)
{
  /* add your custom pre-process here
     we draw a green block for demo */
  int x, y;
  char * uv = NULL;
  unsigned char * rgba = NULL;
  if (sformat[1] == COLOR_FORMAT_U8_V8) {
    uv = (char *)sBaseAddr[1];
    for (y = 0; y < BOX_H; ++y) {
      for (x = 0; x < BOX_W; ++x) {
        uv[y * spitch[1] + 2 * x] = 0;
        uv[y * spitch[1] + 2 * x + 1] = 0;
      }
    }
  } else if (sformat[0] == COLOR_FORMAT_RGBA) {
    rgba = (unsigned char *)sBaseAddr[0];
     for (y = 0; y < BOX_H*2; y++) {
      for (x = 0; x < BOX_W*8; x+=4) {
       rgba[x + 0] = 0;
       rgba[x + 1] = 0;
       rgba[x + 2] = 0;
       rgba[x + 3] = 0;
      }
        rgba+=spitch[0];
    }
  }
}

/**
  * Dummy custom post-process API implematation.
  * It just access mapped surface userspace pointer &
  * memset with specific pattern modifying pixel-data in-place.
  *
  * @param sBaseAddr  : Mapped Surfaces pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param nsurfcount : surfaces count
  */
static void
post_process (void **sBaseAddr,
                unsigned int *smemsize,
                unsigned int *swidth,
                unsigned int *sheight,
                unsigned int *spitch,
                ColorFormat  *sformat,
                unsigned int nsurfcount,
                void ** usrptr)
{
  /* add your custom post-process here
     we draw a green block for demo */
  int x, y;
  char * uv = NULL;
  int xoffset = (CORD_X * 4);
  int yoffset = (CORD_Y * 2);
  unsigned char * rgba = NULL;
  if (sformat[1] == COLOR_FORMAT_U8_V8) {
    uv = (char *)sBaseAddr[1];
    for (y = 0; y < BOX_H; ++y) {
      for (x = 0; x < BOX_W; ++x) {
        uv[(y + BOX_H * 2) * spitch[1] + 2 * (x + BOX_W * 2)] = 0;
        uv[(y + BOX_H * 2) * spitch[1] + 2 * (x + BOX_W * 2) + 1] = 0;
      }
    }
  } else if (sformat[0] == COLOR_FORMAT_RGBA) {
    rgba = (unsigned char *)sBaseAddr[0];
    rgba += ((spitch[0] * yoffset) + xoffset);
     for (y = 0; y < BOX_H*2; y++) {
      for (x = 0; x < BOX_W*8; x+=4) {
       rgba[(x + xoffset) + 0] = 0;
       rgba[(x + xoffset) + 1] = 0;
       rgba[(x + xoffset) + 2] = 0;
       rgba[(x + xoffset) + 3] = 0;
      }
        rgba+=spitch[0];
    }
  }
}

__global__ void addLabelsKernel(int* pDevPtr, int pitch){
  int row = blockIdx.y*blockDim.y + threadIdx.y + BOX_H;
  int col = blockIdx.x*blockDim.x + threadIdx.x + BOX_W;
  char * pElement = (char*)pDevPtr + row * pitch + col * 2;
  pElement[0] = 0;
  pElement[1] = 0;
  return;
}

static int addLabels(CUdeviceptr pDevPtr, int pitch){
    dim3 threadsPerBlock(BOX_W, BOX_H);
    dim3 blocks(1,1);
    addLabelsKernel<<<blocks,threadsPerBlock>>>((int*)pDevPtr, pitch);
    return 0;
}

static void add_metadata(void ** usrptr)
{
    /* User need to fill rectangle data based on their requirement.
     * Here rectangle data is filled for demonstration purpose only */

    int i;
    static int index = 0;

    rect_data[index].framecnt = index;
    rect_data[index].objectcnt = index;

    for(i=0; i < NUM_LOCATIONS; i++)
    {
        rect_data[index].location_list[i].x1 = index;
        rect_data[index].location_list[i].x2 = index;
        rect_data[index].location_list[i].y1 = index;
        rect_data[index].location_list[i].y2 = index;
    }
    *usrptr = &rect_data[index];
    index++;
    if(!(index % MAX_BUFFERS))
    {
        index = 0;
    }
}

/**
  * Performs CUDA Operations on egl image.
  *
  * @param image : EGL image
  */
static void
gpu_process (EGLImageKHR image, void ** usrptr)
{
  CUresult status;
  CUeglFrame eglFrame;
  CUgraphicsResource pResource = NULL;

  cudaFree(0);
  status = cuGraphicsEGLRegisterImage(&pResource, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    fprintf(stderr, "[depthai_cuda_preprocess] cuGraphicsEGLRegisterImage failed: %d\n", status);
    return;
  }

  status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    fprintf(stderr, "[depthai_cuda_preprocess] cuGraphicsResourceGetMappedEglFrame failed: %d\n", status);
    cuGraphicsUnregisterResource(pResource);
    return;
  }

  // Step 2A: convert the decoded NVMM/EGL frame directly into the Torch-owned
  // CUDA tensor.  This is the line that was missing in the previous Step 2A zip.
  convert_egl_frame_to_torch(&eglFrame, 0);

  status = cuGraphicsUnregisterResource(pResource);
  if (status != CUDA_SUCCESS) {
    fprintf(stderr, "[depthai_cuda_preprocess] cuGraphicsUnregisterResource failed: %d\n", status);
  }
}

static void gpu_process_with_cuda_mem (void *cudaMem, void ** usrptr)
{
  // The normal Jetson surface-array path uses fGPUProcess(EGLImageKHR), where we
  // can inspect CUeglFrame color format and pitch.  Keep this callback minimal.
  NvBufSurface *surf = (NvBufSurface *)cudaMem;
  g_last_frame_w = (int)surf->surfaceList[0].width;
  g_last_frame_h = (int)surf->surfaceList[0].height;
  if (!g_logged_color_format) {
    fprintf(stderr,
            "[depthai_cuda_preprocess] fGPUProcessWithCudaMem called width=%d height=%d pitch=%d. Step 2A conversion is implemented in EGL callback.\n",
            g_last_frame_w, g_last_frame_h, (int)surf->surfaceList[0].pitch);
    g_logged_color_format = 1;
  }
}

extern "C" void
init (CustomerFunction * pFuncs)
{
  // Step 2A uses only the CUDA callback.  Keep pre/post disabled so the sample
  // green-block drawing does not modify the decoded frame.
  pFuncs->fPreProcess = NULL;
  pFuncs->fGPUProcess = gpu_process;
  pFuncs->fPostProcess = NULL;
  pFuncs->fGPUProcessWithCudaMem = gpu_process_with_cuda_mem;
}

extern "C" void
deinit (void)
{
  /* deinitialization */
}
