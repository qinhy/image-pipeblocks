/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _CUSTOMER_FUNCTIONS_H_
#define _CUSTOMER_FUNCTIONS_H_

#include <cudaEGL.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  COLOR_FORMAT_Y8 = 0,
  COLOR_FORMAT_U8_V8,
  COLOR_FORMAT_RGBA,
  COLOR_FORMAT_NONE
} ColorFormat;

typedef struct {
  /**
  * cuda-process API
  *
  * @param image   : EGL Image to process
  * @param userPtr : point to user alloc data, should be free by user
  */
  void (*fGPUProcess) (EGLImageKHR image, void ** userPtr);

  /**
  * pre-process API
  *
  * @param sBaseAddr  : Mapped Surfaces(YUV) pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param sformat    : surfaces format array
  * @param nsurfcount : surfaces count
  * @param userPtr    : point to user alloc data, should be free by user
  */
  void (*fPreProcess)(void **sBaseAddr,
                      unsigned int *smemsize,
                      unsigned int *swidth,
                      unsigned int *sheight,
                      unsigned int *spitch,
                      ColorFormat *sformat,
                      unsigned int nsurfcount,
                      void ** userPtr);

  /**
  * post-process API
  *
  * @param sBaseAddr  : Mapped Surfaces(YUV) pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param sformat    : surfaces format array
  * @param nsurfcount : surfaces count
  * @param userPtr    : point to user alloc data, should be free by user
  */
  void (*fPostProcess)(void **sBaseAddr,
                      unsigned int *smemsize,
                      unsigned int *swidth,
                      unsigned int *sheight,
                      unsigned int *spitch,
                      ColorFormat *sformat,
                      unsigned int nsurfcount,
                      void ** userPtr);

  /**
  * cuda-process API
  *
  * @param bufSurf : NvBufSurface holding the cuda memory to process
  * @param userPtr : point to user alloc data, should be free by user
  */
  void (*fGPUProcessWithCudaMem) (void *bufSurf, void ** userPtr);
} CustomerFunction;

void init (CustomerFunction * pFuncs);
void deinit (void);

#if defined(__cplusplus)
}
#endif

#endif//_CUSTOMER_FUNCTIONS_H_
