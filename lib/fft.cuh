/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_FFT_CUH
#define INCLUDED_GR_CUDA_FFT_CUH

#include <cuda_runtime.h>
#include <cufft.h>

void exec_kernel_window(const cufftComplex* in,
                        cufftComplex* out,
                        const float* window,
                        size_t total_items,
                        size_t fft_size,
                        cudaStream_t stream);
void exec_kernel_real_window(const float* in,
                             cufftComplex* out,
                             const float* window,
                             size_t total_items,
                             size_t fft_size,
                             cudaStream_t stream);
void exec_kernel_real_to_complex(const float* in,
                                 cufftComplex* out,
                                 size_t total_items,
                                 cudaStream_t stream);
void exec_kernel_ifftshift(const cufftComplex* in,
                           cufftComplex* out,
                           size_t total_items,
                           size_t fft_size,
                           cudaStream_t stream);
void exec_kernel_real_ifftshift(const float* in,
                                cufftComplex* out,
                                size_t total_items,
                                size_t fft_size,
                                cudaStream_t stream);
void exec_kernel_window_ifftshift(const cufftComplex* in,
                                  cufftComplex* out,
                                  const float* window,
                                  size_t total_items,
                                  size_t fft_size,
                                  cudaStream_t stream);
void exec_kernel_real_window_ifftshift(const float* in,
                                       cufftComplex* out,
                                       const float* window,
                                       size_t total_items,
                                       size_t fft_size,
                                       cudaStream_t stream);
void exec_kernel_fftshift(const cufftComplex* in,
                          cufftComplex* out,
                          size_t total_items,
                          size_t fft_size,
                          cudaStream_t stream);

#endif /* INCLUDED_GR_CUDA_FFT_CUH */
