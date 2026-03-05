/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace {
constexpr int kBlockSize = 256;
}

__device__ __forceinline__ void compute_shifted_index(
    size_t idx, size_t fft_size, size_t shift, size_t* vec, size_t* src)
{
    *vec = idx / fft_size;
    size_t pos = idx % fft_size;
    *src = (pos + shift) % fft_size;
}

// Apply a real-valued window to complex input.
__global__ void kernel_window(const cufftComplex* in,
                              cufftComplex* out,
                              const float* window,
                              size_t total_items,
                              size_t fft_size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        size_t widx = idx % fft_size;
        float w = window[widx];
        cufftComplex v = in[idx];
        v.x *= w;
        v.y *= w;
        out[idx] = v;
    }
}

// Apply a real-valued window to real input and convert to complex.
__global__ void kernel_real_window(const float* in,
                                   cufftComplex* out,
                                   const float* window,
                                   size_t total_items,
                                   size_t fft_size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        size_t widx = idx % fft_size;
        float w = window[widx];
        float v = in[idx] * w;
        out[idx].x = v;
        out[idx].y = 0.0f;
    }
}

// Convert real input to complex for cuFFT.
__global__ void
kernel_real_to_complex(const float* in, cufftComplex* out, size_t total_items)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        out[idx].x = in[idx];
        out[idx].y = 0.0f;
    }
}

// Apply ifftshift to real input and convert to complex.
__global__ void kernel_real_ifftshift(const float* in,
                                      cufftComplex* out,
                                      size_t total_items,
                                      size_t fft_size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        size_t vec = 0;
        size_t src = 0;
        compute_shifted_index(idx, fft_size, fft_size / 2, &vec, &src);
        float v = in[vec * fft_size + src];
        out[idx].x = v;
        out[idx].y = 0.0f;
    }
}

__global__ void kernel_window_ifftshift(const cufftComplex* in,
                                        cufftComplex* out,
                                        const float* window,
                                        size_t total_items,
                                        size_t fft_size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        size_t vec = 0;
        size_t src = 0;
        compute_shifted_index(idx, fft_size, fft_size / 2, &vec, &src);
        float w = window[src];
        cufftComplex v = in[vec * fft_size + src];
        v.x *= w;
        v.y *= w;
        out[idx] = v;
    }
}

// Apply ifftshift and window to real input, then convert to complex.
__global__ void kernel_real_window_ifftshift(const float* in,
                                             cufftComplex* out,
                                             const float* window,
                                             size_t total_items,
                                             size_t fft_size)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        size_t vec = 0;
        size_t src = 0;
        compute_shifted_index(idx, fft_size, fft_size / 2, &vec, &src);
        float v = in[vec * fft_size + src] * window[src];
        out[idx].x = v;
        out[idx].y = 0.0f;
    }
}

// Apply an FFT shift using the provided split point.
__global__ void kernel_shift(const cufftComplex* in,
                             cufftComplex* out,
                             size_t total_items,
                             size_t fft_size,
                             size_t shift)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        size_t vec = 0;
        size_t src = 0;
        compute_shifted_index(idx, fft_size, shift, &vec, &src);
        out[idx] = in[vec * fft_size + src];
    }
}

void exec_kernel_window(const cufftComplex* in,
                        cufftComplex* out,
                        const float* window,
                        size_t total_items,
                        size_t fft_size,
                        cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    kernel_window<<<grid, kBlockSize, 0, stream>>>(
        in, out, window, total_items, fft_size);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_real_window(const float* in,
                             cufftComplex* out,
                             const float* window,
                             size_t total_items,
                             size_t fft_size,
                             cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    kernel_real_window<<<grid, kBlockSize, 0, stream>>>(
        in, out, window, total_items, fft_size);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_real_to_complex(const float* in,
                                 cufftComplex* out,
                                 size_t total_items,
                                 cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    kernel_real_to_complex<<<grid, kBlockSize, 0, stream>>>(in, out, total_items);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_ifftshift(const cufftComplex* in,
                           cufftComplex* out,
                           size_t total_items,
                           size_t fft_size,
                           cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    // ifftshift uses floor(N/2) for the split point.
    kernel_shift<<<grid, kBlockSize, 0, stream>>>(
        in, out, total_items, fft_size, fft_size / 2);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_real_ifftshift(const float* in,
                                cufftComplex* out,
                                size_t total_items,
                                size_t fft_size,
                                cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    kernel_real_ifftshift<<<grid, kBlockSize, 0, stream>>>(
        in, out, total_items, fft_size);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_window_ifftshift(const cufftComplex* in,
                                  cufftComplex* out,
                                  const float* window,
                                  size_t total_items,
                                  size_t fft_size,
                                  cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    kernel_window_ifftshift<<<grid, kBlockSize, 0, stream>>>(
        in, out, window, total_items, fft_size);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_real_window_ifftshift(const float* in,
                                       cufftComplex* out,
                                       const float* window,
                                       size_t total_items,
                                       size_t fft_size,
                                       cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    kernel_real_window_ifftshift<<<grid, kBlockSize, 0, stream>>>(
        in, out, window, total_items, fft_size);
    check_cuda_errors(cudaGetLastError());
}

void exec_kernel_fftshift(const cufftComplex* in,
                          cufftComplex* out,
                          size_t total_items,
                          size_t fft_size,
                          cudaStream_t stream)
{
    int grid = static_cast<int>((total_items + kBlockSize - 1) / kBlockSize);
    // fftshift uses ceil(N/2) for the split point.
    kernel_shift<<<grid, kBlockSize, 0, stream>>>(
        in, out, total_items, fft_size, (fft_size + 1) / 2);
    check_cuda_errors(cudaGetLastError());
}
