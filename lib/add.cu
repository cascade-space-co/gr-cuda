/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>

/*! Element-wise sum of \p num_inputs arrays into \p out. */
template <typename T>
__global__ void kernel_add(T** inputs, T* out, int num_inputs, int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        T sum = 0;
        for (int k = 0; k < num_inputs; k++) {
            sum += inputs[k][i];
        }
        out[i] = sum;
    }
}

template <>
__global__ void kernel_add<cuFloatComplex>(cuFloatComplex** inputs, 
                                           cuFloatComplex* out, 
                                           int num_inputs, 
                                           int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int k = 0; k < num_inputs; k++) {
            sum = cuCaddf(sum, inputs[k][i]);
        }
        out[i] = sum;
    }
}

template <typename T>
void exec_kernel_add(T** inputs,
                     T* out,
                     int num_inputs,
                     int grid_size,
                     int block_size,
                     size_t n,
                     cudaStream_t stream)
{
    kernel_add<T><<<grid_size, block_size, 0, stream>>>(inputs, out, num_inputs, static_cast<int>(n));
    check_cuda_errors(cudaGetLastError());
}

// std::complex<float> is binary-compatible with cuFloatComplex
template <>
void exec_kernel_add<std::complex<float>>(std::complex<float>** inputs,
                                          std::complex<float>* out,
                                          int num_inputs,
                                          int grid_size,
                                          int block_size,
                                          size_t n,
                                          cudaStream_t stream)
{
    kernel_add<cuFloatComplex>
        <<<grid_size, block_size, 0, stream>>>((cuFloatComplex**)inputs,
                                               (cuFloatComplex*)out,
                                               num_inputs,
                                               static_cast<int>(n));
    check_cuda_errors(cudaGetLastError());
}

template <typename T>
void get_add_block_and_grid(int* minGrid, int* minBlock)
{
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_add<T>, 0, 0));
}

template <>
void get_add_block_and_grid<std::complex<float>>(int* minGrid, int* minBlock)
{
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_add<cuFloatComplex>, 0, 0));
}

#define IMPLEMENT_KERNEL(T)                          \
    template void get_add_block_and_grid<T>(int*, int*); \
    template void exec_kernel_add<T>(     \
        T**, T*, int, int, int, size_t, cudaStream_t);

IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)
