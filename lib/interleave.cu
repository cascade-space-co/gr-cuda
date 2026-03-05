/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

// Uses char* to handle arbitrary item sizes without template specialization.

/*! Interleave N input streams into one output vector stream.
 *  One thread per scalar element (N * num_streams total threads). */
__global__ void
kernel_interleave(const void** inputs, char* out, int num_streams, int itemsize, int N)
{
    int total_scalars = N * num_streams;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_scalars) {
        // Output layout: [S0_0, S1_0, S2_0], [S0_1, S1_1, S2_1], ...
        // idx maps to specific output scalar slot.

        // Which vector index (time step)?
        int vec_idx = idx / num_streams;
        // Which stream index?
        int stream_idx = idx % num_streams;

        // Input layout:
        // Stream 0: [S0_0, S0_1, ...]
        // Stream 1: [S1_0, S1_1, ...]

        const char* in_ptr = (const char*)inputs[stream_idx];

        memcpy(&out[idx * itemsize], &in_ptr[vec_idx * itemsize], itemsize);
    }
}

/*! Deinterleave one input vector stream into N output streams.
 *  One thread per scalar element (N * num_streams total threads). */
__global__ void kernel_deinterleave(
    const char* in, void** outputs, int num_streams, int itemsize, int N)
{
    int total_scalars = N * num_streams;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_scalars) {
        // idx corresponds to scalar index in the INPUT buffer (interleaved)
        // [S0_0, S1_0], [S0_1, S1_1] ...

        int vec_idx = idx / num_streams;
        int stream_idx = idx % num_streams;

        char* out_ptr = (char*)outputs[stream_idx];

        memcpy(&out_ptr[vec_idx * itemsize], &in[idx * itemsize], itemsize);
    }
}

void exec_interleave(const void** inputs,
                     void* out,
                     int num_streams,
                     int itemsize,
                     int N, // number of vectors
                     int grid_size,
                     int block_size,
                     cudaStream_t stream)
{
    kernel_interleave<<<grid_size, block_size, 0, stream>>>(
        inputs, (char*)out, num_streams, itemsize, N);
    check_cuda_errors(cudaGetLastError());
}

void exec_deinterleave(const void* in,
                       void** outputs,
                       int num_streams,
                       int itemsize,
                       int N, // number of vectors
                       int grid_size,
                       int block_size,
                       cudaStream_t stream)
{
    kernel_deinterleave<<<grid_size, block_size, 0, stream>>>(
        (const char*)in, outputs, num_streams, itemsize, N);
    check_cuda_errors(cudaGetLastError());
}

// Helpers to calculate grid size based on total SCALARS, not vectors
void get_interleave_block_and_grid(int* minGrid, int* minBlock)
{
    // We use a generic kernel that doesn't depend on T, just char*
    check_cuda_errors(
        cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_interleave, 0, 0));
}

void get_deinterleave_block_and_grid(int* minGrid, int* minBlock)
{
    check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, kernel_deinterleave, 0, 0));
}
