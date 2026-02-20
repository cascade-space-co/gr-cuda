/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_INTERLEAVE_CUH
#define INCLUDED_GR_CUDA_INTERLEAVE_CUH

#include <cuda_runtime.h>

void exec_interleave(const void** inputs,
                     void* out,
                     int num_streams,
                     int itemsize,
                     int N,
                     int grid_size,
                     int block_size,
                     cudaStream_t stream);

void exec_deinterleave(const void* in,
                       void** outputs,
                       int num_streams,
                       int itemsize,
                       int N,
                       int grid_size,
                       int block_size,
                       cudaStream_t stream);

void get_interleave_block_and_grid(int* minGrid, int* minBlock);

void get_deinterleave_block_and_grid(int* minGrid, int* minBlock);

#endif /* INCLUDED_GR_CUDA_INTERLEAVE_CUH */
