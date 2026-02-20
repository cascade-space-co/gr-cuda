/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_ADD_CUH
#define INCLUDED_GR_CUDA_ADD_CUH

#include <cuda_runtime.h>

template <typename T>
void exec_kernel_add(T** inputs,
                     T* out,
                     int num_inputs,
                     int grid_size,
                     int block_size,
                     size_t n,
                     cudaStream_t stream);

template <typename T>
void get_add_block_and_grid(int* minGrid, int* minBlock);

#endif /* INCLUDED_GR_CUDA_ADD_CUH */
