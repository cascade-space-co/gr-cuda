/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_IBV_SOURCE_CUH
#define INCLUDED_GR_CUDA_IBV_SOURCE_CUH

#include <cuda_runtime.h>
#include <cstdint>

void exec_strip_headers_kernel(const uint8_t* landing_buf,
                               const uint32_t* slot_indices,
                               uint8_t* output,
                               int header_len,
                               int slot_size,
                               int payload_size,
                               int num_packets,
                               int grid_size,
                               int block_size,
                               cudaStream_t stream);

void get_strip_headers_block_and_grid(int* minGrid, int* minBlock);

#endif /* INCLUDED_GR_CUDA_IBV_SOURCE_CUH */
