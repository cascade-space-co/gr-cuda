/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_IBV_SINK_CUH
#define INCLUDED_GR_CUDA_IBV_SINK_CUH

#include <cuda_runtime.h>
#include <cstdint>

void exec_build_frames_kernel(uint8_t* landing_buf,
                              int slot_size,
                              uint32_t first_slot,
                              uint32_t num_slots,
                              const uint8_t* header_template,
                              int header_len,
                              const uint8_t* payload_src,
                              int payload_size,
                              int num_frames,
                              int block_size,
                              cudaStream_t stream);

void get_build_frames_block_and_grid(int* minGrid, int* minBlock);

#endif /* INCLUDED_GR_CUDA_IBV_SINK_CUH */
