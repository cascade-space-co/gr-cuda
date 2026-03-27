/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ibv_sink.cuh"

__global__ void build_frames_kernel(uint8_t* __restrict__ landing_buf,
                                    int slot_size,
                                    uint32_t first_slot,
                                    uint32_t num_slots,
                                    const uint8_t* __restrict__ header_template,
                                    int header_len,
                                    const uint8_t* __restrict__ payload_src,
                                    int payload_size,
                                    int num_frames)
{
    int frame_idx = blockIdx.x;
    if (frame_idx >= num_frames)
        return;

    uint32_t slot_idx = (first_slot + (uint32_t)frame_idx) % num_slots;
    uint8_t* slot = landing_buf + (uint64_t)slot_idx * slot_size;

    /*
     * All threads cooperate on the 42-byte header copy.  Since header_len
     * is smaller than blockDim.x, each thread copies at most one byte.
     * Header and payload occupy disjoint address ranges within the slot,
     * so no __syncthreads() barrier is needed between the two copies.
     */
    for (int i = threadIdx.x; i < header_len; i += blockDim.x)
        slot[i] = header_template[i];

    const uint8_t* src = payload_src + (uint64_t)frame_idx * payload_size;
    uint8_t* dst = slot + header_len;
    for (int i = threadIdx.x; i < payload_size; i += blockDim.x)
        dst[i] = src[i];
}

void exec_build_frames_kernel(uint8_t* landing_buf,
                              int slot_size,
                              uint32_t first_slot,
                              uint32_t num_slots,
                              const uint8_t* header_template,
                              int header_len,
                              const uint8_t* payload_src,
                              int payload_size,
                              int num_frames,
                              int grid_size,
                              int block_size,
                              cudaStream_t stream)
{
    if (num_frames <= 0)
        return;
    build_frames_kernel<<<num_frames, block_size, 0, stream>>>(landing_buf,
                                                               slot_size,
                                                               first_slot,
                                                               num_slots,
                                                               header_template,
                                                               header_len,
                                                               payload_src,
                                                               payload_size,
                                                               num_frames);
}

void get_build_frames_block_and_grid(int* minGrid, int* minBlock)
{
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, build_frames_kernel, 0, 0);
}
