/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ibv_source.cuh"

__global__ void strip_headers_kernel(const uint8_t* __restrict__ landing_buf,
                                     const uint32_t* __restrict__ slot_indices,
                                     uint8_t* __restrict__ output,
                                     int header_len,
                                     int slot_size,
                                     int payload_size,
                                     int num_packets)
{
    int pkt_idx = blockIdx.x;
    if (pkt_idx >= num_packets)
        return;

    uint32_t slot = slot_indices[pkt_idx];
    const uint8_t* src = landing_buf + (uint64_t)slot * slot_size + header_len;
    uint8_t* dst = output + (uint64_t)pkt_idx * payload_size;

    for (int i = threadIdx.x; i < payload_size; i += blockDim.x)
        dst[i] = src[i];
}

void exec_strip_headers_kernel(const uint8_t* landing_buf,
                               const uint32_t* slot_indices,
                               uint8_t* output,
                               int header_len,
                               int slot_size,
                               int payload_size,
                               int num_packets,
                               int grid_size,
                               int block_size,
                               cudaStream_t stream)
{
    if (num_packets <= 0)
        return;
    strip_headers_kernel<<<num_packets, block_size, 0, stream>>>(landing_buf,
                                                                 slot_indices,
                                                                 output,
                                                                 header_len,
                                                                 slot_size,
                                                                 payload_size,
                                                                 num_packets);
}

void get_strip_headers_block_and_grid(int* minGrid, int* minBlock)
{
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, strip_headers_kernel, 0, 0);
}
