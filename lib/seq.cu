/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "seq.cuh"

__global__ void write_seq_counters_kernel(uint8_t* __restrict__ out,
                                          int out_pitch,
                                          uint64_t base,
                                          int num_items)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_items)
        return;

    uint64_t seq = base + static_cast<uint64_t>(i);

    // out_pitch need not be 8-byte aligned (payload_size is arbitrary), so
    // store via memcpy to avoid an unaligned 64-bit store.  The compiler
    // lowers this 8-byte copy to the appropriate byte/word stores.
    memcpy(out + static_cast<uint64_t>(i) * out_pitch, &seq, sizeof(uint64_t));
}

void exec_write_seq_counters(
    uint8_t* out, int out_pitch, uint64_t base, int num_items, cudaStream_t stream)
{
    if (num_items <= 0)
        return;

    constexpr int block = 256;
    int grid = (num_items + block - 1) / block;
    write_seq_counters_kernel<<<grid, block, 0, stream>>>(
        out, out_pitch, base, num_items);
}
