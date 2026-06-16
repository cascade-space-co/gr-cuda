/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ibv_source.cuh"

__global__ void strip_headers_kernel(const uint8_t* __restrict__ landing_buf,
                                     int slot_size,
                                     uint32_t first_slot,
                                     uint32_t num_slots,
                                     uint8_t* __restrict__ output,
                                     int header_len,
                                     int payload_size,
                                     int num_packets)
{
    int pkt_idx = blockIdx.x;
    if (pkt_idx >= num_packets)
        return;

    uint32_t slot = (first_slot + (uint32_t)pkt_idx) % num_slots;
    const uint8_t* src = landing_buf + (uint64_t)slot * slot_size + header_len;
    uint8_t* dst = output + (uint64_t)pkt_idx * payload_size;

    /*
     * Vectorized payload copy using 16-byte (uint4) stores.
     *
     * dst starts at output + pkt_idx * payload_size which may not be
     * 16-byte aligned, so we copy in three phases:
     *
     *   1. Head: byte-copy 0-15 bytes until dst is 16-byte aligned.
     *   2. Bulk: aligned uint4 stores (16 B each).  Source may still be
     *      misaligned, so we load via memcpy into a register-local uint4
     *      and let the compiler pick the right load width.
     *   3. Tail: byte-copy the remaining 0-15 bytes.
     *
     * Aligned 16-byte stores avoid the read-modify-write penalty that
     * byte-granularity stores incur on HBM.
     */
    int head = (16 - (reinterpret_cast<uintptr_t>(dst) & 15)) & 15;
    if (head > payload_size)
        head = payload_size;
    for (int i = threadIdx.x; i < head; i += blockDim.x)
        dst[i] = src[i];

    int n16 = (payload_size - head) >> 4;
    uint4* d4 = reinterpret_cast<uint4*>(dst + head);
    /*
     * Load 16 bytes into a register-local uint4 via memcpy (the compiler
     * picks the right load width for possibly-misaligned src), then do a
     * single aligned 16-byte store through the uint4* pointer.
     *
     * The local variable is critical: it makes the compiler emit
     *   st.global.v4.u32  (one 16-byte store)
     * instead of falling back to
     *   st.global.u8 ×16  (sixteen 1-byte stores)
     * which is what happens if memcpy targets the global pointer directly,
     * because memcpy's void* erases the alignment info the compiler needs.
     */
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        uint4 v;
        memcpy(&v, src + head + i * 16, sizeof(uint4));
        d4[i] = v;
    }

    int tail = head + (n16 << 4);
    for (int i = tail + threadIdx.x; i < payload_size; i += blockDim.x)
        dst[i] = src[i];
}

void exec_strip_headers_kernel(const uint8_t* landing_buf,
                               int slot_size,
                               uint32_t first_slot,
                               uint32_t num_slots,
                               uint8_t* output,
                               int header_len,
                               int payload_size,
                               int num_packets,
                               int block_size,
                               cudaStream_t stream)
{
    if (num_packets <= 0)
        return;
    strip_headers_kernel<<<num_packets, block_size, 0, stream>>>(landing_buf,
                                                                 slot_size,
                                                                 first_slot,
                                                                 num_slots,
                                                                 output,
                                                                 header_len,
                                                                 payload_size,
                                                                 num_packets);
}

void get_strip_headers_block_and_grid(int* minGrid, int* minBlock)
{
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, strip_headers_kernel, 0, 0);
}
