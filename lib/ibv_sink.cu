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

    /*
     * Vectorized payload copy using 16-byte (uint4) stores.
     *
     * dst starts at slot + 42 which is generally not 16-byte aligned, so we
     * copy in three phases:
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
    /* See strip_headers_kernel for why the register-local uint4 matters:
     * it makes the compiler emit st.global.v4.u32 instead of st.global.u8 ×16. */
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        uint4 v;
        memcpy(&v, src + head + i * 16, sizeof(uint4));
        d4[i] = v;
    }

    int tail = head + (n16 << 4);
    for (int i = tail + threadIdx.x; i < payload_size; i += blockDim.x)
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
