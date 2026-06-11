/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_SEQ_CUH
#define INCLUDED_GR_CUDA_SEQ_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Write an incrementing little-endian uint64 sequence counter into the first
// 8 bytes of each output item.  `out` points at the first item; items are
// `out_pitch` bytes apart (out_pitch == payload_size + 8).  Item i receives
// (base + i).  Generating the counter on the GPU lets seq_stamp avoid a host
// staging buffer and the per-call stream synchronization that came with it.
void exec_write_seq_counters(
    uint8_t* out, int out_pitch, uint64_t base, int num_items, cudaStream_t stream);

#endif /* INCLUDED_GR_CUDA_SEQ_CUH */
