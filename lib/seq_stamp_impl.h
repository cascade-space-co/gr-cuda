/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_SEQ_STAMP_IMPL_H
#define INCLUDED_CUDA_SEQ_STAMP_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/seq_stamp.h>

namespace gr {
namespace cuda {

class seq_stamp_impl : public seq_stamp, public cuda_block
{
private:
    int d_payload_size;
    uint64_t d_counter = 0;

    // 8-byte sequence-number header prepended to each output item.
    static constexpr int SEQ_HDR = sizeof(uint64_t);

public:
    seq_stamp_impl(int payload_size);
    ~seq_stamp_impl() override = default;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SEQ_STAMP_IMPL_H */
