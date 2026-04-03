/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_SEQ_STRIP_IMPL_H
#define INCLUDED_CUDA_SEQ_STRIP_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/seq_strip.h>

namespace gr {
namespace cuda {

class seq_strip_impl : public seq_strip, public cuda_block
{
private:
    int d_payload_size;

public:
    seq_strip_impl(int payload_size);
    ~seq_strip_impl() override = default;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SEQ_STRIP_IMPL_H */
