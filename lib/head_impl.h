/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_HEAD_IMPL_H
#define INCLUDED_CUDA_HEAD_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/head.h>

namespace gr {
namespace cuda {

class head_impl : public head, public cuda_block
{
private:
    size_t d_itemsize;
    uint64_t d_nitems;
    uint64_t d_ncopied_items;

public:
    head_impl(size_t sizeof_stream_item, uint64_t nitems);
    ~head_impl() override = default;

    void reset() override { d_ncopied_items = 0; }
    void set_length(uint64_t nitems) override { d_nitems = nitems; }

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_HEAD_IMPL_H */
