/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_SKIPHEAD_IMPL_H
#define INCLUDED_CUDA_SKIPHEAD_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/skiphead.h>

namespace gr {
namespace cuda {

class skiphead_impl : public skiphead, public cuda_block
{
private:
    const size_t d_itemsize;
    const uint64_t d_nitems_to_skip;
    uint64_t d_nskipped;

public:
    skiphead_impl(size_t itemsize, uint64_t nitems_to_skip);
    ~skiphead_impl() override = default;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SKIPHEAD_IMPL_H */
