/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_NOP_IMPL_H
#define INCLUDED_CUDA_NOP_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/nop.h>

namespace gr {
namespace cuda {

class nop_impl : public nop, public cuda_block
{
public:
    nop_impl(size_t sizeof_stream_item);
    ~nop_impl() override = default;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_NOP_IMPL_H */
