/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_VECTOR_TO_STREAM_IMPL_H
#define INCLUDED_CUDA_VECTOR_TO_STREAM_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/vector_to_stream.h>

namespace gr {
namespace cuda {

class vector_to_stream_impl : public vector_to_stream, public cuda_block
{
private:
    size_t d_itemsize;
    size_t d_vlen;

public:
    vector_to_stream_impl(size_t itemsize, size_t vlen);
    ~vector_to_stream_impl() override = default;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_VECTOR_TO_STREAM_IMPL_H */
