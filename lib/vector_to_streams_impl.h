/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_VECTOR_TO_STREAMS_IMPL_H
#define INCLUDED_CUDA_VECTOR_TO_STREAMS_IMPL_H

#include <gnuradio/cuda/vector_to_streams.h>
#include <gnuradio/cuda/cuda_block.h>

namespace gr {
namespace cuda {

class vector_to_streams_impl : public vector_to_streams, public cuda_block
{
private:
    size_t d_itemsize;
    size_t d_num_streams;
    void** d_output_ptrs_dev; // Array of pointers on device

public:
    vector_to_streams_impl(size_t itemsize, size_t num_streams);
    ~vector_to_streams_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_VECTOR_TO_STREAMS_IMPL_H */

