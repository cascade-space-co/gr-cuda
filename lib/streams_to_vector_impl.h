/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_STREAMS_TO_VECTOR_IMPL_H
#define INCLUDED_CUDA_STREAMS_TO_VECTOR_IMPL_H

#include <gnuradio/cuda/streams_to_vector.h>
#include <gnuradio/cuda/cuda_block.h>
#include <vector>

namespace gr {
namespace cuda {

class streams_to_vector_impl : public streams_to_vector, public cuda_block
{
private:
    size_t d_itemsize;
    size_t d_num_streams;
    void** d_input_ptrs_dev; // Array of pointers on device

public:
    streams_to_vector_impl(size_t itemsize, size_t num_streams);
    ~streams_to_vector_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_STREAMS_TO_VECTOR_IMPL_H */

