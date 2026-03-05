/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_ADD_IMPL_H
#define INCLUDED_CUDA_ADD_IMPL_H

#include <gnuradio/cuda/add.h>
#include <gnuradio/cuda/cuda_block.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {
namespace cuda {

template <class T>
class add_impl : public add<T>, public cuda_block
{

private:
    const size_t d_num_inputs;
    const size_t d_vlen;

    // Device memory to hold the array of input pointers
    T** d_input_ptrs_dev;

public:
    add_impl(size_t num_inputs, size_t vlen);
    ~add_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_ADD_IMPL_H */
