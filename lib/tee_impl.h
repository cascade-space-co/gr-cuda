/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_TEE_IMPL_H
#define INCLUDED_CUDA_TEE_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/tee.h>

namespace gr {
namespace cuda {

class tee_impl : public tee, public cuda_block
{
private:
    size_t d_itemsize;
    bool d_gpu;

public:
    tee_impl(size_t itemsize, bool gpu);
    ~tee_impl() override = default;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_TEE_IMPL_H */
