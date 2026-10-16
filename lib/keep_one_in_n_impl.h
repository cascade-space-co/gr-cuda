/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_KEEP_ONE_IN_N_IMPL_H
#define INCLUDED_CUDA_KEEP_ONE_IN_N_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/keep_one_in_n.h>

namespace gr {
namespace cuda {

class keep_one_in_n_impl : public keep_one_in_n, public cuda_block
{
private:
    const size_t d_itemsize;
    int d_n;
    int d_count; // items left until the next kept item, inclusive

public:
    keep_one_in_n_impl(size_t itemsize, int n, int output_multiple);
    ~keep_one_in_n_impl() override = default;

    void set_n(int n) override;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_KEEP_ONE_IN_N_IMPL_H */
