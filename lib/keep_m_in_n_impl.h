/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_KEEP_M_IN_N_IMPL_H
#define INCLUDED_CUDA_KEEP_M_IN_N_IMPL_H

#include "gather.cuh"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/keep_m_in_n.h>

namespace gr {
namespace cuda {

class keep_m_in_n_impl : public keep_m_in_n, public cuda_block
{
private:
    const size_t d_itemsize;
    const int d_output_multiple;
    int d_m;
    int d_n;
    int d_offset;
    gather_columns d_cols;

    // Validate and apply a new (m, n, offset). Caller holds d_setlock.
    void configure(int m, int n, int offset);

public:
    keep_m_in_n_impl(size_t itemsize, int m, int n, int offset, int output_multiple);
    ~keep_m_in_n_impl() override = default;

    void set_m(int m) override;
    void set_n(int n) override;
    void set_offset(int offset) override;

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_KEEP_M_IN_N_IMPL_H */
