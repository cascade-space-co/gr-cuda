/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_VECTOR_SELECT_IMPL_H
#define INCLUDED_CUDA_VECTOR_SELECT_IMPL_H

#include "gather.cuh"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/vector_select.h>

namespace gr {
namespace cuda {

class vector_select_impl : public vector_select, public cuda_block
{
private:
    const size_t d_itemsize;
    const size_t d_vlen;
    const size_t d_num_selected;
    gather_columns d_cols;

    void check_indices(const std::vector<int>& indices) const;

public:
    vector_select_impl(size_t itemsize, size_t vlen, const std::vector<int>& indices);
    ~vector_select_impl() override = default;

    void set_indices(const std::vector<int>& indices) override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_VECTOR_SELECT_IMPL_H */
