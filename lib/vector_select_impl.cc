/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_select_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <stdexcept>
#include <string>

namespace gr {
namespace cuda {

namespace {

// Used in the constructor's initializer list, where the output signature is
// built before the constructor body can validate the indices.
size_t nonempty_size(const std::vector<int>& indices)
{
    if (indices.empty()) {
        throw std::invalid_argument("vector_select: indices must not be empty");
    }
    return indices.size();
}

} // namespace

vector_select::sptr
vector_select::make(size_t itemsize, size_t vlen, const std::vector<int>& indices)
{
    return gnuradio::make_block_sptr<vector_select_impl>(itemsize, vlen, indices);
}

vector_select_impl::vector_select_impl(size_t itemsize,
                                       size_t vlen,
                                       const std::vector<int>& indices)
    : gr::sync_block("vector_select",
                     io_signature::make(1, 1, itemsize * vlen, cuda_buffer::type),
                     io_signature::make(
                         1, 1, itemsize * nonempty_size(indices), cuda_buffer::type)),
      d_itemsize(itemsize),
      d_vlen(vlen),
      d_num_selected(indices.size())
{
    if (itemsize == 0 || vlen == 0) {
        throw std::invalid_argument(
            "vector_select: itemsize and vlen must be positive");
    }
    check_indices(indices);
    d_cols.set(indices);
}

void vector_select_impl::check_indices(const std::vector<int>& indices) const
{
    for (int i : indices) {
        if (i < 0 || static_cast<size_t>(i) >= d_vlen) {
            throw std::invalid_argument("vector_select: index " + std::to_string(i) +
                                        " is out of range for vlen " +
                                        std::to_string(d_vlen));
        }
    }
}

void vector_select_impl::set_indices(const std::vector<int>& indices)
{
    check_indices(indices);
    if (indices.size() != d_num_selected) {
        throw std::invalid_argument("vector_select: set_indices needs " +
                                    std::to_string(d_num_selected) + " indices, got " +
                                    std::to_string(indices.size()));
    }
    gr::thread::scoped_lock guard(d_setlock);
    d_cols.set(indices);
}

int vector_select_impl::work(int noutput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    gr::thread::scoped_lock guard(d_setlock);
    // One row per input vector; d_cols holds the selected indices.
    exec_gather(input_items[0],
                output_items[0],
                d_cols.device(d_stream),
                noutput_items,
                d_num_selected,
                d_vlen,
                d_itemsize,
                d_stream);
    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
