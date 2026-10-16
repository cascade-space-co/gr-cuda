/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "keep_m_in_n_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace gr {
namespace cuda {

keep_m_in_n::sptr
keep_m_in_n::make(size_t itemsize, int m, int n, int offset, int output_multiple)
{
    return gnuradio::make_block_sptr<keep_m_in_n_impl>(
        itemsize, m, n, offset, output_multiple);
}

keep_m_in_n_impl::keep_m_in_n_impl(
    size_t itemsize, int m, int n, int offset, int output_multiple)
    : gr::block("keep_m_in_n",
                io_signature::make(1, 1, itemsize, cuda_buffer::type),
                io_signature::make(1, 1, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize),
      d_output_multiple(output_multiple)
{
    if (itemsize == 0) {
        throw std::invalid_argument("keep_m_in_n: itemsize must be positive");
    }
    if (output_multiple < 1) {
        throw std::invalid_argument("keep_m_in_n: output_multiple must be positive");
    }
    gr::thread::scoped_lock guard(d_setlock);
    configure(m, n, offset);
}

void keep_m_in_n_impl::configure(int m, int n, int offset)
{
    if (m < 1 || n < 1 || offset < 0) {
        throw std::invalid_argument(
            "keep_m_in_n: m and n must be positive and offset non-negative");
    }
    if (m > n) {
        throw std::invalid_argument("keep_m_in_n: m (" + std::to_string(m) +
                                    ") must not exceed n (" + std::to_string(n) + ")");
    }
    if (offset >= n) {
        throw std::invalid_argument("keep_m_in_n: offset (" + std::to_string(offset) +
                                    ") must be less than n (" + std::to_string(n) +
                                    ")");
    }
    // Output always comes in whole groups of m.
    const int64_t batch = std::lcm(int64_t{ m }, int64_t{ d_output_multiple });
    if (batch > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("keep_m_in_n: lcm(m, output_multiple) = " +
                                    std::to_string(batch) + " is too large");
    }
    d_m = m;
    d_n = n;
    d_offset = offset;

    // Same order as gr::blocks::keep_m_in_n: block[0:excess], then
    // block[offset:n]. Written as offset - (n - m) so it cannot overflow.
    const int excess = offset > n - m ? offset - (n - m) : 0;
    std::vector<int> cols(m);
    for (int j = 0; j < m; j++) {
        cols[j] = j < excess ? j : offset + j - excess;
    }
    d_cols.set(std::move(cols));

    set_output_multiple(static_cast<int>(batch));
    set_relative_rate(m, n);
}

void keep_m_in_n_impl::set_m(int m)
{
    gr::thread::scoped_lock guard(d_setlock);
    configure(m, d_n, d_offset);
}

void keep_m_in_n_impl::set_n(int n)
{
    gr::thread::scoped_lock guard(d_setlock);
    configure(d_m, n, d_offset);
}

void keep_m_in_n_impl::set_offset(int offset)
{
    gr::thread::scoped_lock guard(d_setlock);
    configure(d_m, d_n, offset);
}

void keep_m_in_n_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
    gr::thread::scoped_lock guard(d_setlock);
    ninput_items_required[0] = d_n * (noutput_items / d_m);
}

int keep_m_in_n_impl::general_work(int noutput_items,
                                   gr_vector_int& ninput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    gr::thread::scoped_lock guard(d_setlock);
    const int blocks = std::min(noutput_items / d_m, ninput_items[0] / d_n);
    // One row per block of n inputs; d_cols picks its m kept items.
    exec_gather(input_items[0],
                output_items[0],
                d_cols.device(d_stream),
                blocks,
                d_m,
                d_n,
                d_itemsize,
                d_stream);

    consume_each(blocks * d_n);
    return blocks * d_m;
}

} /* namespace cuda */
} /* namespace gr */
