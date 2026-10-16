/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gather.cuh"
#include "keep_one_in_n_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <stdexcept>

namespace gr {
namespace cuda {

keep_one_in_n::sptr keep_one_in_n::make(size_t itemsize, int n, int output_multiple)
{
    return gnuradio::make_block_sptr<keep_one_in_n_impl>(itemsize, n, output_multiple);
}

keep_one_in_n_impl::keep_one_in_n_impl(size_t itemsize, int n, int output_multiple)
    : gr::block("keep_one_in_n",
                io_signature::make(1, 1, itemsize, cuda_buffer::type),
                io_signature::make(1, 1, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize)
{
    if (itemsize == 0) {
        throw std::invalid_argument("keep_one_in_n: itemsize must be positive");
    }
    if (output_multiple < 1) {
        throw std::invalid_argument("keep_one_in_n: output_multiple must be positive");
    }
    set_n(n);
    set_output_multiple(output_multiple);
    // The output rate is not fixed, so tags are placed in general_work.
    set_tag_propagation_policy(TPP_DONT);
}

void keep_one_in_n_impl::set_n(int n)
{
    if (n < 1) {
        throw std::invalid_argument("keep_one_in_n: n must be positive");
    }
    gr::thread::scoped_lock guard(d_setlock);
    d_n = n;
    d_count = n;
}

int keep_one_in_n_impl::general_work(int noutput_items,
                                     gr_vector_int& ninput_items,
                                     gr_vector_const_void_star& input_items,
                                     gr_vector_void_star& output_items)
{
    gr::thread::scoped_lock guard(d_setlock);
    const auto in = static_cast<const char*>(input_items[0]);
    const int ninput = ninput_items[0];
    const int n = d_n;
    const int count = d_count;

    // Kept items are in[count - 1], in[count - 1 + n], ... The explicit check
    // is needed because integer division truncates toward zero, so
    // (ninput - count) / n would be 0 rather than -1 when ninput < count.
    const int kept =
        ninput < count ? 0 : std::min((ninput - count) / n + 1, noutput_items);
    int consumed;
    if (kept == noutput_items) {
        // Output is full: stop right after the last kept item.
        consumed = count + (kept - 1) * n;
        d_count = n;
    } else {
        // Input ran out: consume all of it and carry the partial group's
        // count into the next call.
        consumed = ninput;
        d_count = count + kept * n - consumed;
    }

    // One row per kept item, n items apart, starting at the first kept item.
    exec_gather(in + (count - 1) * d_itemsize,
                output_items[0],
                nullptr,
                kept,
                1,
                n,
                d_itemsize,
                d_stream);

    // Each tag moves to the first kept item at or after it. A tag at position
    // p in this call maps to output (p + n - count) / n: positions up to the
    // first kept item (count - 1) map to 0, the next n positions to 1, etc.
    const uint64_t start = nitems_read(0);
    const uint64_t written = nitems_written(0);
    std::vector<tag_t> tags;
    get_tags_in_range(tags, 0, start, start + consumed);
    for (auto& tag : tags) {
        tag.offset = written + (tag.offset - start + n - count) / n;
        add_item_tag(0, tag);
    }

    consume_each(consumed);
    return kept;
}

} /* namespace cuda */
} /* namespace gr */
