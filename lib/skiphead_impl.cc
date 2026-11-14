/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "skiphead_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <algorithm>

namespace gr {
namespace cuda {

skiphead::sptr skiphead::make(size_t itemsize, uint64_t nitems_to_skip)
{
    return gnuradio::make_block_sptr<skiphead_impl>(itemsize, nitems_to_skip);
}

skiphead_impl::skiphead_impl(size_t itemsize, uint64_t nitems_to_skip)
    : gr::block("skiphead",
                io_signature::make(1, 1, itemsize, cuda_buffer::type),
                io_signature::make(1, 1, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize),
      d_nitems_to_skip(nitems_to_skip),
      d_nskipped(0)
{
    // Output offsets are input offsets minus the skip, so tags are moved in
    // general_work.
    set_tag_propagation_policy(TPP_DONT);
}

int skiphead_impl::general_work(int noutput_items,
                                gr_vector_int& ninput_items,
                                gr_vector_const_void_star& input_items,
                                gr_vector_void_star& output_items)
{
    // Of this call's input, the first `skip` items are dropped and the rest
    // are copied.  The skipped part of the call is still capped by the output
    // space, as in blocks.skiphead.
    const int ninput = std::min(ninput_items[0], noutput_items);
    const int skip = static_cast<int>(
        std::min(d_nitems_to_skip - d_nskipped, static_cast<uint64_t>(ninput)));
    const int ncopy = ninput - skip;

    if (ncopy > 0) {
        const auto in = static_cast<const char*>(input_items[0]);
        check_cuda_errors(cudaMemcpyAsync(output_items[0],
                                          in + skip * d_itemsize,
                                          ncopy * d_itemsize,
                                          cudaMemcpyDefault,
                                          d_stream),
                          "skiphead: cudaMemcpyAsync",
                          d_logger);

        const uint64_t start = nitems_read(0) + skip;
        std::vector<tag_t> tags;
        get_tags_in_range(tags, 0, start, start + ncopy);
        for (auto& tag : tags) {
            tag.offset -= d_nitems_to_skip;
            add_item_tag(0, tag);
        }
    }

    d_nskipped += skip;
    consume_each(ninput);
    return ncopy;
}

} /* namespace cuda */
} /* namespace gr */
