/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "head_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <algorithm>

namespace gr {
namespace cuda {

head::sptr head::make(size_t sizeof_stream_item, uint64_t nitems)
{
    return gnuradio::make_block_sptr<head_impl>(sizeof_stream_item, nitems);
}

head_impl::head_impl(size_t sizeof_stream_item, uint64_t nitems)
    : sync_block("head",
                 io_signature::make(1, 1, sizeof_stream_item, cuda_buffer::type),
                 io_signature::make(0, 1, sizeof_stream_item, cuda_buffer::type)),
      d_itemsize(sizeof_stream_item),
      d_nitems(nitems),
      d_ncopied_items(0)
{
}

int head_impl::work(int noutput_items,
                    gr_vector_const_void_star& input_items,
                    gr_vector_void_star& output_items)
{
    // Done is reported on the call after the last item passes, so the items
    // produced by the previous call are published normally first.
    if (d_ncopied_items >= d_nitems) {
        return WORK_DONE;
    }

    const auto n = static_cast<int>(
        std::min(d_nitems - d_ncopied_items, static_cast<uint64_t>(noutput_items)));

    // With no output connected the block acts as a sink: items are counted
    // and consumed without being read.  Otherwise the copy is queued on
    // d_stream, and cuda_buffer makes the consumer wait for it.
    if (!output_items.empty()) {
        check_cuda_errors(cudaMemcpyAsync(output_items[0],
                                          input_items[0],
                                          n * d_itemsize,
                                          cudaMemcpyDefault,
                                          d_stream),
                          "head: cudaMemcpyAsync",
                          d_logger);
    }
    d_ncopied_items += n;

    return n;
}

} /* namespace cuda */
} /* namespace gr */
