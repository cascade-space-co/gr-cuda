/* -*- c++ -*- */
/*
 * Copyright 2025
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "null_source_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

null_source::sptr null_source::make(size_t sizeof_stream_item, size_t num_outputs)
{
    return gnuradio::make_block_sptr<null_source_impl>(sizeof_stream_item, num_outputs);
}

null_source_impl::null_source_impl(size_t sizeof_stream_item, size_t num_outputs)
    : sync_block("null_source",
                 io_signature::make(0, 0, 0),
                 io_signature::make(num_outputs, num_outputs, sizeof_stream_item, cuda_buffer::type)),
      d_itemsize(sizeof_stream_item),
      d_num_outputs(num_outputs)
{
    cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking);
}

null_source_impl::~null_source_impl()
{
    cudaStreamDestroy(d_stream);
}

int null_source_impl::work(int noutput_items,
                          gr_vector_const_void_star& input_items,
                          gr_vector_void_star& output_items)
{
    auto out = static_cast<uint8_t*>(output_items[0]);

    // Fill GPU buffer with zeros (mimics GNU Radio's memset behavior)
    cudaMemsetAsync(out, 0, noutput_items * d_itemsize, d_stream);

    // Mark outputs ready
    gr::cuda::mark_outputs_ready(detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */


