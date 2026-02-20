/* -*- c++ -*- */
/*
 * Copyright 2025 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "null_sink_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

null_sink::sptr null_sink::make(size_t sizeof_stream_item, size_t num_inputs)
{
    return gnuradio::make_block_sptr<null_sink_impl>(sizeof_stream_item, num_inputs);
}

null_sink_impl::null_sink_impl(size_t sizeof_stream_item, size_t num_inputs)
    : sync_block("null_sink",
                 io_signature::make(num_inputs, num_inputs, sizeof_stream_item, cuda_buffer::type),
                 io_signature::make(0, 0, 0)),
      d_itemsize(sizeof_stream_item),
      d_num_inputs(num_inputs)
{
}

int null_sink_impl::work(int noutput_items,
                        gr_vector_const_void_star& input_items,
                        gr_vector_void_star& output_items)
{
    // Wait for inputs to be ready
    gr::cuda::wait_for_work(detail(), d_stream);

    // Do nothing - just consume the data (mimics GNU Radio's null_sink)
    // Signal inputs consumed so upstream producers can safely overwrite
    gr::cuda::mark_work_done(detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */


