/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "nop_impl.h"
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

nop::sptr nop::make(size_t sizeof_stream_item)
{
    return gnuradio::make_block_sptr<nop_impl>(sizeof_stream_item);
}

nop_impl::nop_impl(size_t sizeof_stream_item)
    : gr::sync_block(
          "nop",
          gr::io_signature::make(1, 1, sizeof_stream_item, cuda_buffer::type),
          gr::io_signature::make(1, 1, sizeof_stream_item, cuda_buffer::type))
{
}

int nop_impl::work(int noutput_items,
                   gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items)
{
    gr::cuda::wait_for_work(detail(), d_stream);
    gr::cuda::mark_work_done(detail(), d_stream);
    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
