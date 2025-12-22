/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "copy_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/block_detail.h>

namespace gr {
namespace cuda {

copy::sptr copy::make(size_t itemsize)
{
    return gnuradio::make_block_sptr<copy_impl>(itemsize);
}

/*
 * The private constructor
 */
copy_impl::copy_impl(size_t itemsize)
    : gr::sync_block("copy",
                     gr::io_signature::make(1, 1, itemsize, cuda_buffer::type),
                     gr::io_signature::make(1, 1, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize)
{
    cudaStreamCreate(&d_stream);
}

/*
 * Our virtual destructor.
 */
copy_impl::~copy_impl() 
{
    cudaStreamDestroy(d_stream);
}

int copy_impl::work(int noutput_items,
                    gr_vector_const_void_star& input_items,
                    gr_vector_void_star& output_items)
{

    auto in = static_cast<const uint8_t*>(input_items[0]);
    auto out = static_cast<uint8_t*>(output_items[0]);

    // 1. Wait on inputs
    for (size_t i = 0; i < input_items.size(); i++) {
        auto buf = detail()->input(i)->buffer();
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->wait_device_ready(d_stream);
        }
    }

    cudaMemcpyAsync(out, in, noutput_items * d_itemsize, cudaMemcpyDeviceToDevice, d_stream);

    // 2. Mark outputs
    for (size_t i = 0; i < output_items.size(); i++) {
        auto buf = detail()->output(i);
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->mark_device_ready(d_stream);
        }
    }

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
