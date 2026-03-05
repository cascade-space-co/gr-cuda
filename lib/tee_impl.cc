/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tee_impl.h"
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <cstring>

namespace gr {
namespace cuda {

tee::sptr tee::make(size_t itemsize, bool gpu)
{
    return gnuradio::make_block_sptr<tee_impl>(itemsize, gpu);
}

tee_impl::tee_impl(size_t itemsize, bool gpu)
    : gr::sync_block(
          "tee",
          gr::io_signature::make(1,
                                 1,
                                 itemsize,
                                 gpu ? cuda_buffer::type
                                     : gr::io_signature::default_buftype::type),
          gr::io_signature::make(2,
                                 2,
                                 itemsize,
                                 gpu ? cuda_buffer::type
                                     : gr::io_signature::default_buftype::type)),
      d_itemsize(itemsize),
      d_gpu(gpu)
{
}

int tee_impl::work(int noutput_items,
                   gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items)
{
    const size_t nbytes = noutput_items * d_itemsize;

    if (d_gpu) {
        gr::cuda::wait_for_work(detail(), d_stream);

        auto in = static_cast<const uint8_t*>(input_items[0]);
        auto out0 = static_cast<uint8_t*>(output_items[0]);
        auto out1 = static_cast<uint8_t*>(output_items[1]);

        check_cuda_errors(
            cudaMemcpyAsync(out0, in, nbytes, cudaMemcpyDeviceToDevice, d_stream),
            "tee: D2D copy to port 0",
            d_logger);
        check_cuda_errors(
            cudaMemcpyAsync(out1, in, nbytes, cudaMemcpyDeviceToDevice, d_stream),
            "tee: D2D copy to port 1",
            d_logger);

        gr::cuda::mark_work_done(detail(), d_stream);
    } else {
        auto in = static_cast<const uint8_t*>(input_items[0]);
        auto out0 = static_cast<uint8_t*>(output_items[0]);
        auto out1 = static_cast<uint8_t*>(output_items[1]);

        std::memcpy(out0, in, nbytes);
        std::memcpy(out1, in, nbytes);
    }

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
