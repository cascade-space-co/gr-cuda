/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "seq_strip_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

seq_strip::sptr seq_strip::make(int payload_size)
{
    return gnuradio::make_block_sptr<seq_strip_impl>(payload_size);
}

seq_strip_impl::seq_strip_impl(int payload_size)
    : sync_block("seq_strip",
                 io_signature::make(1, 1, payload_size, cuda_buffer::type),
                 io_signature::make(1, 1, sizeof(uint64_t))),
      d_payload_size(payload_size)
{
    if (d_payload_size < 8 || d_payload_size % 8 != 0)
        throw std::runtime_error(
            "seq_strip: payload_size must be >= 8 and a multiple of 8");
}

int seq_strip_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items)
{
    auto in = static_cast<const uint8_t*>(input_items[0]);
    auto out = static_cast<uint64_t*>(output_items[0]);

    check_cuda_errors(cudaMemcpy2DAsync(out,
                                        sizeof(uint64_t),
                                        in,
                                        d_payload_size,
                                        sizeof(uint64_t),
                                        noutput_items,
                                        cudaMemcpyDeviceToHost,
                                        d_stream),
                      "seq_strip: cudaMemcpy2DAsync",
                      d_logger);

    check_cuda_errors(
        cudaStreamSynchronize(d_stream), "seq_strip: cudaStreamSynchronize", d_logger);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
