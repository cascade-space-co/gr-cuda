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
                 io_signature::make(1, 1, payload_size + SEQ_HDR, cuda_buffer::type),
                 // Port 0: the recovered uint64 sequence number (required).
                 // Port 1: the original payload, header removed (optional).
                 io_signature::make2(1,
                                     2,
                                     sizeof(uint64_t),
                                     payload_size,
                                     cuda_buffer::type,
                                     cuda_buffer::type)),
      d_payload_size(payload_size)
{
    if (d_payload_size < 1)
        throw std::runtime_error("seq_strip: payload_size must be >= 1");
}

int seq_strip_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items)
{
    auto in = static_cast<const uint8_t*>(input_items[0]);
    const int in_pitch = d_payload_size + SEQ_HDR;

    // Port 0: extract the 8-byte sequence header from the front of each input
    // item into a packed uint64 output buffer.
    auto seq_out = static_cast<uint64_t*>(output_items[0]);
    check_cuda_errors(cudaMemcpy2DAsync(seq_out,
                                        sizeof(uint64_t),
                                        in,
                                        in_pitch,
                                        sizeof(uint64_t),
                                        noutput_items,
                                        cudaMemcpyDeviceToDevice,
                                        d_stream),
                      "seq_strip: cudaMemcpy2DAsync seq",
                      d_logger);

    // Port 1 (optional): pass the original payload through, with the 8-byte
    // sequence header removed.  Only copied when the port is connected.
    if (output_items.size() > 1) {
        auto payload_out = static_cast<uint8_t*>(output_items[1]);
        check_cuda_errors(cudaMemcpy2DAsync(payload_out,
                                            d_payload_size,
                                            in + SEQ_HDR,
                                            in_pitch,
                                            d_payload_size,
                                            noutput_items,
                                            cudaMemcpyDeviceToDevice,
                                            d_stream),
                          "seq_strip: cudaMemcpy2DAsync payload",
                          d_logger);
    }

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
