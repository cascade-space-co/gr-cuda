/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "seq_stamp_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

seq_stamp::sptr seq_stamp::make(int payload_size)
{
    return gnuradio::make_block_sptr<seq_stamp_impl>(payload_size);
}

seq_stamp_impl::seq_stamp_impl(int payload_size)
    : sync_block("seq_stamp",
                 io_signature::make(1, 1, payload_size, cuda_buffer::type),
                 io_signature::make(1, 1, payload_size, cuda_buffer::type)),
      d_payload_size(payload_size),
      d_counters(MAX_BATCH)
{
    if (d_payload_size < 8 || d_payload_size % 8 != 0)
        throw std::runtime_error(
            "seq_stamp: payload_size must be >= 8 and a multiple of 8");
}

int seq_stamp_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items)
{
    int n = std::min(noutput_items, MAX_BATCH);

    auto out = static_cast<uint8_t*>(output_items[0]);

    for (int i = 0; i < n; i++)
        d_counters[i] = d_counter + static_cast<uint64_t>(i);

    check_cuda_errors(cudaMemcpy2DAsync(out,
                                        d_payload_size,
                                        d_counters.data(),
                                        sizeof(uint64_t),
                                        sizeof(uint64_t),
                                        n,
                                        cudaMemcpyHostToDevice,
                                        d_stream),
                      "seq_stamp: cudaMemcpy2DAsync",
                      d_logger);

    check_cuda_errors(
        cudaStreamSynchronize(d_stream), "seq_stamp: cudaStreamSynchronize", d_logger);

    d_counter += static_cast<uint64_t>(n);

    return n;
}

} /* namespace cuda */
} /* namespace gr */
