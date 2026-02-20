/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_shift_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_error.h>
#include "fft.cuh"

namespace gr {
namespace cuda {

fft_shift::sptr fft_shift::make(size_t fft_size)
{
    return gnuradio::make_block_sptr<fft_shift_impl>(fft_size);
}

fft_shift_impl::fft_shift_impl(size_t fft_size)
    : gr::sync_block("fft_shift",
                     io_signature::make(1, 1, sizeof(gr_complex) * fft_size, cuda_buffer::type),
                     io_signature::make(1, 1, sizeof(gr_complex) * fft_size, cuda_buffer::type)),
      d_fft_size(fft_size)
{
    if (d_fft_size == 0) {
        throw std::invalid_argument("fft_size must be > 0");
    }
}

int fft_shift_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items)
{
    // Ensure upstream GPU work is complete before reading inputs.
    gr::cuda::wait_for_work(detail(), d_stream);

    auto in = static_cast<const cufftComplex*>(input_items[0]);
    auto out = static_cast<cufftComplex*>(output_items[0]);
    const size_t total_items = static_cast<size_t>(noutput_items) * d_fft_size;

    exec_kernel_fftshift(in, out, total_items, d_fft_size, d_stream);

    // Notify downstream CUDA buffers that output is ready.
    gr::cuda::mark_work_done(detail(), d_stream);
    return noutput_items;
}

} // namespace cuda
} // namespace gr
