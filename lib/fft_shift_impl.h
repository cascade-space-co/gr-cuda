/* -*- c++ -*- */
/*
 * Copyright 2026
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_FFT_SHIFT_IMPL_H
#define INCLUDED_CUDA_FFT_SHIFT_IMPL_H

#include <gnuradio/cuda/fft_shift.h>
#include <cuda_runtime_api.h>

namespace gr {
namespace cuda {

class fft_shift_impl : public fft_shift
{
public:
    explicit fft_shift_impl(size_t fft_size);
    ~fft_shift_impl() override;

    // Apply fftshift on each vector in the CUDA buffer.
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

private:
    const size_t d_fft_size;
    cudaStream_t d_stream;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_FFT_SHIFT_IMPL_H */
