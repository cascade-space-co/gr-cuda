/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_FFT_IMPL_H
#define INCLUDED_CUDA_FFT_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/fft.h>
#include <cufft.h>
#include <unordered_map>
#include <mutex>

namespace gr {
namespace cuda {

class fft_impl : public fft, public cuda_block
{
public:
    fft_impl(size_t fft_size,
             bool forward,
             const std::vector<float>& window,
             bool shift,
             bool real_input);
    ~fft_impl() override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

private:
    /*! Return a cuFFT plan for the given batch count, creating one if needed. */
    cufftHandle get_plan(int batch);

    /*! (Re-)allocate the GPU staging buffer so it can hold at least \p total_items. */
    void ensure_work_buffers(size_t total_items);

    const size_t d_fft_size;
    const bool d_forward;
    const bool d_shift;
    const bool d_has_window;
    const bool d_real_input;

    float* d_window_dev;
    size_t d_window_size;

    cufftComplex* d_work_dev;
    size_t d_work_items;

    std::unordered_map<int, cufftHandle> d_plan_cache;
    std::mutex d_plan_mutex;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_FFT_IMPL_H */
