/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_FFT_H
#define INCLUDED_CUDA_FFT_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/gr_complex.h>
#include <vector>

namespace gr {
namespace cuda {

/*!
 * \brief CUDA FFT block (cuFFT)
 * \ingroup cuda
 *
 * Performs FFT or IFFT on vectors of length fft_size.
 * Input can be complex64 or real float32 (float-to-complex conversion).
 * Optional windowing (real-valued) and FFT shift are supported.
 */
class CUDA_API fft : virtual public sync_block
{
public:
    typedef std::shared_ptr<fft> sptr;

    /*!
     * \brief Create an instance of cuda::fft
     * \param fft_size size of FFT (vector length)
     * \param forward true for forward FFT, false for inverse FFT
     * \param window real-valued window coefficients (size fft_size) or empty
     * \param shift true to apply fftshift
     * \param real_input true for float32 input (float-to-complex FFT)
     */
    static sptr make(size_t fft_size,
                     bool forward = true,
                     const std::vector<float>& window = {},
                     bool shift = false,
                     bool real_input = false);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_FFT_H */
