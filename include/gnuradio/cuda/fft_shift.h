/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_FFT_SHIFT_H
#define INCLUDED_CUDA_FFT_SHIFT_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/gr_complex.h>

namespace gr {
namespace cuda {

/*!
 * \brief CUDA FFT shift block
 * \ingroup cuda
 *
 * Applies fftshift on vectors of complex64 of length fft_size.
 */
class CUDA_API fft_shift : virtual public sync_block
{
public:
    typedef std::shared_ptr<fft_shift> sptr;

    /*!
     * \brief Create an instance of cuda::fft_shift
     * \param fft_size size of FFT (vector length)
     */
    static sptr make(size_t fft_size);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_FFT_SHIFT_H */
