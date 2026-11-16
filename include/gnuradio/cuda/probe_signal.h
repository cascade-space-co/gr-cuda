/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_PROBE_SIGNAL_H
#define INCLUDED_CUDA_PROBE_SIGNAL_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <cstdint>

namespace gr {
namespace cuda {

/*!
 * \brief Sink that lets the last sample of a GPU stream be read from Python.
 * \ingroup cuda
 *
 * The GPU counterpart of blocks.probe_signal_X.  Each work() call copies only
 * its last sample to the host, so the stream itself isn't copied off the GPU.
 */
template <class T>
class CUDA_API probe_signal : virtual public sync_block
{
public:
    typedef std::shared_ptr<probe_signal<T>> sptr;

    static sptr make();

    /*!
     * \brief The last sample seen, or 0 before the first work() call.
     *
     * Waits for that sample's copy from the GPU to finish.
     */
    virtual T level() const = 0;
};

typedef probe_signal<std::uint8_t> probe_signal_b;
typedef probe_signal<std::int16_t> probe_signal_s;
typedef probe_signal<std::int32_t> probe_signal_i;
typedef probe_signal<float> probe_signal_f;
typedef probe_signal<gr_complex> probe_signal_c;

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_PROBE_SIGNAL_H */
