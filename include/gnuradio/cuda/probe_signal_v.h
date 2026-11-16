/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_PROBE_SIGNAL_V_H
#define INCLUDED_CUDA_PROBE_SIGNAL_V_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <cstdint>
#include <vector>

namespace gr {
namespace cuda {

/*!
 * \brief Sink that lets the last vector of a GPU stream be read from Python.
 * \ingroup cuda
 *
 * The GPU counterpart of blocks.probe_signal_vX.  Each work() call copies only
 * its last vector to the host, so the stream itself isn't copied off the GPU.
 */
template <class T>
class CUDA_API probe_signal_v : virtual public sync_block
{
public:
    typedef std::shared_ptr<probe_signal_v<T>> sptr;

    /*!
     * \param size Vector length, in samples.
     */
    static sptr make(size_t size);

    /*!
     * \brief The last vector seen, or zeros before the first work() call.
     *
     * Waits for that vector's copy from the GPU to finish.
     */
    virtual std::vector<T> level() const = 0;
};

typedef probe_signal_v<std::uint8_t> probe_signal_vb;
typedef probe_signal_v<std::int16_t> probe_signal_vs;
typedef probe_signal_v<std::int32_t> probe_signal_vi;
typedef probe_signal_v<float> probe_signal_vf;
typedef probe_signal_v<gr_complex> probe_signal_vc;

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_PROBE_SIGNAL_V_H */
