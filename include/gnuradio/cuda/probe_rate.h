/* -*- c++ -*- */
/*
 * Copyright 2025 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_PROBE_RATE_H
#define INCLUDED_CUDA_PROBE_RATE_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <string_view>

namespace gr {
namespace cuda {

/*!
 * \brief GPU-native throughput measurement block
 * \ingroup cuda
 *
 * Measures throughput of GPU data streams without D2H transfer overhead.
 * Operates directly on GPU buffers and reports sample rate via message port.
 */
class CUDA_API probe_rate : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<probe_rate> sptr;

    /*!
     * \brief Make a GPU throughput measurement block
     * \param itemsize size of each stream item
     * \param update_rate_ms minimum update time in milliseconds
     * \param alpha gain for running average filter
     * \param name name for this probe (used in generated dictionaries)
     *
     * If the name is empty, the "name" field in the emitted dictionaries is
     * omitted.
     */
    static sptr make(size_t itemsize,
                     double update_rate_ms = 500.0,
                     double alpha = 0.0001,
                     std::string_view name = "");

    /*!
     * \brief Set the decay of the exponential rate average
     */
    virtual void set_alpha(double alpha) = 0;

    /*!
     * \brief Set the name of this probe
     * Used in the emitted dictionaries if not empty
     */
    virtual void set_name(std::string_view name) = 0;

    virtual double rate() = 0;

    bool start() override = 0;
    bool stop() override = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_PROBE_RATE_H */
