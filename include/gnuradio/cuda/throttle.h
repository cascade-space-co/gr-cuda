/* -*- c++ -*- */
/*
 * Copyright 2026
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_THROTTLE_H
#define INCLUDED_CUDA_THROTTLE_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Throttle flow of samples on GPU
 * \ingroup cuda
 *
 * Limits the throughput of the flowgraph to the specified sample rate.
 * Useful for simulations or when connecting to hardware sinks.
 *
 * Note: This throttles the CPU submission thread, which indirectly
 * throttles the GPU execution.
 */
class CUDA_API throttle : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<throttle> sptr;

  /*!
   * \brief Make a GPU throttle block
   * \param itemsize size of each stream item
   * \param sample_rate Sample rate in items per second
   */
  static sptr make(size_t itemsize, double sample_rate);

  /*!
   * \brief Set the sample rate
   */
  virtual void set_sample_rate(double rate) = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_THROTTLE_H */


