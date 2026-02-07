/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_COPY_H
#define INCLUDED_CUDA_COPY_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief <+description of block+>
 * \ingroup cuda
 *
 */
class CUDA_API copy : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<copy> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of cuda::copy.
   *
   * To avoid accidental use of raw pointers, cuda::copy's
   * constructor is in a private implementation
   * class. cuda::copy::make is the public interface for
   * creating new instances.
   *
   * \param itemsize Size of a stream item in bytes
   * \param passthrough If true, skip memcpy and just pass through (for benchmarking)
   */
  static sptr make(size_t itemsize, bool passthrough = false);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_COPY_H */
