/* -*- c++ -*- */
/*
 * Copyright 2025 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_NULL_SOURCE_H
#define INCLUDED_CUDA_NULL_SOURCE_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief GPU-native null source that generates zeros directly on the GPU
 * \ingroup cuda
 *
 * Generates zeros directly in GPU device memory without any H2D transfer.
 * Useful for benchmarking GPU blocks in isolation.
 */
class CUDA_API null_source : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<null_source> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of cuda::null_source.
   *
   * To avoid accidental use of raw pointers, cuda::null_source's
   * constructor is in a private implementation
   * class. cuda::null_source::make is the public interface for
   * creating new instances.
   *
   * \param sizeof_stream_item Size of a stream item in bytes
   * \param num_outputs Number of output ports (default: 1)
   */
  static sptr make(size_t sizeof_stream_item, size_t num_outputs = 1);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_NULL_SOURCE_H */