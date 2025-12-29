/* -*- c++ -*- */
/*
 * Copyright 2025
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_NULL_SINK_H
#define INCLUDED_CUDA_NULL_SINK_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief GPU-native null sink that consumes data directly on the GPU
 * \ingroup cuda
 *
 * Consumes data directly from GPU device memory without any D2H transfer.
 * Useful for benchmarking GPU blocks in isolation.
 */
class CUDA_API null_sink : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<null_sink> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of cuda::null_sink.
   *
   * To avoid accidental use of raw pointers, cuda::null_sink's
   * constructor is in a private implementation
   * class. cuda::null_sink::make is the public interface for
   * creating new instances.
   */
  static sptr make(size_t sizeof_stream_item);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_NULL_SINK_H */