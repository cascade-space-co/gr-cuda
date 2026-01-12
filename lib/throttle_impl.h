/* -*- c++ -*- */
/*
 * Copyright 2026
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_THROTTLE_IMPL_H
#define INCLUDED_CUDA_THROTTLE_IMPL_H

#include <gnuradio/cuda/throttle.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace gr {
namespace cuda {

class throttle_impl : public throttle {
private:
  size_t d_itemsize;
  std::atomic<double> d_sample_rate;
  cudaStream_t d_stream;

  mutable std::mutex d_mutex;
  std::chrono::steady_clock::time_point d_start_time;
  uint64_t d_total_samples;

public:
  throttle_impl(size_t itemsize, double sample_rate);
  ~throttle_impl() override;

  void set_sample_rate(double rate) override;

  bool start() override;
  
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_THROTTLE_IMPL_H */


