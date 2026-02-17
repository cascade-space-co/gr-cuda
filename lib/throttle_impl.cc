/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "throttle_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/io_signature.h>
#include <thread>
#include <gnuradio/cuda/cuda_error.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gr {
namespace cuda {

namespace {
inline double validate_rate_or_throw(double rate)
{
  if (!std::isfinite(rate) || !(rate > 0.0)) {
    throw std::invalid_argument("cuda::throttle sample_rate must be finite and > 0");
  }
  return rate;
}
} // namespace

throttle::sptr throttle::make(size_t itemsize, double sample_rate) {
  return gnuradio::make_block_sptr<throttle_impl>(itemsize, sample_rate);
}

throttle_impl::throttle_impl(size_t itemsize, double sample_rate)
    : sync_block("throttle",
                 io_signature::make(1, 1, itemsize, cuda_buffer::type),
                 io_signature::make(1, 1, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize),
      d_sample_rate(validate_rate_or_throw(sample_rate)),
      d_total_samples(0) {
}

void throttle_impl::set_sample_rate(double rate) {
  d_sample_rate.store(validate_rate_or_throw(rate), std::memory_order_relaxed);
  // Reset timing to avoid discontinuities when changing the rate mid-run.
  const std::lock_guard<std::mutex> lock(d_mutex);
  d_start_time = std::chrono::steady_clock::now();
  d_total_samples = 0;
}

bool throttle_impl::start() {
  const std::lock_guard<std::mutex> lock(d_mutex);
  d_start_time = std::chrono::steady_clock::now();
  d_total_samples = 0;
  return true;
}

int throttle_impl::work(int noutput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items) {
  auto in = static_cast<const uint8_t *>(input_items[0]);
  auto out = static_cast<uint8_t *>(output_items[0]);

  // 1. Wait for inputs to be ready on GPU
  gr::cuda::wait_for_inputs(detail(), d_stream);

  // 2. Perform copy (throttle is just a pass-through data-wise)
  check_cuda_errors(cudaMemcpyAsync(
      out, in, noutput_items * d_itemsize, cudaMemcpyDeviceToDevice, d_stream));

  // 3. Mark outputs as ready (GPU work is queued)
  gr::cuda::mark_outputs_ready(detail(), d_stream);

  // 4. Throttling Logic (happens on CPU side to delay next scheduler call)
  const double rate = d_sample_rate.load(std::memory_order_relaxed);
  if (rate > 0.0) {
    std::chrono::steady_clock::time_point start_time;
    uint64_t total_samples;
    {
      const std::lock_guard<std::mutex> lock(d_mutex);
      d_total_samples += static_cast<uint64_t>(noutput_items);
      start_time = d_start_time;
      total_samples = d_total_samples;
    }

    const double expected_s = static_cast<double>(total_samples) / rate;
    const auto target_time = start_time + std::chrono::duration<double>(expected_s);
    std::this_thread::sleep_until(target_time);
  }

  return noutput_items;
}

} // namespace cuda
} // namespace gr


