/* -*- c++ -*- */
/*
 * Copyright 2025
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "probe_rate_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/io_signature.h>
#include <cmath>

namespace gr {
namespace cuda {

probe_rate::sptr probe_rate::make(size_t itemsize, double update_rate_ms,
                                  double alpha, std::string_view name) {
  return gnuradio::make_block_sptr<probe_rate_impl>(itemsize, update_rate_ms,
                                                     alpha, name);
}

probe_rate_impl::probe_rate_impl(size_t itemsize, double update_rate_ms,
                                 double alpha, std::string_view name)
    : sync_block("probe_rate",
                 io_signature::make(1, 1, itemsize, cuda_buffer::type),
                 io_signature::make(0, 0, 0)),
      d_itemsize(itemsize),
      d_alpha(alpha),
      d_beta(1.0 - alpha),
      d_avg(0),
      d_min_update_time(update_rate_ms),
      d_last_update(std::chrono::steady_clock::now()),
      d_lastthru(0),
      d_port(pmt::mp("rate")),
      d_dict_avg(pmt::mp("rate_avg")),
      d_dict_now(pmt::mp("rate_now")) {
  cudaStreamCreate(&d_stream);
  message_port_register_out(d_port);
  set_name(name);
}

void probe_rate_impl::set_name(std::string_view name) {
  if (name.empty()) {
    d_data_dict.erase(pmt::mp("name"));
  } else {
    d_data_dict[pmt::mp("name")] = pmt::mp(std::string(name));
  }
}

probe_rate_impl::~probe_rate_impl() { cudaStreamDestroy(d_stream); }

int probe_rate_impl::work(int noutput_items,
                          gr_vector_const_void_star &input_items,
                          gr_vector_void_star &output_items) {
  // Wait for GPU data to be ready (no D2H transfer!)
  gr::cuda::wait_for_inputs(detail(), d_stream);

  // Count throughput (data stays on GPU)
  d_lastthru += noutput_items;
  auto now = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> diff = now - d_last_update;
  double diff_ms = diff.count();
  
  // Only update if enough time has passed AND we have valid timing
  if (diff_ms >= d_min_update_time && diff_ms > 0) {
    double rate_this_update = d_lastthru * 1e3 / diff_ms;
    d_lastthru = 0;
    d_last_update = now;
    
    // Sanity check: ignore unrealistic rates
    if (std::isfinite(rate_this_update) && rate_this_update >= 0) {
      if (d_avg == 0) {
        d_avg = rate_this_update;
      } else {
        // Compute exponential average with overflow protection
        double new_avg = rate_this_update * d_alpha + d_avg * d_beta;
        // If the new average is not finite or grew too large, reset
        if (std::isfinite(new_avg) && new_avg < 1e15) {
          d_avg = new_avg;
        } else {
          // Reset to current rate on overflow
          d_avg = rate_this_update;
        }
      }
      d_data_dict[d_dict_avg] = pmt::mp(d_avg);
      d_data_dict[d_dict_now] = pmt::mp(rate_this_update);
      message_port_pub(d_port, pmt::dict_from_mapping(d_data_dict));
    }
  }
  return noutput_items;
}

void probe_rate_impl::setup_rpc() {
#ifdef GR_CTRLPORT
  add_rpc_variable(rpcbasic_sptr(
      new rpcbasic_register_get<probe_rate_impl, double>(
          alias(), "rate_items", &probe_rate_impl::rate, pmt::mp(0),
          pmt::mp(1e6), pmt::mp(1), "items/sec", "Item rate", RPC_PRIVLVL_MIN,
          DISPTIME | DISPOPTSTRIP)));
  add_rpc_variable(rpcbasic_sptr(new rpcbasic_register_get<probe_rate_impl, double>(
      alias(), "timesincelast", &probe_rate_impl::timesincelast, pmt::mp(0),
      pmt::mp(d_min_update_time * 2), pmt::mp(0), "ms",
      "Time since last update", RPC_PRIVLVL_MIN, DISPTIME | DISPOPTSTRIP)));
#endif
}

void probe_rate_impl::set_alpha(double alpha) {
  d_alpha = alpha;
  d_beta = 1.0 - alpha;
}

double probe_rate_impl::rate() { return d_avg; }

double probe_rate_impl::timesincelast() {
  auto now = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> diff = now - d_last_update;
  return diff.count();
}

bool probe_rate_impl::start() {
  d_avg = 0;
  d_lastthru = 0;
  d_last_update = std::chrono::steady_clock::now();
  return true;
}

bool probe_rate_impl::stop() { return true; }

} /* namespace cuda */
} /* namespace gr */

