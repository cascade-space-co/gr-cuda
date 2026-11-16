/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "last_item_probe.h"
#include <gnuradio/cuda/cuda_error.h>
#include <cstring>

namespace gr {
namespace detail {

last_item_probe::last_item_probe(size_t item_bytes,
                                 const std::shared_ptr<gr::logger>& logger)
    : d_item_bytes(item_bytes), d_logger(logger)
{
    try {
        check_cuda_errors(cudaMallocHost(&d_staging, d_item_bytes),
                          "last_item_probe: cudaMallocHost",
                          d_logger);
        std::memset(d_staging, 0, d_item_bytes);
        check_cuda_errors(cudaEventCreateWithFlags(&d_copied, cudaEventDisableTiming),
                          "last_item_probe: cudaEventCreateWithFlags",
                          d_logger);
    } catch (...) {
        release();
        throw;
    }
}

last_item_probe::~last_item_probe() { release(); }

void last_item_probe::release()
{
    if (d_copied) {
        // A copy into d_staging may still be queued.
        cudaEventSynchronize(d_copied);
        cudaEventDestroy(d_copied);
        d_copied = nullptr;
    }
    if (d_staging) {
        cudaFreeHost(d_staging);
        d_staging = nullptr;
    }
}

void last_item_probe::capture(const void* item, cudaStream_t stream)
{
    gr::thread::scoped_lock l(d_mutex);
    check_cuda_errors(
        cudaMemcpyAsync(d_staging, item, d_item_bytes, cudaMemcpyDefault, stream),
        "last_item_probe: cudaMemcpyAsync",
        d_logger);
    check_cuda_errors(cudaEventRecord(d_copied, stream),
                      "last_item_probe: cudaEventRecord",
                      d_logger);
    d_captured = true;
}

void last_item_probe::read(void* dst) const
{
    gr::thread::scoped_lock l(d_mutex);
    if (d_captured) {
        check_cuda_errors(cudaEventSynchronize(d_copied),
                          "last_item_probe: cudaEventSynchronize",
                          d_logger);
    }
    std::memcpy(dst, d_staging, d_item_bytes);
}

} // namespace detail
} // namespace gr
