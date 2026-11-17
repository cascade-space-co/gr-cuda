/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_CUDA_DETAIL_LAST_ITEM_PROBE_H
#define INCLUDED_GR_CUDA_DETAIL_LAST_ITEM_PROBE_H

#include <gnuradio/logger.h>
#include <gnuradio/thread/thread.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>

namespace gr {
namespace detail {

/*!
 * \brief Keeps a host copy of the last item a probe block saw on the GPU.
 *
 * capture() runs in work(): it queues a copy of one item from GPU memory into
 * a pinned host buffer on the block's stream and records an event, without
 * waiting.  On zero-copy edges the item is already in host memory, and CUDA
 * does that host-to-host copy synchronously; it is a single item.
 *
 * read() runs on the caller's thread: it waits for the latest copy to land
 * and copies the item out.  Until the first capture() the item reads as
 * zeros.  While read() waits, capture() can't queue the next copy, so the
 * probe's work() falls behind briefly; upstream blocks keep queuing GPU work
 * meanwhile, and polling level() even in a tight loop didn't change the data
 * rate.
 */
class last_item_probe
{
public:
    last_item_probe(size_t item_bytes, const std::shared_ptr<gr::logger>& logger);
    ~last_item_probe();

    last_item_probe(const last_item_probe&) = delete;
    last_item_probe& operator=(const last_item_probe&) = delete;

    void capture(const void* item, cudaStream_t stream);
    void read(void* dst) const;

private:
    void release();

    const size_t d_item_bytes;
    std::shared_ptr<gr::logger> d_logger;
    void* d_staging = nullptr;
    cudaEvent_t d_copied = nullptr;
    bool d_captured = false;
    // Held across capture()'s enqueue and read()'s wait-and-copy, so a new
    // copy can't be queued into d_staging while read() is copying it out.
    mutable gr::thread::mutex d_mutex;
};

} // namespace detail
} // namespace gr

#endif /* INCLUDED_GR_CUDA_DETAIL_LAST_ITEM_PROBE_H */
