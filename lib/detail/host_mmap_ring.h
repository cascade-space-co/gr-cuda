/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_CUDA_DETAIL_HOST_MMAP_RING_H
#define INCLUDED_GR_CUDA_DETAIL_HOST_MMAP_RING_H

#include <cuda_runtime_api.h>
#include <memory>

namespace gr {
class logger;
namespace detail {

/*!
 * \brief Owns a host mmap double-mapped 2N virtual region.
 *
 * Allocates RAM-backed pages via memfd + mmap, maps the same pages twice into
 * adjacent virtual ranges, and optionally pins the full 2N range for async
 * CUDA DMA.
 */
class host_mmap_ring
{
public:
    ~host_mmap_ring();

    host_mmap_ring(const host_mmap_ring&) = delete;
    host_mmap_ring& operator=(const host_mmap_ring&) = delete;
    host_mmap_ring(host_mmap_ring&&) = delete;
    host_mmap_ring& operator=(host_mmap_ring&&) = delete;

    static std::unique_ptr<host_mmap_ring> create(
        size_t requested_bytes,
        const std::shared_ptr<gr::logger>& logger = nullptr);

    void register_pinned();
    char* base_ptr();

private:
    host_mmap_ring() = default;
    void reset();

    std::shared_ptr<gr::logger> d_logger;
    void* d_base = nullptr; // start of the 2N virtual address region
    size_t d_bytes = 0;     // N (one half), page-aligned
    bool d_registered = false;
};

} // namespace detail
} // namespace gr

#endif /* INCLUDED_GR_CUDA_DETAIL_HOST_MMAP_RING_H */
