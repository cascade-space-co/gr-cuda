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

#ifndef INCLUDED_GR_CUDA_DETAIL_DEVICE_VMM_RING_H
#define INCLUDED_GR_CUDA_DETAIL_DEVICE_VMM_RING_H

#include <cuda.h>
#include <memory>

namespace gr {
namespace detail {

/*!
 * \brief Query VMM minimum allocation granularity on current CUDA device.
 */
size_t query_vmm_granularity_for_current_device();

/*!
 * \brief Owns a CUDA VMM double-mapped 2N virtual region.
 *
 * Allocates one physical GPU allocation of N bytes and maps it twice into a
 * contiguous 2N virtual range using CUDA VMM. This makes wrap-around accesses
 * contiguous from the caller's perspective.
 */
class device_vmm_ring
{
public:
    ~device_vmm_ring();

    device_vmm_ring(const device_vmm_ring&) = delete;
    device_vmm_ring& operator=(const device_vmm_ring&) = delete;
    device_vmm_ring(device_vmm_ring&&) = delete;
    device_vmm_ring& operator=(device_vmm_ring&&) = delete;

    static std::unique_ptr<device_vmm_ring> create(size_t requested_bytes);

    char* data();

private:
    device_vmm_ring() = default;
    void reset();

    CUdeviceptr d_ptr = 0;
    CUmemGenericAllocationHandle d_handle = 0;
    size_t d_aligned_bytes = 0; // N (one half), aligned to VMM granularity
};

} // namespace detail
} // namespace gr

#endif /* INCLUDED_GR_CUDA_DETAIL_DEVICE_VMM_RING_H */
