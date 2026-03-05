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

#include "device_vmm_ring.h"

#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/logger.h>

#include <cassert>

namespace gr {
namespace detail {

size_t query_vmm_granularity_for_current_device()
{
    int device;
    check_cuda_errors(cudaGetDevice(&device), "vmm: cudaGetDevice");

    CUdevice cu_dev;
    check_cuda_errors(cuDeviceGet(&cu_dev, device), "vmm: cuDeviceGet");

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;

    size_t granularity = 0;
    check_cuda_errors(cuMemGetAllocationGranularity(
                          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "vmm: cuMemGetAllocationGranularity");

    return granularity;
}

device_vmm_ring::~device_vmm_ring() { this->reset(); }

std::unique_ptr<device_vmm_ring>
device_vmm_ring::create(size_t requested_bytes,
                        const std::shared_ptr<gr::logger>& logger)
{
    // Create a 2N virtual range where both halves map to one physical allocation:
    // [ ---- half 1 ---- | ---- half 2 ---- ]
    //   ^                   ^
    //   ptr                 ptr + N
    //   \__ same physical __/
    //       allocation
    //
    // Both halves alias one physical allocation. This makes wrap-around
    // accesses look contiguous up to N bytes.

    auto ring = std::unique_ptr<device_vmm_ring>(new device_vmm_ring());
    ring->d_logger = logger;

    int device;
    check_cuda_errors(cudaGetDevice(&device), "vmm_create: cudaGetDevice", logger);

    CUdevice cu_dev;
    check_cuda_errors(cuDeviceGet(&cu_dev, device), "vmm_create: cuDeviceGet", logger);

    // Pinned, device-local, matches what vmm_create_double_mapped will use.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;

    // Caller (cuda_buffer::allocate_buffer) already rounds up to VMM
    // granularity, so requested_bytes should be aligned.
    size_t granularity = query_vmm_granularity_for_current_device();
    logger->debug("device_vmm_ring: requesting {} bytes (granularity={})",
                  requested_bytes,
                  granularity);
    assert(requested_bytes % granularity == 0);

    ring->d_aligned_bytes = requested_bytes;

    // Reserve 2N contiguous VA (no physical memory yet).
    check_cuda_errors(
        cuMemAddressReserve(&ring->d_ptr, 2 * ring->d_aligned_bytes, 0, 0, 0),
        "vmm_create: cuMemAddressReserve",
        logger);

    // Create one physical allocation of size N.
    check_cuda_errors(cuMemCreate(&ring->d_handle, ring->d_aligned_bytes, &prop, 0),
                      "vmm_create: cuMemCreate",
                      logger);

    // Map the physical allocation into the first half [ptr, ptr+N).
    check_cuda_errors(
        cuMemMap(ring->d_ptr, ring->d_aligned_bytes, 0, ring->d_handle, 0),
        "vmm_create: cuMemMap first half",
        logger);

    // Map the same allocation into the second half [ptr+N, ptr+2N).
    check_cuda_errors(cuMemMap(ring->d_ptr + ring->d_aligned_bytes,
                               ring->d_aligned_bytes,
                               0,
                               ring->d_handle,
                               0),
                      "vmm_create: cuMemMap second half",
                      logger);

    // Grant read/write access across the full 2N range.
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    check_cuda_errors(
        cuMemSetAccess(ring->d_ptr, 2 * ring->d_aligned_bytes, &access, 1),
        "vmm_create: cuMemSetAccess",
        logger);

    logger->debug("device_vmm_ring: mapped 2x{} bytes at VA {:#x}",
                  ring->d_aligned_bytes,
                  (uintptr_t)ring->d_ptr);

    return ring;
}

char* device_vmm_ring::data()
{
    return reinterpret_cast<char*>(static_cast<uintptr_t>(d_ptr));
}

void device_vmm_ring::reset()
{
    if (!d_ptr)
        return;

    if (d_handle) {
        cuMemUnmap(d_ptr, d_aligned_bytes);
        cuMemUnmap(d_ptr + d_aligned_bytes, d_aligned_bytes);
        cuMemRelease(d_handle);
        d_handle = 0;
    }
    cuMemAddressFree(d_ptr, 2 * d_aligned_bytes);

    d_ptr = 0;
    d_aligned_bytes = 0;
}

} // namespace detail
} // namespace gr
