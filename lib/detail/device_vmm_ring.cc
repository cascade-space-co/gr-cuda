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

#include <stdexcept>

namespace gr {
namespace detail {

size_t query_vmm_granularity_for_current_device()
{
    int device;
    check_cuda_errors(cudaGetDevice(&device), "vmm: cudaGetDevice");

    CUdevice cu_dev;
    CUresult res = cuDeviceGet(&cu_dev, device);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm: cuDeviceGet failed");

    // Pinned, device-local, matches what vmm_create_double_mapped will use.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;

    size_t granularity = 0;
    res = cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm: cuMemGetAllocationGranularity failed");

    return granularity;
}

device_vmm_ring::~device_vmm_ring() { this->reset(); }

std::unique_ptr<device_vmm_ring> device_vmm_ring::create(size_t requested_bytes)
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

    int device;
    check_cuda_errors(cudaGetDevice(&device), "vmm_create: cudaGetDevice");

    CUdevice cu_dev;
    CUresult res = cuDeviceGet(&cu_dev, device);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm_create: cuDeviceGet failed");

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cu_dev;

    size_t granularity = query_vmm_granularity_for_current_device();

    // Round up to granularity, VMM requires all sizes/offsets to be multiples.
    ring->d_aligned_bytes =
        ((requested_bytes + granularity - 1) / granularity) * granularity;

    // Reserve 2N contiguous VA (no physical memory yet).
    res = cuMemAddressReserve(&ring->d_ptr, 2 * ring->d_aligned_bytes, 0, 0, 0);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm_create: cuMemAddressReserve failed");

    // Create one physical allocation of size N.
    res = cuMemCreate(&ring->d_handle, ring->d_aligned_bytes, &prop, 0);
    if (res != CUDA_SUCCESS) {
        ring->reset();
        throw std::runtime_error("vmm_create: cuMemCreate failed");
    }

    // Map the physical allocation into the first half [ptr, ptr+N).
    res = cuMemMap(ring->d_ptr, ring->d_aligned_bytes, 0, ring->d_handle, 0);
    if (res != CUDA_SUCCESS) {
        ring->reset();
        throw std::runtime_error("vmm_create: cuMemMap first half failed");
    }

    // Map the same allocation into the second half [ptr+N, ptr+2N).
    res = cuMemMap(
        ring->d_ptr + ring->d_aligned_bytes, ring->d_aligned_bytes, 0, ring->d_handle, 0);
    if (res != CUDA_SUCCESS) {
        ring->reset();
        throw std::runtime_error("vmm_create: cuMemMap second half failed");
    }

    // Grant read/write access across the full 2N range.
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    res = cuMemSetAccess(ring->d_ptr, 2 * ring->d_aligned_bytes, &access, 1);
    if (res != CUDA_SUCCESS) {
        ring->reset();
        throw std::runtime_error("vmm_create: cuMemSetAccess failed");
    }

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

    cuMemUnmap(d_ptr, d_aligned_bytes);
    cuMemUnmap(d_ptr + d_aligned_bytes, d_aligned_bytes);
    cuMemRelease(d_handle);
    cuMemAddressFree(d_ptr, 2 * d_aligned_bytes);

    d_ptr = 0;
    d_handle = 0;
    d_aligned_bytes = 0;
}

} // namespace detail
} // namespace gr
