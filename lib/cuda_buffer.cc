/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 * Copyright 2021 BlackLynx, Inc.
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/block.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace gr {

/*!
 * \brief Result of a VMM device-side double-mapped allocation.
 *
 * ptr points to the start of the 2N VA region; the same physical allocation
 * (handle) is mapped at [ptr, ptr+aligned_bytes) and
 * [ptr+aligned_bytes, ptr+2*aligned_bytes).
 */
struct vmm_device_alloc {
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle handle;
    size_t aligned_bytes;  // N (one half), aligned to VMM granularity
};

/*!
 * \brief Query the minimum VMM allocation granularity for the current device.
 *
 * All VMM allocations and mappings must be multiples of this value
 * (typically 2 MB on modern NVIDIA GPUs).
 */
static size_t vmm_query_granularity()
{
    int device;
    check_cuda_errors(cudaGetDevice(&device), "vmm: cudaGetDevice");

    CUdevice cu_dev;
    CUresult res = cuDeviceGet(&cu_dev, device);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm: cuDeviceGet failed");

    // Pinned, device-local, matches what vmm_create_double_mapped will use
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

/*!
 * \brief Create a device-side circular buffer using CUDA Driver VMM.
 *
 * Allocates N bytes of physical GPU memory and maps it into two adjacent
 * virtual address ranges [ptr, ptr+N) and [ptr+N, ptr+2N), both backed by
 * the same physical allocation.  This is the GPU-side equivalent of the
 * host mmap double-mapping: CUDA kernels and DMA see a contiguous 2N
 * region that wraps transparently.
 *
 * \code
 *   GPU virtual address space:
 *   [ ---- half 1 ---- | ---- half 2 ---- ]
 *     ^                   ^
 *     ptr                 ptr + N
 *     \__ same physical __/
 *         allocation
 * \endcode
 *
 * Cleanup order (in vmm_destroy): unmap both halves, release handle, free VA.
 */
static vmm_device_alloc vmm_create_double_mapped(size_t requested_bytes)
{
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

    size_t granularity = 0;
    res = cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm_create: cuMemGetAllocationGranularity failed");

    // Round up to granularity, VMM requires all sizes/offsets to be multiples
    size_t aligned =
        ((requested_bytes + granularity - 1) / granularity) * granularity;

    // Reserve 2N contiguous VA (no physical memory yet)
    CUdeviceptr ptr = 0;
    res = cuMemAddressReserve(&ptr, 2 * aligned, 0, 0, 0);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("vmm_create: cuMemAddressReserve failed");

    // Create one physical allocation of size N
    CUmemGenericAllocationHandle handle = 0;
    res = cuMemCreate(&handle, aligned, &prop, 0);
    if (res != CUDA_SUCCESS) {
        cuMemAddressFree(ptr, 2 * aligned);
        throw std::runtime_error("vmm_create: cuMemCreate failed");
    }

    // Map the physical allocation into the first half [ptr, ptr+N)
    res = cuMemMap(ptr, aligned, 0, handle, 0);
    if (res != CUDA_SUCCESS) {
        cuMemRelease(handle);
        cuMemAddressFree(ptr, 2 * aligned);
        throw std::runtime_error("vmm_create: cuMemMap first half failed");
    }

    // Map the same allocation into the second half [ptr+N, ptr+2N)
    res = cuMemMap(ptr + aligned, aligned, 0, handle, 0);
    if (res != CUDA_SUCCESS) {
        cuMemUnmap(ptr, aligned);
        cuMemRelease(handle);
        cuMemAddressFree(ptr, 2 * aligned);
        throw std::runtime_error("vmm_create: cuMemMap second half failed");
    }

    // Grant read/write access across the full 2N range
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    res = cuMemSetAccess(ptr, 2 * aligned, &access, 1);
    if (res != CUDA_SUCCESS) {
        cuMemUnmap(ptr, aligned);
        cuMemUnmap(ptr + aligned, aligned);
        cuMemRelease(handle);
        cuMemAddressFree(ptr, 2 * aligned);
        throw std::runtime_error("vmm_create: cuMemSetAccess failed");
    }

    return {ptr, handle, aligned};
}

static void vmm_destroy(vmm_device_alloc& alloc)
{
    if (alloc.ptr) {
        cuMemUnmap(alloc.ptr, alloc.aligned_bytes);
        cuMemUnmap(alloc.ptr + alloc.aligned_bytes, alloc.aligned_bytes);
        cuMemRelease(alloc.handle);
        cuMemAddressFree(alloc.ptr, 2 * alloc.aligned_bytes);
        alloc.ptr = 0;
    }
}

/*!
 * \brief Result of a host-side mmap double-mapped allocation.
 */
struct host_circ_alloc {
    void* base;   // start of the 2N virtual address region
    size_t bytes;  // N (one half), page-aligned
};

/*!
 * \brief Create a host-side circular buffer using mmap double-mapping.
 *
 * Allocates N bytes of physical memory (via memfd) and maps it into two
 * adjacent virtual address ranges [base, base+N) and [base+N, base+2N),
 * both backed by the same physical pages.  This makes a contiguous read
 * starting near the end of the buffer transparently wrap around to the
 * beginning without any memcpy, the hardware MMU handles it natively.
 */
static host_circ_alloc host_circ_create(size_t requested_bytes)
{
    long page_size = sysconf(_SC_PAGESIZE);
    size_t bytes =
        ((requested_bytes + page_size - 1) / (size_t)page_size) * page_size;

    // Anonymous file backed by RAM, no filesystem path needed
    int fd = memfd_create("gr_cuda_buf", 0);
    if (fd < 0)
        throw std::runtime_error("host_circ_create: memfd_create failed");

    if (ftruncate(fd, (off_t)bytes) != 0) {
        close(fd);
        throw std::runtime_error("host_circ_create: ftruncate failed");
    }

    // Reserve 2N contiguous VA with no access rights (placeholder)
    void* region =
        mmap(nullptr, 2 * bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("host_circ_create: VA reservation mmap failed");
    }

    // Map the fd into the first half [base, base+N), replacing the placeholder
    void* p1 =
        mmap(region, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    if (p1 == MAP_FAILED) {
        munmap(region, 2 * bytes);
        close(fd);
        throw std::runtime_error("host_circ_create: first-half mmap failed");
    }

    // Map the same fd into the second half [base+N, base+2N)
    void* p2 = mmap(static_cast<char*>(region) + bytes, bytes,
                     PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    if (p2 == MAP_FAILED) {
        munmap(region, 2 * bytes);
        close(fd);
        throw std::runtime_error("host_circ_create: second-half mmap failed");
    }

    // fd can be closed immediately; the mappings hold a reference
    close(fd);
    return {region, bytes};
}

static void host_circ_destroy(host_circ_alloc& alloc)
{
    if (alloc.base) {
        munmap(alloc.base, 2 * alloc.bytes);
        alloc.base = nullptr;
    }
}


buffer_type cuda_buffer::type(buftype<cuda_buffer, cuda_buffer>{});

cuda_buffer::cuda_buffer(int nitems,
                         size_t sizeof_item,
                         uint64_t downstream_lcm_nitems,
                         uint32_t downstream_max_out_mult,
                         block_sptr link,
                         block_sptr buf_owner)
    : buffer_single_mapped(nitems, sizeof_item, downstream_lcm_nitems,
                           downstream_max_out_mult, link, buf_owner)
{
    gr::configure_default_loggers(d_logger, d_debug_logger, "cuda");

    if (!allocate_buffer(nitems))
        throw std::bad_alloc();

    cudaError_t rc = cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking);
    check_cuda_errors(rc, "cuda_buffer: cudaStreamCreateWithFlags", d_logger);

    for (int i = 0; i < PIPELINE_DEPTH; i++) {
        rc = cudaEventCreateWithFlags(&d_dev_ready_evt[i], cudaEventDisableTiming);
        check_cuda_errors(rc, "cuda_buffer: device-ready event create", d_logger);

        rc = cudaEventCreateWithFlags(&d_host_ready_evt[i], cudaEventDisableTiming);
        check_cuda_errors(rc, "cuda_buffer: host-ready event create", d_logger);
    }

    rc = cudaEventCreateWithFlags(&d_read_done_evt, cudaEventDisableTiming);
    check_cuda_errors(rc, "cuda_buffer: read-done event create", d_logger);
}

cuda_buffer::~cuda_buffer()
{
    for (int i = 0; i < PIPELINE_DEPTH; i++) {
        if (d_dev_ready_evt[i])
            cudaStreamWaitEvent(d_stream, d_dev_ready_evt[i], 0);
        if (d_host_ready_evt[i])
            cudaStreamWaitEvent(d_stream, d_host_ready_evt[i], 0);
    }
    if (d_read_done_evt)
        cudaStreamWaitEvent(d_stream, d_read_done_evt, 0);

    cudaStreamSynchronize(d_stream);

    cudaStreamDestroy(d_stream);

    for (int i = 0; i < PIPELINE_DEPTH; i++) {
        if (d_dev_ready_evt[i])
            cudaEventDestroy(d_dev_ready_evt[i]);
        if (d_host_ready_evt[i])
            cudaEventDestroy(d_host_ready_evt[i]);
    }
    if (d_read_done_evt)
        cudaEventDestroy(d_read_done_evt);

    if (d_host_registered && d_host_mmap_base) {
        cudaHostUnregister(d_host_mmap_base);
        d_host_registered = false;
    }

    vmm_device_alloc dev = {d_vmm_ptr, d_vmm_handle, d_vmm_aligned_bytes};
    vmm_destroy(dev);
    d_vmm_ptr = 0;
    d_cuda_buf = nullptr;

    host_circ_alloc host = {d_host_mmap_base, d_host_mmap_bytes};
    host_circ_destroy(host);
    d_host_mmap_base = nullptr;

    d_base = nullptr;
}

/*!
 * \brief Allocate the double-mapped host + device circular buffers.
 *
 * Called by buffer_single_mapped::allocate_buffer() after it computes
 * final_nitems from the scheduler's requirements.  We ignore the base
 * class's d_buffer (std::unique_ptr<char[]>) and set up our own
 * double-mapped host + device regions instead.
 */
bool cuda_buffer::do_allocate_buffer(size_t final_nitems, size_t sizeof_item)
{
    size_t vmm_granularity = vmm_query_granularity();

    // Round up to VMM granularity.  This may significantly increase the
    // buffer size (e.g. 128 items * 8 bytes → 2 MB).
    size_t raw_bytes = final_nitems * sizeof_item;
    size_t aligned_bytes =
        ((raw_bytes + vmm_granularity - 1) / vmm_granularity) * vmm_granularity;

    // Ensure aligned_bytes is an exact multiple of sizeof_item so that
    // d_bufsize * sizeof_item == aligned_bytes (no partial items).
    while (aligned_bytes % sizeof_item != 0)
        aligned_bytes += vmm_granularity;

    d_bufsize = static_cast<unsigned>(aligned_bytes / sizeof_item);

    // 1) Host: mmap double-mapped circular buffer (see host_circ_create)
    host_circ_alloc host = host_circ_create(aligned_bytes);
    d_host_mmap_base = host.base;
    d_host_mmap_bytes = host.bytes;
    d_base = static_cast<char*>(d_host_mmap_base);

    // Pin the full 2N host region for async DMA (cudaMemcpyAsync).
    // We register the entire double-mapped range so copies that
    // straddle the wrap boundary still hit pinned memory.
    cudaError_t rc =
        cudaHostRegister(d_host_mmap_base, 2 * d_host_mmap_bytes,
                         cudaHostRegisterDefault);
    if (rc != cudaSuccess) {
        d_logger->error("cudaHostRegister failed: {} ({})",
                        cudaGetErrorName(rc), cudaGetErrorString(rc));
        host_circ_destroy(host);
        d_host_mmap_base = nullptr;
        d_base = nullptr;
        return false;
    }
    d_host_registered = true;

    // 2) Device: VMM double-mapped circular buffer (see vmm_create_double_mapped)
    vmm_device_alloc dev = vmm_create_double_mapped(aligned_bytes);
    d_vmm_ptr = dev.ptr;
    d_vmm_handle = dev.handle;
    d_vmm_aligned_bytes = dev.aligned_bytes;
    d_cuda_buf = reinterpret_cast<char*>(static_cast<uintptr_t>(d_vmm_ptr));

    return true;
}


/*!
 * \brief Return the number of items the writer may produce in this call.
 *
 * Classic ring-buffer rule: the writer must not lap the slowest reader.
 * most_data = max items_available across all readers (i.e. the fullest
 * reader).  The "-1" reserves one sentinel slot so that a completely
 * full buffer is distinguishable from an empty one (write_index never
 * equals read_index unless the buffer is empty).
 */
int cuda_buffer::space_available()
{
    if (d_readers.empty())
        return d_bufsize - 1;

    int most_data = d_readers[0]->items_available();
    uint64_t min_items_read = d_readers[0]->nitems_read();
    for (size_t i = 1; i < d_readers.size(); i++) {
        most_data = std::max(most_data, d_readers[i]->items_available());
        min_items_read = std::min(min_items_read, d_readers[i]->nitems_read());
    }

    // Prune tags that all readers have consumed
    if (min_items_read != d_last_min_items_read) {
        prune_tags(d_last_min_items_read);
        d_last_min_items_read = min_items_read;
    }

    return d_bufsize - most_data - 1;
}

// No-op compaction
bool cuda_buffer::input_blkd_cb_ready(int, unsigned) { return false; }
bool cuda_buffer::output_blkd_cb_ready(int) { return false; }
bool cuda_buffer::input_blocked_callback(int, int, unsigned) { return false; }
bool cuda_buffer::output_blocked_callback(int, bool) { return false; }

/*!
 * \brief Return where the upstream block should write its output.
 *
 * d_write_index and read_index are always in [0, N) (modulo d_bufsize).
 * Because both d_base and d_cuda_buf point to the start of a 2N
 * double-mapped region, the returned pointer may fall anywhere in [0, 2N).
 * A contiguous write/read that starts near N and spills past it
 * transparently wraps to the beginning via the second-half alias;
 * no split copies or boundary checks are needed by the caller.
 *
 * - H2D: write into host (d_base), post_work will DMA to device.
 * - D2H / D2D: write directly into device memory (d_cuda_buf).
 */
void* cuda_buffer::write_pointer()
{
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE:
        return &d_base[d_write_index * d_sizeof_item];

    case transfer_type::DEVICE_TO_HOST:
    case transfer_type::DEVICE_TO_DEVICE:
        return &d_cuda_buf[d_write_index * d_sizeof_item];

    default:
        throw_unexpected_transfer_type();
    }
}

/*!
 * \brief Return where the downstream block should read its input.
 *
 * - H2D / D2D: read from device memory (d_cuda_buf).
 * - D2H: read from host (d_base), after ensuring the DMA has landed.
 */
const void* cuda_buffer::_read_pointer(unsigned int read_index)
{
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE:
    case transfer_type::DEVICE_TO_DEVICE:
        return &d_cuda_buf[read_index * d_sizeof_item];

    case transfer_type::DEVICE_TO_HOST:
        wait_host_ready();
        return &d_base[read_index * d_sizeof_item];

    default:
        throw_unexpected_transfer_type();
    }
}

/*!
 * \brief Transfer data between host and device after general_work().
 *
 * Host DMA via cudaHostRegister may not honour mmap aliasing on the second
 * half of the 2N VA region, so every memcpy that would cross the d_bufsize
 * boundary is split into two copies that each stay within [0, N).
 * The device side uses VMM, which handles aliasing natively.
 */
void cuda_buffer::post_work(int nitems)
{
    if (nitems <= 0)
        return;

    cudaError_t rc;
    const unsigned wi = d_write_index;
    const unsigned tail = d_bufsize - wi;

    switch (d_transfer_type) {

    // H2D: CPU block produced into host buffer, DMA it to the device
    case transfer_type::HOST_TO_DEVICE: {
        // Wait until all downstream GPU consumers have finished reading
        // from device memory before we overwrite it with new data.
        cudaStreamWaitEvent(d_stream, d_read_done_evt, 0);

        char* h_src = &d_base[wi * d_sizeof_item];
        char* d_dst = &d_cuda_buf[wi * d_sizeof_item];

        // If the write doesn't cross the buffer boundary, one copy suffices.
        // Otherwise split at the boundary: cudaHostRegister may not
        // correctly resolve mmap aliases in the second half of the 2N region.
        if ((unsigned)nitems <= tail) {
            rc = cudaMemcpyAsync(d_dst, h_src,
                                 nitems * d_sizeof_item,
                                 cudaMemcpyHostToDevice, d_stream);
            check_cuda_errors(rc, "cuda_buffer post_work: H2D", d_logger);
        } else {
            rc = cudaMemcpyAsync(d_dst, h_src,
                                 tail * d_sizeof_item,
                                 cudaMemcpyHostToDevice, d_stream);
            check_cuda_errors(rc, "cuda_buffer post_work: H2D part1", d_logger);

            unsigned wrap = nitems - tail;
            rc = cudaMemcpyAsync(d_cuda_buf, d_base,
                                 wrap * d_sizeof_item,
                                 cudaMemcpyHostToDevice, d_stream);
            check_cuda_errors(rc, "cuda_buffer post_work: H2D part2", d_logger);
        }

        // Signal downstream GPU consumers that new device data is ready.
        mark_device_ready(d_stream);
    } break;

    // D2H: GPU block produced into device buffer, DMA it to the host
    case transfer_type::DEVICE_TO_HOST: {
        // Wait until the upstream GPU kernel has finished writing to
        // device memory before we read it for the D2H copy.
        wait_device_ready(d_stream);

        char* d_src = &d_cuda_buf[wi * d_sizeof_item];
        char* h_dst = &d_base[wi * d_sizeof_item];

        // Same split logic as H2D, keep each copy within [0, N).
        if ((unsigned)nitems <= tail) {
            rc = cudaMemcpyAsync(h_dst, d_src,
                                 nitems * d_sizeof_item,
                                 cudaMemcpyDeviceToHost, d_stream);
            check_cuda_errors(rc, "cuda_buffer post_work: D2H", d_logger);
        } else {
            rc = cudaMemcpyAsync(h_dst, d_src,
                                 tail * d_sizeof_item,
                                 cudaMemcpyDeviceToHost, d_stream);
            check_cuda_errors(rc, "cuda_buffer post_work: D2H part1", d_logger);

            unsigned wrap = nitems - tail;
            rc = cudaMemcpyAsync(d_base, d_cuda_buf,
                                 wrap * d_sizeof_item,
                                 cudaMemcpyDeviceToHost, d_stream);
            check_cuda_errors(rc, "cuda_buffer post_work: D2H part2", d_logger);
        }

        // Signal downstream CPU readers that host data has landed.
        mark_host_ready(d_stream);

        // Record that the D2H copy has finished reading from device memory,
        // so the upstream GPU kernel (via wait_for_work → wait_read_done)
        // knows it's safe to overwrite.
        cudaEventRecord(d_read_done_evt, d_stream);
    } break;

    // D2D: both sides are on the GPU, no DMA needed
    case transfer_type::DEVICE_TO_DEVICE:
        break;

    default:
        throw_unexpected_transfer_type();
    }
}

// Event pipeline (ring-buffer, PIPELINE_DEPTH deep)

void cuda_buffer::mark_device_ready(cudaStream_t producer_stream)
{
    int slot = d_dev_ready_next % PIPELINE_DEPTH;
    cudaEvent_t evt = d_dev_ready_evt[slot];

    if (cudaEventQuery(evt) != cudaSuccess)
        cudaEventSynchronize(evt);

    cudaEventRecord(evt, producer_stream);
    d_dev_ready_next++;
}

void cuda_buffer::wait_device_ready(cudaStream_t consumer_stream)
{
    for (int i = 0; i < PIPELINE_DEPTH; i++)
        cudaStreamWaitEvent(consumer_stream, d_dev_ready_evt[i], 0);
}

void cuda_buffer::mark_host_ready(cudaStream_t copy_stream)
{
    int slot = d_host_ready_next % PIPELINE_DEPTH;
    cudaEventRecord(d_host_ready_evt[slot], copy_stream);
    d_host_ready_next++;
}

void cuda_buffer::wait_host_ready()
{
    for (int i = 0; i < PIPELINE_DEPTH; i++)
        cudaEventSynchronize(d_host_ready_evt[i]);
}

void cuda_buffer::mark_read_done(cudaStream_t consumer_stream)
{
    std::lock_guard<std::mutex> lock(d_read_done_mutex);
    cudaStreamWaitEvent(consumer_stream, d_read_done_evt, 0);
    cudaEventRecord(d_read_done_evt, consumer_stream);
}

void cuda_buffer::wait_read_done(cudaStream_t stream)
{
    cudaStreamWaitEvent(stream, d_read_done_evt, 0);
}

// ─── Factory ────────────────────────────────────────────────────────────────

buffer_sptr cuda_buffer::make_buffer(int nitems,
                                     size_t sizeof_item,
                                     uint64_t downstream_lcm_nitems,
                                     uint32_t downstream_max_out_mult,
                                     block_sptr link,
                                     block_sptr buf_owner)
{
    return buffer_sptr(new cuda_buffer(
        nitems, sizeof_item, downstream_lcm_nitems,
        downstream_max_out_mult, link, buf_owner));
}

void cuda_buffer::throw_unexpected_transfer_type()
{
    std::ostringstream msg;
    msg << "cuda_buffer: unexpected transfer_type: " << d_transfer_type;
    d_logger->error("{}", msg.str());
    throw std::runtime_error(msg.str());
}

} /* namespace gr */
