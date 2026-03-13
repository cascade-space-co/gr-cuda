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

#include "detail/device_vmm_ring.h"
#include "detail/host_mmap_ring.h"
#include <gnuradio/block.h>
#include <gnuradio/block_detail.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_buffer_reader.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/prefs.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace {

/*!
 * Return the CUDA event creation flags, reading the user preference once.
 *
 * By default events use cudaEventBlockingSync (thread sleeps on
 * cudaEventSynchronize), saving CPU when the GPU is the bottleneck.
 * To switch to spin-wait for lowest latency at the cost of CPU usage,
 * set in the path returned by `gnuradio-config-info --userprefsdir`:
 *
 *   [cuda_buffer]
 *   blocking_sync = false
 */
unsigned int cuda_event_flags()
{
    static const unsigned int flags = [] {
        unsigned int f = cudaEventDisableTiming;
        bool blocking =
            gr::prefs::singleton()->get_bool("cuda_buffer", "blocking_sync", true);
        if (blocking)
            f |= cudaEventBlockingSync;

        gr::logger_ptr log, dlog;
        gr::configure_default_loggers(log, dlog, "cuda");
        log->debug("CUDA event sync mode: {}",
                   blocking ? "blocking (sleep, default)" : "spin-wait");
        return f;
    }();
    return flags;
}

void CUDART_CB gpu_notify_cb(void* user_data)
{
    static_cast<gr::tpb_detail*>(user_data)->notify_msg();
}

} // namespace

namespace gr {

constexpr size_t CUDA_MIN_BUFFER_BYTES = 32 << 20; // 32 MB
buffer_type cuda_buffer::type(buftype<cuda_buffer, cuda_buffer>{});

cuda_buffer::cuda_buffer(int nitems,
                         size_t sizeof_item,
                         uint64_t downstream_lcm_nitems,
                         uint32_t downstream_max_out_mult,
                         block_sptr link)
    : buffer_double_mapped(nitems,
                           sizeof_item,
                           downstream_lcm_nitems,
                           downstream_max_out_mult,
                           link,
                           defer_alloc_t::defer_alloc)
{
    gr::configure_default_loggers(d_logger, d_debug_logger, "cuda");

    if (!allocate_buffer(nitems))
        throw std::bad_alloc();

    cudaError_t rc = cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking);
    check_cuda_errors(rc, "cuda_buffer: cudaStreamCreateWithFlags", d_logger);

    rc = cudaStreamCreateWithFlags(&d_notify_stream, cudaStreamNonBlocking);
    check_cuda_errors(rc, "cuda_buffer: notify stream create", d_logger);

    const unsigned int evt_flags = cuda_event_flags();
    rc = cudaEventCreateWithFlags(&d_dev_ready_evt, evt_flags);
    check_cuda_errors(rc, "cuda_buffer: device-ready event create", d_logger);
    rc = cudaEventCreateWithFlags(&d_host_ready_evt, evt_flags);
    check_cuda_errors(rc, "cuda_buffer: host-ready event create", d_logger);
    rc = cudaEventCreateWithFlags(&d_read_done_evt, evt_flags);
    check_cuda_errors(rc, "cuda_buffer: read-done event create", d_logger);
}

cuda_buffer::~cuda_buffer()
{
    // Drain any pending GPU-notify callbacks before touching events.
    if (d_notify_stream) {
        cudaStreamSynchronize(d_notify_stream);
        cudaStreamDestroy(d_notify_stream);
    }

    if (d_dev_ready_evt)
        cudaStreamWaitEvent(d_stream, d_dev_ready_evt, 0);
    if (d_host_ready_evt)
        cudaStreamWaitEvent(d_stream, d_host_ready_evt, 0);
    if (d_read_done_evt)
        cudaStreamWaitEvent(d_stream, d_read_done_evt, 0);

    cudaStreamSynchronize(d_stream);
    cudaStreamDestroy(d_stream);

    if (d_dev_ready_evt)
        cudaEventDestroy(d_dev_ready_evt);
    if (d_host_ready_evt)
        cudaEventDestroy(d_host_ready_evt);
    if (d_read_done_evt)
        cudaEventDestroy(d_read_done_evt);

    d_device_ring.reset();
    d_cuda_buf = nullptr;

    d_host_ring.reset();
    d_base = nullptr;
}

/*!
 * \brief Allocate the double-mapped host + device circular buffers.
 *
 * Called from the cuda_buffer constructor (the base buffer_double_mapped
 * used defer_alloc_t, so no vmcircbuf was created).  Sets d_base,
 * d_bufsize, d_cuda_buf from our own CUDA VMM + mmap allocations.
 */
bool cuda_buffer::allocate_buffer(int nitems)
{
    size_t vmm_granularity = detail::query_vmm_granularity_for_current_device();

    size_t raw_bytes = static_cast<size_t>(nitems) * d_sizeof_item;

    // GPU batching needs large buffers to amortise kernel launch overhead
    // and saturate PCIe bandwidth.  The scheduler caps each work() call at
    // bufsize/2, so a 32 MB buffer yields ~16 MB per call -- enough to
    // saturate PCIe and amortise launches.
    //
    // Override in the GR user prefs (gnuradio-config-info --userprefsdir):
    //   [cuda_buffer]
    //   min_buffer_bytes = 16777216   # 16 MB
    static const size_t min_cuda_bytes =
        static_cast<size_t>(gr::prefs::singleton()->get_long(
            "cuda_buffer", "min_buffer_bytes", CUDA_MIN_BUFFER_BYTES));
    size_t target_bytes = std::max(raw_bytes, min_cuda_bytes);

    // Round up to VMM granularity.
    size_t aligned_bytes =
        ((target_bytes + vmm_granularity - 1) / vmm_granularity) * vmm_granularity;

    // Ensure aligned_bytes is an exact multiple of sizeof_item so that
    // d_bufsize * sizeof_item == aligned_bytes (no partial items).
    while (aligned_bytes % d_sizeof_item != 0)
        aligned_bytes += vmm_granularity;

    d_bufsize = static_cast<unsigned>(aligned_bytes / d_sizeof_item);
    d_aligned_bytes = aligned_bytes;
    d_logger->debug("cuda_buffer: requested {} items x {} bytes = {} bytes, "
                    "floor {} bytes, aligned to {} bytes ({} items)",
                    nitems,
                    d_sizeof_item,
                    raw_bytes,
                    min_cuda_bytes,
                    aligned_bytes,
                    d_bufsize);


    // Device: VMM double-mapped circular buffer (owned by RAII helper).
    d_device_ring = detail::device_vmm_ring::create(aligned_bytes, d_logger);
    d_cuda_buf = d_device_ring->data();

    // Host ring is deferred to on_transfer_type_set(); D2D edges skip it.
    return true;
}

void cuda_buffer::on_transfer_type_set(const transfer_type& type)
{
    if (type == transfer_type::DEVICE_TO_DEVICE) {
        d_logger->debug("cuda_buffer: D2D edge — skipping host allocation");
        return;
    }

    d_host_ring = detail::host_mmap_ring::create(d_aligned_bytes, d_logger);
    d_host_ring->register_pinned();
    d_base = d_host_ring->base_ptr();
    d_logger->debug("cuda_buffer: allocated {} byte host ring for {} edge",
                    d_aligned_bytes,
                    (type == transfer_type::HOST_TO_DEVICE) ? "H2D" : "D2H");
}


int cuda_buffer::space_available()
{
    if (d_dev_ready_evt) {
        cudaError_t rc = cudaEventQuery(d_dev_ready_evt);
        if (rc == cudaErrorNotReady) {
            if (!d_dev_notify_pending.load(std::memory_order_acquire)) {
                d_dev_notify_pending.store(true, std::memory_order_release);
                cudaStreamWaitEvent(d_notify_stream, d_dev_ready_evt, 0);
                cudaLaunchHostFunc(
                    d_notify_stream, gpu_notify_cb, &link()->detail()->d_tpb);
            }
            return 0;
        }
        d_dev_notify_pending.store(false, std::memory_order_release);
    }
    return buffer_double_mapped::space_available();
}

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
    // Auto-sync: wait_read_done; see autosync table in cuda_buffer.h
    cudaStream_t ps = resolve_producer_stream();
    if (ps)
        wait_read_done(ps);

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

    // Auto-sync: mark_device_ready; see autosync table in cuda_buffer.h
    cudaStream_t ps = resolve_producer_stream();
    if (ps)
        mark_device_ready(ps);

    const unsigned wi = d_write_index;
    const unsigned tail = d_bufsize - wi;
    const unsigned produced = static_cast<unsigned>(nitems);

    switch (d_transfer_type) {

    // H2D: CPU block produced into host buffer, DMA it to the device
    case transfer_type::HOST_TO_DEVICE:
        post_work_h2d(wi, tail, produced);
        break;

    // D2H: GPU block produced into device buffer, DMA it to the host
    case transfer_type::DEVICE_TO_HOST:
        post_work_d2h(wi, tail, produced);
        break;

    // D2D: both sides are on the GPU, no DMA needed
    case transfer_type::DEVICE_TO_DEVICE:
        post_work_d2d(wi, tail, produced);
        break;

    default:
        throw_unexpected_transfer_type();
    }

    // Auto-sync: wait_device_ready; see autosync table in cuda_buffer.h
    // The natural hook would be _read_pointer() (called before the consumer
    // reads), but placing the wait here is correct because the GR scheduler
    // guarantees consumers cannot run until post_work completes. Since
    // cudaStreamWaitEvent is a GPU-side dependency, it does not matter which
    // CPU thread inserts it — only that it is in the consumer's stream queue
    // before the consumer's kernel launches.
    for (size_t i = 0; i < nreaders(); i++) {
        auto* cbr = dynamic_cast<cuda_buffer_reader*>(reader(i));
        if (cbr) {
            cudaStream_t cs = cbr->consumer_stream();
            if (cs)
                wait_device_ready(cs);
        }
    }
}

void cuda_buffer::post_work_h2d(unsigned wi, unsigned tail, unsigned nitems)
{
    cudaError_t rc;

    // Wait until all downstream GPU consumers have finished reading
    // from device memory before we overwrite it with new data.
    wait_read_done(d_stream);

    char* h_src = &d_base[wi * d_sizeof_item];
    char* d_dst = &d_cuda_buf[wi * d_sizeof_item];

    // If the write doesn't cross the buffer boundary, one copy suffices.
    // Otherwise split at the boundary: cudaHostRegister may not
    // correctly resolve mmap aliases in the second half of the 2N region.
    if (nitems <= tail) {
        rc = cudaMemcpyAsync(
            d_dst, h_src, nitems * d_sizeof_item, cudaMemcpyHostToDevice, d_stream);
        check_cuda_errors(rc, "cuda_buffer post_work: H2D", d_logger);
    } else {
        rc = cudaMemcpyAsync(
            d_dst, h_src, tail * d_sizeof_item, cudaMemcpyHostToDevice, d_stream);
        check_cuda_errors(rc, "cuda_buffer post_work: H2D part1", d_logger);

        unsigned wrap = nitems - tail;
        rc = cudaMemcpyAsync(
            d_cuda_buf, d_base, wrap * d_sizeof_item, cudaMemcpyHostToDevice, d_stream);
        check_cuda_errors(rc, "cuda_buffer post_work: H2D part2", d_logger);
    }

    // Signal downstream GPU consumers that new device data is ready.
    mark_device_ready(d_stream);
}

void cuda_buffer::post_work_d2h(unsigned wi, unsigned tail, unsigned nitems)
{
    cudaError_t rc;

    // Wait until the upstream GPU kernel has finished writing to
    // device memory before we read it for the D2H copy.
    wait_device_ready(d_stream);

    char* d_src = &d_cuda_buf[wi * d_sizeof_item];
    char* h_dst = &d_base[wi * d_sizeof_item];

    // Same split logic as H2D, keep each copy within [0, N).
    if (nitems <= tail) {
        rc = cudaMemcpyAsync(
            h_dst, d_src, nitems * d_sizeof_item, cudaMemcpyDeviceToHost, d_stream);
        check_cuda_errors(rc, "cuda_buffer post_work: D2H", d_logger);
    } else {
        rc = cudaMemcpyAsync(
            h_dst, d_src, tail * d_sizeof_item, cudaMemcpyDeviceToHost, d_stream);
        check_cuda_errors(rc, "cuda_buffer post_work: D2H part1", d_logger);

        unsigned wrap = nitems - tail;
        rc = cudaMemcpyAsync(
            d_base, d_cuda_buf, wrap * d_sizeof_item, cudaMemcpyDeviceToHost, d_stream);
        check_cuda_errors(rc, "cuda_buffer post_work: D2H part2", d_logger);
    }

    // Signal downstream CPU readers that host data has landed.
    mark_host_ready(d_stream);

    // Record that the D2H copy has finished reading from device memory,
    // so the upstream producer (via wait_read_done in write_pointer())
    // knows it's safe to overwrite.
    cudaEventRecord(d_read_done_evt, d_stream);
}

void cuda_buffer::post_work_d2d(unsigned, unsigned, unsigned)
{
    // D2D has no host/device DMA here.
}

void cuda_buffer::mark_device_ready(cudaStream_t producer_stream)
{
    cudaEventRecord(d_dev_ready_evt, producer_stream);
}

void cuda_buffer::wait_device_ready(cudaStream_t consumer_stream)
{
    cudaStreamWaitEvent(consumer_stream, d_dev_ready_evt, 0);
}

void cuda_buffer::mark_host_ready(cudaStream_t copy_stream)
{
    cudaEventRecord(d_host_ready_evt, copy_stream);
}

void cuda_buffer::wait_host_ready() { cudaEventSynchronize(d_host_ready_evt); }

void cuda_buffer::mark_read_done(cudaStream_t consumer_stream)
{
    std::lock_guard<std::mutex> lock(d_read_done_mutex);
    cudaStreamWaitEvent(consumer_stream, d_read_done_evt, 0);
    cudaEventRecord(d_read_done_evt, consumer_stream);
}

void cuda_buffer::wait_read_done(cudaStream_t producer_stream)
{
    cudaStreamWaitEvent(producer_stream, d_read_done_evt, 0);
}

// Stream discovery

void cuda_buffer::set_producer_stream(cudaStream_t s)
{
    d_producer_stream = s;
    d_producer_stream_resolved = true;
}

cudaStream_t cuda_buffer::resolve_producer_stream()
{
    if (!d_producer_stream_resolved) {
        d_producer_stream_resolved = true;
        auto* cb = dynamic_cast<cuda_block*>(link().get());
        if (cb)
            d_producer_stream = cb->get_cuda_stream();
    }
    return d_producer_stream;
}

// Reader factory

buffer_reader_sptr cuda_buffer::create_reader(buffer_sptr buf,
                                              int nzero_preload,
                                              block_sptr link,
                                              int delay)
{
    buffer_reader_sptr r;
    r.reset(new cuda_buffer_reader(buf, index_sub(d_write_index, nzero_preload), link));
    r->declare_sample_delay(delay);
    return r;
}

// Factory

buffer_sptr cuda_buffer::make_buffer(int nitems,
                                     size_t sizeof_item,
                                     uint64_t downstream_lcm_nitems,
                                     uint32_t downstream_max_out_mult,
                                     block_sptr link,
                                     block_sptr /*buf_owner*/)
{
    return buffer_sptr(new cuda_buffer(
        nitems, sizeof_item, downstream_lcm_nitems, downstream_max_out_mult, link));
}

void cuda_buffer::throw_unexpected_transfer_type()
{
    std::ostringstream msg;
    msg << "cuda_buffer: unexpected transfer_type: " << d_transfer_type;
    d_logger->error("{}", msg.str());
    throw std::runtime_error(msg.str());
}

} /* namespace gr */
