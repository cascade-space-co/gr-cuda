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

#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace gr {

buffer_type cuda_buffer::type(buftype<cuda_buffer, cuda_buffer>{});

void* cuda_buffer::cuda_memcpy(void* dest, const void* src, std::size_t count)
{
    cudaError_t rc =
        cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, d_stream);
    cudaStreamSynchronize(d_stream);
    if (rc)
        throw_cuda_error("Error performing cudaMemcpy", rc);

    return dest;
}

void* cuda_buffer::cuda_memmove(void* dest, const void* src, std::size_t count)
{
    cudaError_t rc;

    // Allocate temp buffer if needed
    if (count > d_temp_buffer_size) {
        if (d_temp_buffer != nullptr) {
            cudaFree(d_temp_buffer);
            d_temp_buffer = nullptr;
            d_temp_buffer_size = 0;
        }
        rc = cudaMalloc((void**)&d_temp_buffer, count);
        if (rc)
            throw_cuda_error("Error allocating device temp buffer", rc);
        d_temp_buffer_size = count;
    }

    // First copy data from source to temp buffer
    rc = cudaMemcpyAsync(d_temp_buffer, src, count, cudaMemcpyDeviceToDevice, d_stream);

    if (rc)
        throw_cuda_error("Error performing cudaMemcpy", rc);

    // Then copy data from temp buffer to destination to avoid overlap
    rc = cudaMemcpyAsync(dest, d_temp_buffer, count, cudaMemcpyDeviceToDevice, d_stream);

    if (rc)
        throw_cuda_error("Error performing cudaMemcpy", rc);

    // Sync for consistency with cuda_memcpy: callers (blocked-callback logic)
    // expect the data to be in place when the function returns.
    cudaStreamSynchronize(d_stream);

    return dest;
}

cuda_buffer::cuda_buffer(int nitems,
                         size_t sizeof_item,
                         uint64_t downstream_lcm_nitems,
                         uint32_t downstream_max_out_mult,
                         block_sptr link,
                         block_sptr buf_owner)
    : buffer_single_mapped(nitems, sizeof_item, downstream_lcm_nitems,
                           downstream_max_out_mult, link, buf_owner),
      d_cuda_buf(nullptr),
      d_half_nitems(0),
      d_temp_buffer(nullptr),
      d_temp_buffer_size(0)
{
    gr::configure_default_loggers(d_logger, d_debug_logger, "cuda");
    if (!allocate_buffer(nitems))
        throw std::bad_alloc();

    f_cuda_memcpy = [this](void* dest, const void* src, std::size_t count){ return this->cuda_memcpy(dest, src, count); };
    f_cuda_memmove = [this](void* dest, const void* src, std::size_t count){ return this->cuda_memmove(dest, src, count); };
    cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking);
    for (int i = 0; i < NUM_HALF_EVENTS; i++) {
        cudaEventCreateWithFlags(&d_dev_ready_evt[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&d_host_ready_evt[i], cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&d_read_done_evt, cudaEventDisableTiming);
}

cuda_buffer::~cuda_buffer()
{
    // Drain all in-flight GPU work before releasing any resources.
    // cudaStreamDestroy alone is not sufficient — it returns immediately
    // and defers cleanup, so a subsequent cudaFreeHost could free pinned
    // memory while an async H2D/D2H copy is still in flight.
    cudaStreamSynchronize(d_stream);

    cudaStreamDestroy(d_stream);
    for (int i = 0; i < NUM_HALF_EVENTS; i++) {
        cudaEventDestroy(d_dev_ready_evt[i]);
        cudaEventDestroy(d_host_ready_evt[i]);
    }
    cudaEventDestroy(d_read_done_evt);

    // Free host buffer
    if (d_base != nullptr) {
        cudaFreeHost(d_base);
        d_base = nullptr;
    }

    // Free device buffer
    if (d_cuda_buf != nullptr) {
        cudaFree(d_cuda_buf);
        d_cuda_buf = nullptr;
    }

    // Free scratch buffer
    if (d_temp_buffer != nullptr) {
        cudaFree(d_temp_buffer);
        d_temp_buffer = nullptr;
    }
}

void cuda_buffer::post_work(int nitems)
{
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "cuda [" << d_transfer_type << "] -- post_work: " << nitems;
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    if (nitems <= 0) {
        return;
    }

    cudaError_t rc;

    // NOTE: when this function is called the write pointer has not yet been
    // advanced so it can be used directly as the source ptr
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE: {
        // Ensure all consumers have finished reading from the device buffer
        // before the H2D copy overwrites it.  GPU-side wait — does not block CPU.
        cudaStreamWaitEvent(d_stream, d_read_done_evt, 0);

        // Copy data from host buffer to device buffer
        void* dest_ptr = &d_cuda_buf[d_write_index * d_sizeof_item];
        rc = cudaMemcpyAsync(
            dest_ptr, write_pointer(), nitems * d_sizeof_item, cudaMemcpyHostToDevice, d_stream);
        if (rc)
            throw_cuda_error("Error performing cudaMemcpy", rc);

        mark_device_ready(d_stream);

    } break;

    case transfer_type::DEVICE_TO_HOST: {
        // Ensure producing GPU work is complete before copying
        wait_device_ready(d_stream);

        // Copy data from device buffer to host buffer.
        // Compute the source address directly rather than calling
        // write_pointer() again (it was already called by the executor
        // to set up the output items).
        void* src_ptr  = &d_cuda_buf[d_write_index * d_sizeof_item];
        void* dest_ptr = &d_base[d_write_index * d_sizeof_item];
        rc = cudaMemcpyAsync(
            dest_ptr, src_ptr, nitems * d_sizeof_item, cudaMemcpyDeviceToHost, d_stream);
        if (rc)
            throw_cuda_error("Error performing cudaMemcpy", rc);

        mark_host_ready(d_stream);

        // Signal that the D2H copy is done reading from the device buffer.
        // wait_for_work() adds a GPU-side wait on this event to the upstream
        // producer's kernel stream, preventing a race between the async D2H
        // copy and the next producer write.
        cudaEventRecord(d_read_done_evt, d_stream);

    } break;

    case transfer_type::DEVICE_TO_DEVICE:
        // No op FTW!
        break;

    default:
        throw_unexpected_transfer_type();
    }
}

bool cuda_buffer::do_allocate_buffer(size_t final_nitems, size_t sizeof_item)
{
#ifdef BUFFER_DEBUG
    {
        std::ostringstream msg;
        msg << "[" << this << "] "
            << "cuda constructor -- nitems: " << final_nitems;
        GR_LOG_DEBUG(d_logger, msg.str());
    }
#endif

    // Store half-buffer boundary for double-buffered event selection
    d_half_nitems = final_nitems / 2;

    // Pinned host buffer
    cudaError_t rc = cudaMallocHost((void**)&d_base, final_nitems * sizeof_item);
    if (rc)
        throw_cuda_error("Error allocating pinned host buffer", rc);

    // Device buffer
    rc = cudaMalloc((void**)&d_cuda_buf, final_nitems * sizeof_item);
    if (rc)
        throw_cuda_error("Error allocating device buffer", rc);

    return true;
}

void* cuda_buffer::write_pointer()
{
    void* ptr = nullptr;
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE:
        // Write into host buffer
        ptr = &d_base[d_write_index * d_sizeof_item];
        break;

    case transfer_type::DEVICE_TO_HOST:
    case transfer_type::DEVICE_TO_DEVICE:
        // Return the device pointer without CPU-side synchronisation.
        //
        // Safety against consumer-to-producer data hazards (an async consumer
        // kernel or D2H copy still reading from a region the producer is about
        // to overwrite) is enforced GPU-side: wait_for_work() in
        // cuda_block_helper.h adds a cudaStreamWaitEvent on d_read_done_evt
        // to the producer's kernel stream, so the kernel will not execute
        // until every consumer has finished.  This avoids blocking the CPU
        // thread and allows H2D / D2H transfers to overlap on dual-copy-
        // engine GPUs.
        //
        // Index-level safety (non-overlapping regions) is guaranteed by
        // buffer_single_mapped::space_available().
        ptr = &d_cuda_buf[d_write_index * d_sizeof_item];
        break;

    default:
        throw_unexpected_transfer_type();
    }

    return ptr;
}

const void* cuda_buffer::_read_pointer(unsigned int read_index)
{
    void* ptr = nullptr;
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE:
    case transfer_type::DEVICE_TO_DEVICE:
        // Read from "device" buffer
        ptr = &d_cuda_buf[read_index * d_sizeof_item];
        break;

    case transfer_type::DEVICE_TO_HOST:
        // Read from host buffer
        wait_host_ready();
        ptr = &d_base[read_index * d_sizeof_item];
        break;

    default:
        throw_unexpected_transfer_type();
    }

    return ptr;
}

bool cuda_buffer::input_blocked_callback(int items_required,
                                         int items_avail,
                                         unsigned read_index)
{
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "cuda [" << d_transfer_type << "] -- input_blocked_callback";
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    sync_all_gpu_work();

    bool rc = false;
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE:
    case transfer_type::DEVICE_TO_DEVICE:
        // Adjust "device" buffer
        rc = input_blocked_callback_logic(items_required,
                                          items_avail,
                                          read_index,
                                          d_cuda_buf,
                                          f_cuda_memcpy,
                                          f_cuda_memmove);
        break;

    case transfer_type::DEVICE_TO_HOST:
        // Adjust host buffer
        rc = input_blocked_callback_logic(
            items_required, items_avail, read_index, d_base, std::memcpy, std::memmove);
        break;

    default:
        throw_unexpected_transfer_type();
    }

    return rc;
}

bool cuda_buffer::output_blocked_callback(int output_multiple, bool force)
{
#ifdef BUFFER_DEBUG
    std::ostringstream msg;
    msg << "[" << this << "] "
        << "cuda [" << d_transfer_type << "] -- output_blocked_callback";
    GR_LOG_DEBUG(d_logger, msg.str());
#endif

    sync_all_gpu_work();

    bool rc = false;
    switch (d_transfer_type) {
    case transfer_type::HOST_TO_DEVICE: {
        // Realign both host and device so the consumer (GPU, reading d_cuda_buf)
        // sees the same layout as the producer (CPU, wrote d_base).
        auto h2d_memmove = [this](void* dest, const void* src, std::size_t count) {
            std::memmove(dest, src, count);
            const std::ptrdiff_t dest_off =
                static_cast<char*>(dest) - static_cast<char*>(d_base);
            const std::ptrdiff_t src_off =
                static_cast<const char*>(src) - static_cast<const char*>(d_base);
            cuda_memmove(static_cast<char*>(d_cuda_buf) + dest_off,
                         static_cast<const char*>(d_cuda_buf) + src_off,
                         count);
            return dest;
        };
        rc = output_blocked_callback_logic(
            output_multiple, force, d_base, h2d_memmove);
        break;
    }

    case transfer_type::DEVICE_TO_HOST: {
        // Realign both device and host buffers so the consumer (reading d_base)
        // sees the same logical layout. The base class updates indices based on
        // the single buffer we pass; we must apply the same move to d_base so
        // _read_pointer() does not return stale data from the previous wrap.
        auto d2h_memmove = [this](void* dest, const void* src, std::size_t count) {
            cuda_memmove(dest, src, count);
            const std::ptrdiff_t dest_off =
                static_cast<char*>(dest) - static_cast<char*>(d_cuda_buf);
            const std::ptrdiff_t src_off =
                static_cast<const char*>(src) - static_cast<const char*>(d_cuda_buf);
            std::memmove(static_cast<char*>(d_base) + dest_off,
                        static_cast<const char*>(d_base) + src_off,
                        count);
            return dest;
        };
        rc = output_blocked_callback_logic(
            output_multiple, force, d_cuda_buf, d2h_memmove);
        break;
    }

    case transfer_type::DEVICE_TO_DEVICE:
        // Producer and consumer both use d_cuda_buf; no host buffer in the path.
        rc = output_blocked_callback_logic(
            output_multiple, force, d_cuda_buf, f_cuda_memmove);
        break;

    default:
        throw_unexpected_transfer_type();
    }

    return rc;
}

void cuda_buffer::sync_all_gpu_work()
{
    for (int i = 0; i < NUM_HALF_EVENTS; i++) {
        cudaEventSynchronize(d_dev_ready_evt[i]);
    }
    cudaEventSynchronize(d_read_done_evt);
    cudaStreamSynchronize(d_stream);
}

void cuda_buffer::throw_cuda_error(const char* context, cudaError_t rc)
{
    // Log via GR logger before throwing
    std::ostringstream msg;
    msg << context << ": " << cudaGetErrorName(rc) << " -- " << cudaGetErrorString(rc);
    GR_LOG_ERROR(d_logger, msg.str());
    throw_on_cuda_error(context, rc);
}

void cuda_buffer::throw_unexpected_transfer_type()
{
    std::ostringstream msg;
    msg << "Unexpected context for cuda: " << d_transfer_type;
    GR_LOG_ERROR(d_logger, msg.str());
    throw std::runtime_error(msg.str());
}

void cuda_buffer::mark_device_ready(cudaStream_t producer_stream)
{
    // Double-buffered events: record on the event for the current buffer half.
    // Before re-recording, wait for the previous recording of this half's event
    // to complete.  This provides essential CPU-side backpressure: without it
    // the producer's CPU code can race far ahead, re-recording events before
    // the GPU processes the earlier recording.
    //
    // Consumer-to-producer data-hazard safety (preventing overwrite of data
    // still being read by an async consumer kernel) is handled separately by
    // d_read_done_evt in write_pointer() / post_work().
    int half = (d_write_index >= d_half_nitems) ? 1 : 0;
    cudaEventSynchronize(d_dev_ready_evt[half]);
    cudaEventRecord(d_dev_ready_evt[half], producer_stream);
}

void cuda_buffer::mark_read_done(cudaStream_t consumer_stream)
{
    // Chain: make this stream depend on any previous consumer's read-done,
    // then record our own.  The lock serialises the CPU-side event management
    // so that fan-out consumers form a proper dependency chain rather than
    // racing on cudaEventRecord (which would let the last recorder silently
    // discard earlier consumers).  GPU kernels still run in parallel — only
    // the two API calls are serialized.
    std::lock_guard<std::mutex> lock(d_read_done_mutex);
    cudaStreamWaitEvent(consumer_stream, d_read_done_evt, 0);
    cudaEventRecord(d_read_done_evt, consumer_stream);
}

void cuda_buffer::wait_read_done(cudaStream_t stream)
{
    cudaStreamWaitEvent(stream, d_read_done_evt, 0);
}

void cuda_buffer::wait_device_ready(cudaStream_t consumer_stream)
{
    // Wait on both half-events.  A targeted single-half wait is possible but
    // not worthwhile: cudaStreamWaitEvent is a GPU-side dependency that never
    // blocks the CPU, and an already-completed event resolves instantly on the
    // GPU.  Waiting on both also avoids the cross-boundary bug that a
    // single-half strategy would reintroduce (a producer write that spans the
    // half-boundary only records one half's event).
    cudaStreamWaitEvent(consumer_stream, d_dev_ready_evt[0], 0);
    cudaStreamWaitEvent(consumer_stream, d_dev_ready_evt[1], 0);
}

void cuda_buffer::mark_host_ready(cudaStream_t copy_stream)
{
    int half = (d_write_index >= d_half_nitems) ? 1 : 0;
    cudaEventRecord(d_host_ready_evt[half], copy_stream);
}

void cuda_buffer::wait_host_ready()
{
    // Wait on both halves (CPU-blocking).  Both halves are needed because
    // a single D2H copy can span the half-boundary while mark_host_ready()
    // only records on the starting half.  Since both events are on d_stream,
    // the later one subsumes the earlier, so the effective cost is one sync.
    cudaEventSynchronize(d_host_ready_evt[0]);
    cudaEventSynchronize(d_host_ready_evt[1]);
}

buffer_sptr cuda_buffer::make_buffer(int nitems,
                                     size_t sizeof_item,
                                     uint64_t downstream_lcm_nitems,
                                     uint32_t downstream_max_out_mult,
                                     block_sptr link,
                                     block_sptr buf_owner)
{
    return buffer_sptr(new cuda_buffer(nitems, sizeof_item, downstream_lcm_nitems,
                                       downstream_max_out_mult, link, buf_owner));
}

} // namespace gr
