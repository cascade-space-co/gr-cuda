/* -*- c++ -*- */
/*
 * Copyright 2021 BlackLynx, Inc.
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_CUDA_H
#define INCLUDED_GR_CUDA_H

#include <gnuradio/buffer_single_mapped.h>
#include <gnuradio/buffer_type.h>
#include <gnuradio/version.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mutex>

namespace gr {

/*!
 * \brief GPU-aware circular buffer with double-buffered event synchronization.
 *
 * cuda_buffer is a buffer_single_mapped subclass that manages a matched pair of
 * buffers: a pinned host buffer (cudaMallocHost) and a device buffer
 * (cudaMalloc).  The buffer's transfer_type determines data flow:
 *
 *   - HOST_TO_DEVICE: CPU writes to host buffer, post_work() copies H2D.
 *   - DEVICE_TO_HOST: GPU writes to device buffer, post_work() copies D2H.
 *   - DEVICE_TO_DEVICE: GPU reads/writes device buffer directly (no copy).
 *
 * \section sync Synchronization model
 *
 * Three event-based mechanisms ensure race-free asynchronous execution across
 * independent CUDA streams, while preserving GPU pipeline overlap:
 *
 * 1. **Device-ready events (d_dev_ready_evt[2])**
 *    Double-buffered: one event per buffer half.  The producer records an event
 *    after writing; the consumer's stream GPU-waits on it before reading.
 *    Having two halves prevents the producer from overwriting an event that the
 *    consumer hasn't observed yet (which a single event would allow when the
 *    producer's CPU thread runs ahead of the GPU).  mark_device_ready() also
 *    does a CPU-side cudaEventSynchronize() for backpressure, preventing the
 *    CPU from outrunning the GPU.
 *
 * 2. **Host-ready events (d_host_ready_evt[2])**
 *    Same double-buffered scheme, but for D2H copies.  Recorded after the async
 *    D2H memcpy completes; _read_pointer() CPU-waits on both halves before
 *    returning a host pointer to the downstream CPU block.
 *
 * 3. **Read-done event (d_read_done_evt)**
 *    A single event that tracks when ALL consumers have finished reading from
 *    the device buffer.  Uses event chaining (cudaStreamWaitEvent + record) so
 *    fan-out consumers run in parallel but the event captures the latest
 *    completion.  The producer waits on this event before overwriting:
 *      - H2D buffers: GPU-side wait in post_work() (does not block CPU).
 *      - D2H buffers: CPU-side wait in write_pointer(), plus the D2H copy
 *        itself records d_read_done_evt so the upstream producer cannot
 *        overwrite device data while an async D2H copy is still reading it.
 *      - D2D buffers: CPU-side wait in write_pointer().
 *
 * \section usage Usage from GPU blocks
 *
 * See cuda_block.h for the standard pattern for writing GPU blocks, and
 * cuda_block_helper.h for the wait_for_inputs() / mark_outputs_ready()
 * helper functions that handle all event bookkeeping automatically.
 *
 * Blocks that do NOT use these helpers (e.g. CPU-only blocks connected via
 * cuda_buffer) still work correctly: the synchronization in post_work(),
 * write_pointer(), and _read_pointer() handles the CPU-side waits.
 *
 */
class GR_RUNTIME_API cuda_buffer : public buffer_single_mapped
{
public:
    static buffer_type type;

    // Requires the fan-out fix for custom buffers introduced in:
    // https://github.com/gnuradio/gnuradio/pull/8029
#if (GR_VERSION_API == 10 && GR_VERSION_MINOR > 12) || (GR_VERSION_API >= 11)
    buffer_type get_buffer_type() const override { return type; }
#else
#warning "GNU Radio <= 3.10.12 detected: custom buffer fan-out is broken "    \
         "in this version. Flowgraphs with fan-out on cuda_buffer edges "     \
         "will deadlock (including QA tests). Upgrade to >3.10.12 for "       \
         "full support. See https://github.com/gnuradio/gnuradio/pull/8029"
#endif

    virtual ~cuda_buffer();

    /*!
     * \brief Handle post-general_work() cleanup and data transfer
     *
     * Called directly after call to general_work() completes and
     * is used for data transfer (and perhaps other administrative
     * activities)
     *
     * \param nitems is the number of items produced by the general_work() function.
     */
    virtual void post_work(int nitems);

    /*!
     * \brief Mark the device buffer as ready (producer done)
     * \param producer_stream Stream that produced the data
     */
    void mark_device_ready(cudaStream_t producer_stream);

    /*!
     * \brief Wait for the device buffer to be ready (consumer waits)
     * \param consumer_stream Stream that will consume the data
     */
    void wait_device_ready(cudaStream_t consumer_stream);

    /*!
     * \brief Mark the host buffer as ready (copy done)
     * \param copy_stream Stream that performed the copy
     */
    void mark_host_ready(cudaStream_t copy_stream);

    /*!
     * \brief Wait for the host buffer to be ready (host reads)
     */
    void wait_host_ready();

    /*!
     * \brief Mark this buffer's device data as consumed by a reader.
     *
     * Called by consumers after their GPU work that reads from this buffer.
     * Uses event chaining so that the single event captures the completion
     * of all consumers (even in fan-out), without serializing their kernels.
     *
     * \param consumer_stream Stream that finished reading from this buffer
     */
    void mark_read_done(cudaStream_t consumer_stream);

    /*!
     * \brief Do actual buffer allocation. Inherited from buffer_single_mapped.
     */
    bool do_allocate_buffer(size_t final_nitems, size_t sizeof_item);

    /*!
     * \brief Return a pointer to the write buffer depending on the context
     */
    virtual void* write_pointer();

    /*!
     * \brief return pointer to read buffer depending on the context
     *
     * The return value points to at least items_available() items.
     */
    virtual const void* _read_pointer(unsigned int read_index);

    /*!
     * \brief Callback function that the scheduler will call when it determines
     * that the input is blocked. Override this function if needed.
     */
    bool input_blocked_callback(int items_required, int items_avail, unsigned read_index);

    /*!
     * \brief Callback function that the scheduler will call when it determines
     * that the output is blocked
     */
    bool output_blocked_callback(int output_multiple, bool force);

    /*!
     * \brief Creates a new cuda object
     *
     * \param nitems
     * \param sizeof_item
     * \param downstream_lcm_nitems
     * \param link
     * \param buf_owner
     *
     * \return pointer to buffer base class
     */
    static buffer_sptr make_buffer(int nitems,
                                   size_t sizeof_item,
                                   uint64_t downstream_lcm_nitems,
                                   uint32_t downstream_max_out_mult,
                                   block_sptr link,
                                   block_sptr buf_owner);

private:
    // Internal copy functions and their std::function wrappers for use by
    // the blocked-callback realignment logic (input_blocked_callback_logic /
    // output_blocked_callback_logic).
    void* cuda_memcpy(void* dest, const void* src, std::size_t count);
    void* cuda_memmove(void* dest, const void* src, std::size_t count);
    mem_func_t f_cuda_memcpy;
    mem_func_t f_cuda_memmove;

    cudaStream_t d_stream;

    /*!
     * \brief Drain all GPU work touching this buffer before a destructive
     *        operation (e.g. buffer realignment).
     */
    void sync_all_gpu_work();

    //! Log a CUDA error and throw std::runtime_error.  Never returns.
    [[noreturn]] void throw_cuda_error(const char* context, cudaError_t rc);

    //! Throw for an unhandled transfer_type in a switch.  Never returns.
    [[noreturn]] void throw_unexpected_transfer_type();

    // Double-buffered events: one per buffer half for pipeline overlap.
    // Each half of the circular buffer tracks its own device-ready and
    // host-ready state, preventing event overwrite when the producer
    // runs ahead of the consumer.
    static constexpr int NUM_HALF_EVENTS = 2;
    cudaEvent_t d_dev_ready_evt[NUM_HALF_EVENTS];
    cudaEvent_t d_host_ready_evt[NUM_HALF_EVENTS];
    size_t d_half_nitems;  // Boundary index between buffer halves

    // Consumer-to-producer sync: tracks when ALL consumers have finished
    // reading from the device buffer, so the producer can safely overwrite.
    cudaEvent_t d_read_done_evt;
    std::mutex d_read_done_mutex; // Serialises mark_read_done() for fan-out safety

    char* d_cuda_buf; // CUDA buffer

    // Scratch buffer management for internal copies
    // Used in cuda_memmove to handle overlapping memory regions safely.
    // We persist this buffer to avoid expensive cudaMalloc/Free calls on every move.
    void* d_temp_buffer;
    size_t d_temp_buffer_size;

    /*!
     * \brief constructor is private.  Use gr_make_buffer to create instances.
     *
     * Allocate a buffer that holds at least \p nitems of size \p sizeof_item.
     *
     * \param nitems is the minimum number of items the buffer will hold.
     * \param sizeof_item is the size of an item in bytes.
     * \param downstream_lcm_nitems is the least common multiple of the items to
     *                              read by downstream blocks
     * \param downstream_max_out_mult is the maximum output multiple of all 
     *                                downstream blocks
     * \param link is the block that writes to this buffer.
     * \param buf_owner if the block that owns the buffer which may or may not
     *                  be the same as the block that writes to this buffer
     *
     * The total size of the buffer will be rounded up to a system
     * dependent boundary.  This is typically the system page size, but
     * under MS windows is 64KB.
     */
    cuda_buffer(int nitems,
                size_t sizeof_item,
                uint64_t downstream_lcm_nitems,
                uint32_t downstream_max_out_mult,
                block_sptr link,
                block_sptr buf_owner);
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_H */
