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

#ifndef INCLUDED_GR_CUDA_BUFFER_H
#define INCLUDED_GR_CUDA_BUFFER_H

#include <gnuradio/buffer_double_mapped.h>
#include <gnuradio/buffer_type.h>
#include <gnuradio/version.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <memory>
#include <mutex>

namespace gr {
class tpb_detail;
namespace detail {
class device_vmm_ring;
class host_mmap_ring;
} // namespace detail

/*!
 * \brief GPU-aware circular buffer using CUDA VMM + mmap double-mapping.
 *
 * cuda_buffer inherits from buffer_double_mapped, using the deferred-
 * allocation constructor so that it can provide its own double-mapped
 * memory for both host and device:
 *
 *   - **Device buffer**: CUDA VMM API (cuMemAddressReserve 2N, cuMemCreate N,
 *     cuMemMap twice).  Any contiguous span up to N items starting at any
 *     index in [0, N) is valid because the second N aliases the first.
 *
 *   - **Host buffer**: POSIX mmap with memfd_create for double-mapping, then
 *     cudaHostRegister to pin the pages for async DMA.
 *
 * Because both sides are double-mapped, compaction is never needed and
 * space_available() / index_sub() from buffer_double_mapped work correctly.
 *
 * \section sync Synchronization model
 *
 * Three CUDA events coordinate producer/consumer overlap:
 *
 * 1. **Device-ready** -- producer records after writing device data;
 *    consumer GPU-waits before reading.
 * 2. **Host-ready** -- recorded after async D2H copy; _read_pointer()
 *    CPU-waits before returning a host pointer.
 * 3. **Read-done** -- tracks when ALL consumers have finished reading
 *    so the producer can safely overwrite.
 *
 * \section blocking Blocking vs spin-wait synchronization
 *
 * By default, CPU-side event waits (cudaEventSynchronize) use blocking
 * sync (cudaEventBlockingSync), which puts the thread to sleep and
 * frees the CPU core while waiting. To switch to spin-polling for
 * lowest latency at the cost of CPU usage, set in
 * `gnuradio-config-info --userprefsdir`
 * (usually ~/.config/gnuradio/config.conf for linux):
 *
 * @code
 * [cuda_buffer]
 * blocking_sync = false
 * @endcode
 *
 * \section autosync Automatic synchronization
 *
 * cuda_buffer inserts GPU-side event waits/records at three
 * hooks in the scheduler loop.  No per-block sync code is
 * needed for C++ blocks that inherit from cuda_block.
 *
 * @code
 * Hook                  | Action                    | When
 * ----------------------|---------------------------|-------------------
 * space_available()     | query d_dev_ready_evt     | before prod offered space
 * write_pointer()       | wait_read_done(prod)      | before prod writes
 * post_work()           | mark_device_ready(prod)   | after prod writes
 * post_work()           | wait_device_ready(cons)   | after prod writes
 * _read_pointer()       | wait_host_ready (D2H)     | before cons reads host
 * update_read_pointer() | mark_read_done(cons) [*]  | after cons reads
 *
 * [*] Implemented in cuda_buffer_reader, not cuda_buffer.
 * @endcode
 *
 * space_available() uses cudaEventQuery (non-blocking) with a
 * cudaLaunchHostFunc callback to wake the scheduler thread when
 * the GPU event fires, eliminating CPU-blocking on the producer
 * side.  D2H consumer reads still use cudaEventSynchronize in
 * _read_pointer() since the CPU consumer inherently must wait
 * for host data to land.
 *
 * wait_device_ready is placed in post_work() rather than
 * _read_pointer() because the GR scheduler guarantees consumers
 * cannot run until post_work completes.  cudaStreamWaitEvent
 * is a GPU-side dependency, so it does not matter which CPU
 * thread inserts it.
 *
 * For H2D/D2H edges where one side is a CPU block, the stream resolves
 * to nullptr and the auto-sync is skipped. The DMA path in
 * post_work_h2d / post_work_d2h handles those cases using the buffer's
 * own CUDA stream.
 *
 * \sa cuda_block.h for the standard GPU block pattern.
 */
class GR_RUNTIME_API cuda_buffer : public buffer_double_mapped
{
public:
    static buffer_type type;

    /*!
     * Requires the fan-out fix for custom buffers introduced in:
     * https://github.com/gnuradio/gnuradio/pull/8029
     */
#if (GR_VERSION_API == 10 && GR_VERSION_MINOR > 12) || (GR_VERSION_API >= 11)
    buffer_type get_buffer_type() const override { return type; }
#else
#warning "GNU Radio <= 3.10.12 detected: custom buffer fan-out is broken "    \
         "in this version. Flowgraphs with fan-out on cuda_buffer edges "     \
         "will deadlock (including QA tests). Upgrade to >3.10.12 for "       \
         "full support. See https://github.com/gnuradio/gnuradio/pull/8029"
#endif

    ~cuda_buffer() override;

    /*!
     * \brief Non-blocking space check with GPU event backpressure.
     *
     * Before returning available space, queries d_dev_ready_evt to
     * verify the GPU has consumed the previous batch.  If the event
     * is not ready, registers a cudaLaunchHostFunc callback that
     * wakes the scheduler thread and returns 0 (triggering BLKD_OUT).
     */
    int space_available() override;

    /*!
     * \brief Transfer data between host and device after general_work().
     *
     * Also handles two auto-sync operations (see \ref autosync):
     * marks device data ready, and makes consumer streams wait for it.
     */
    void post_work(int nitems) override;
    /*!
     * \brief Pointer into the 2N region where the producer should write.
     *
     * For D2D/D2H edges, inserts a GPU-side wait on the read-done event
     * so the producer kernel does not overwrite data still being
     * consumed (see \ref autosync).
     */
    void* write_pointer() override;
    /*!
     * \brief Pointer into the 2N region where the consumer should read.
     */
    const void* _read_pointer(unsigned int read_index) override;

    /*!
     * \brief Create a cuda_buffer_reader instead of the default reader.
     *
     * The custom reader overrides update_read_pointer() to automatically
     * record read-done events (see \ref autosync).
     */
    buffer_reader_sptr create_reader(buffer_sptr buf,
                                     int nzero_preload,
                                     block_sptr link,
                                     int delay) override;

    /*!
     * \brief Register the producer's CUDA stream explicitly.
     *
     * Used by Python GPU blocks that cannot be discovered via
     * dynamic_cast<cuda_block*>.  If set, resolve_producer_stream()
     * returns this stream instead of attempting the cast.
     */
    void set_producer_stream(cudaStream_t s);

    /*!
     * \brief Record a device-ready event after producing data on the GPU.
     */
    void mark_device_ready(cudaStream_t producer_stream);
    /*!
     * \brief GPU-wait for device-ready before consuming GPU data.
     */
    void wait_device_ready(cudaStream_t consumer_stream);
    /*!
     * \brief Record that a consumer has finished reading from this buffer.
     *
     * Thread-safe: serialised by d_read_done_mutex to support fan-out
     * (multiple consumers chaining through a single event).
     */
    void mark_read_done(cudaStream_t consumer_stream);
    /*!
     * \brief GPU-wait until all consumers have finished reading.
     */
    void wait_read_done(cudaStream_t producer_stream);

    /*!
     * \brief Factory method used by GNU Radio's buffer allocation machinery.
     */
    static buffer_sptr make_buffer(int nitems,
                                   size_t sizeof_item,
                                   uint64_t downstream_lcm_nitems,
                                   uint32_t downstream_max_out_mult,
                                   block_sptr link,
                                   block_sptr buf_owner);

protected:
    /*!
     * \brief Allocate double-mapped host + device circular buffers.
     *
     * Replaces the vmcircbuf path with CUDA VMM for device memory and
     * POSIX mmap + cudaHostRegister for pinned host memory.
     */
    bool allocate_buffer(int nitems) override;

    /*!
     * \brief Allocate host ring when the transfer type requires it.
     *
     * Called exactly once by gr::buffer::set_transfer_type().
     * H2D and D2H edges get a pinned, double-mapped host ring;
     * D2D edges skip host allocation entirely.
     */
    void on_transfer_type_set(const transfer_type& type) override;

private:
    void post_work_h2d(unsigned write_index, unsigned tail, unsigned nitems);
    void post_work_d2h(unsigned write_index, unsigned tail, unsigned nitems);
    void post_work_d2d(unsigned write_index, unsigned tail, unsigned nitems);

    [[noreturn]] void throw_unexpected_transfer_type();

    void mark_host_ready(cudaStream_t copy_stream);
    void wait_host_ready();

    cudaStream_t resolve_producer_stream();

    std::unique_ptr<detail::device_vmm_ring> d_device_ring;
    char* d_cuda_buf = nullptr;

    std::unique_ptr<detail::host_mmap_ring> d_host_ring;
    size_t d_aligned_bytes = 0;

    cudaStream_t d_stream = nullptr;

    cudaStream_t d_producer_stream = nullptr;
    bool d_producer_stream_resolved = false;

    cudaEvent_t d_dev_ready_evt = nullptr;
    cudaEvent_t d_host_ready_evt = nullptr;
    cudaEvent_t d_read_done_evt = nullptr;
    std::mutex d_read_done_mutex;

    cudaStream_t d_notify_stream = nullptr;
    std::atomic<bool> d_dev_notify_pending{ false };

public:
    struct notify_ctx {
        std::atomic<bool> alive{ true };
        tpb_detail* tpb = nullptr;
    };

private:
    std::shared_ptr<notify_ctx> d_notify_ctx;

    cuda_buffer(int nitems,
                size_t sizeof_item,
                uint64_t downstream_lcm_nitems,
                uint32_t downstream_max_out_mult,
                block_sptr link);
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BUFFER_H */
