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
#include <memory>
#include <mutex>

namespace gr {
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
 * \section usage Usage from GPU blocks
 *
 * See cuda_block.h for the standard pattern, and cuda_block_helper.h for
 * wait_for_work() / mark_work_done() helpers.
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
     * \brief Transfer data between host and device after general_work().
     */
    void post_work(int nitems) override;
    /*!
     * \brief Pointer into the 2N region where the producer should write.
     */
    void* write_pointer() override;
    /*!
     * \brief Pointer into the 2N region where the consumer should read.
     */
    const void* _read_pointer(unsigned int read_index) override;

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

private:
    void post_work_h2d(unsigned write_index, unsigned tail, unsigned nitems);
    void post_work_d2h(unsigned write_index, unsigned tail, unsigned nitems);
    void post_work_d2d(unsigned write_index, unsigned tail, unsigned nitems);

    [[noreturn]] void throw_unexpected_transfer_type();

    void mark_host_ready(cudaStream_t copy_stream);
    void wait_host_ready();

    std::unique_ptr<detail::device_vmm_ring> d_device_ring;
    char* d_cuda_buf = nullptr;

    std::unique_ptr<detail::host_mmap_ring> d_host_ring;

    cudaStream_t d_stream = nullptr;

    cudaEvent_t d_dev_ready_evt = nullptr;
    cudaEvent_t d_host_ready_evt = nullptr;
    cudaEvent_t d_read_done_evt = nullptr;
    std::mutex d_read_done_mutex;

    cuda_buffer(int nitems,
                size_t sizeof_item,
                uint64_t downstream_lcm_nitems,
                uint32_t downstream_max_out_mult,
                block_sptr link);
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BUFFER_H */
