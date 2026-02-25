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

#include <gnuradio/buffer_single_mapped.h>
#include <gnuradio/buffer_type.h>
#include <gnuradio/version.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <memory>

namespace gr {
namespace detail {
class device_vmm_ring;
class host_mmap_ring;
} // namespace detail

/*!
 * \brief GPU-aware circular buffer using CUDA VMM + mmap double-mapping.
 *
 * cuda_buffer inherits from buffer_single_mapped but replaces the linear
 * allocation with double-mapped virtual address ranges:
 *
 *   - **Device buffer**: CUDA VMM API (cuMemAddressReserve 2N, cuMemCreate N,
 *     cuMemMap twice).  Any contiguous span up to N items starting at any
 *     index in [0, N) is valid because the second N aliases the first.
 *
 *   - **Host buffer**: POSIX mmap with memfd_create for double-mapping, then
 *     cudaHostRegister to pin the pages for async DMA.
 *
 * Because the VA aliasing makes wrap-around transparent, compaction
 * (output_blocked_callback / input_blocked_callback) is never needed.
 * space_available() uses circular semantics, and the blocked-callback ready
 * checks return false so the scheduler never invokes compaction.
 *
 * \section sync Synchronization model
 *
 * Ring-buffered CUDA events (PIPELINE_DEPTH=4) for pipeline overlap:
 *
 * 1. **Device-ready events** -- producer records after writing device data;
 *    consumer GPU-waits before reading.
 * 2. **Host-ready events** -- recorded after async D2H copy; _read_pointer()
 *    CPU-waits before returning a host pointer.
 * 3. **Read-done event** -- tracks when ALL consumers have finished reading
 *    so the producer can safely overwrite.
 *
 * \section usage Usage from GPU blocks
 *
 * See cuda_block.h for the standard pattern, and cuda_block_helper.h for
 * wait_for_work() / mark_work_done() helpers.
 */
class GR_RUNTIME_API cuda_buffer : public buffer_single_mapped
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
     * \brief Return items the writer may produce (circular semantics).
     */
    int space_available() override;
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
     * \brief Always false -- double-mapping eliminates compaction.
     */
    bool input_blkd_cb_ready(int items_required, unsigned read_index) override;
    /*!
     * \brief Always false -- double-mapping eliminates compaction.
     */
    bool output_blkd_cb_ready(int output_multiple) override;
    /*!
     * \brief No-op -- compaction is never needed with double-mapping.
     */
    bool input_blocked_callback(int items_required, int items_avail,
                                unsigned read_index) override;
    /*!
     * \brief No-op -- compaction is never needed with double-mapping.
     */
    bool output_blocked_callback(int output_multiple, bool force) override;

    /*!
     * \brief Record a device-ready event after producing data on the GPU.
     */
    virtual void mark_device_ready(cudaStream_t producer_stream);
    /*!
     * \brief GPU-wait on all device-ready events before consuming GPU data.
     */
    virtual void wait_device_ready(cudaStream_t consumer_stream);
    /*!
     * \brief Record that a consumer has finished reading from this buffer.
     */
    virtual void mark_read_done(cudaStream_t consumer_stream);
    /*!
     * \brief GPU-wait until all consumers have finished reading.
     */
    virtual void wait_read_done(cudaStream_t stream);

    /*!
     * \brief Record a host-ready event after a D2H copy completes.
     */
    void mark_host_ready(cudaStream_t copy_stream);
    /*!
     * \brief CPU-wait until the D2H copy has landed in host memory.
     */
    void wait_host_ready();

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
     */
    bool do_allocate_buffer(size_t final_nitems, size_t sizeof_item) override;

    /*!
     * \brief Modular distance (a - b) mod d_bufsize.
     *
     * a = write_index, b = read_index.  Used by items_available() to
     * compute how many unread items sit in the buffer.
     *
     * Overrides buffer_single_mapped::index_sub which has a bug:
     *   int s = a - b; if (s < 0) s = d_bufsize - b;   // drops 'a'
     * The correct result when a < b (writer has wrapped but reader
     * hasn't) is d_bufsize - b + a.
     */
    unsigned index_sub(unsigned a, unsigned b) override
    {
        if (a >= b)
            return a - b;
        return d_bufsize - b + a;
    }

private:
    void post_work_h2d(unsigned write_index, unsigned tail, unsigned nitems);
    void post_work_d2h(unsigned write_index, unsigned tail, unsigned nitems);
    void post_work_d2d(unsigned write_index, unsigned tail, unsigned nitems);

    [[noreturn]] void throw_unexpected_transfer_type();

    /*!
     * CUDA Driver VMM double-mapped device memory ownership.
     */
    std::unique_ptr<detail::device_vmm_ring> d_device_ring;
    char* d_cuda_buf = nullptr;

    /*!
     * POSIX mmap double-mapped host memory ownership (incl. pinning).
     */
    std::unique_ptr<detail::host_mmap_ring> d_host_ring;

    cudaStream_t d_stream = nullptr; // dedicated stream for H2D / D2H DMA

    /*!
     * Event pipeline.
     * Ring-buffered CUDA events for pipelined overlap (PIPELINE_DEPTH deep).
     */
    static constexpr int PIPELINE_DEPTH = 2;
    cudaEvent_t d_dev_ready_evt[PIPELINE_DEPTH] = {};
    cudaEvent_t d_host_ready_evt[PIPELINE_DEPTH] = {};
    uint32_t d_dev_ready_next = 0;   // next slot index for device-ready ring
    uint32_t d_host_ready_next = 0;  // next slot index for host-ready ring

    cudaEvent_t d_read_done_evt = nullptr; // single event: all consumers done
    std::mutex d_read_done_mutex;          // serialises mark_read_done calls

    cuda_buffer(int nitems,
                size_t sizeof_item,
                uint64_t downstream_lcm_nitems,
                uint32_t downstream_max_out_mult,
                block_sptr link,
                block_sptr buf_owner);
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BUFFER_H */
