/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_BLOCK_HELPER_H
#define INCLUDED_GR_CUDA_BLOCK_HELPER_H

#include <gnuradio/block.h>
#include <gnuradio/block_detail.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <memory>

namespace gr {
namespace cuda {

/*!
 * \brief Add GPU-side dependencies before launching kernels.
 *
 * Adds all necessary GPU-side dependencies so that any work enqueued on
 * \p stream after this call will not execute until:
 *   1. All input buffers' data is ready on the device (device-ready events).
 *   2. All output buffers are safe to write (read-done events) — i.e. every
 *      downstream consumer and in-flight D2H copy has finished reading.
 *
 * These are GPU-side waits (cudaStreamWaitEvent) and do **not** block the
 * calling CPU thread.  This is the primary mechanism that allows H2D and D2H
 * transfers to overlap on dual-copy-engine GPUs.
 *
 * \param detail The block's detail pointer (e.g. call with detail())
 * \param stream The CUDA stream to synchronize
 *
 * \sa mark_work_done() for the post-kernel counterpart.
 */
inline void wait_for_work(const gr::block_detail_sptr& detail, cudaStream_t stream)
{
    // 1. Input buffers: wait for upstream data to be ready on the device.
    int ninputs = detail->ninputs();
    for (int i = 0; i < ninputs; i++) {
        auto buf = detail->input(i)->buffer();
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->wait_device_ready(stream);
        }
    }

    // 2. Output buffers: ensure downstream consumers / D2H copies have
    //    finished reading before our kernel writes new data.
    int noutputs = detail->noutputs();
    for (int i = 0; i < noutputs; i++) {
        auto buf = detail->output(i);
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->wait_read_done(stream);
        }
    }
}

/*!
 * \brief Mark all output CUDA buffers as ready and all input CUDA buffers as
 *        consumed.  Call this after launching all GPU work for the current
 *        work() invocation.
 *
 * For outputs: records the DEV_READY event so downstream consumers know data is
 * available.
 *
 * For inputs: records the READ_DONE event so the upstream producer knows it is
 * safe to overwrite the buffer.  This eliminates the consumer-to-producer data
 * hazard without requiring any per-block code changes.
 *
 * Input-side operations (mark_read_done) are non-blocking GPU event records.
 * Output-side operations (mark_device_ready) may briefly block the CPU via
 * cudaEventSynchronize for backpressure when the GPU falls behind.
 *
 * \param detail The block's detail pointer (e.g. call with detail())
 * \param stream The CUDA stream that finished both reading inputs and producing
 *               outputs
 */
inline void mark_work_done(const gr::block_detail_sptr& detail, cudaStream_t stream)
{
    // 1. Signal output buffers as ready for downstream consumers
    int noutputs = detail->noutputs();
    for (int i = 0; i < noutputs; i++) {
        auto buf = detail->output(i); // Returns buffer_sptr directly
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->mark_device_ready(stream);
        }
    }

    // 2. Signal input buffers that this consumer is done reading.
    //    The upstream producer waits on d_read_done_evt before overwriting.
    int ninputs = detail->ninputs();
    for (int i = 0; i < ninputs; i++) {
        auto buf = detail->input(i)->buffer();
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->mark_read_done(stream);
        }
    }
}

} /* namespace cuda */
} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BLOCK_HELPER_H */
