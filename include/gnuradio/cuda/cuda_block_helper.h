/* -*- c++ -*- */
/*
 * Copyright 2024
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_BLOCK_H
#define INCLUDED_GR_CUDA_BLOCK_H

#include <gnuradio/block.h>
#include <gnuradio/block_detail.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <memory>

namespace gr {
namespace cuda {

/*!
 * \brief Wait for inputs using the block detail.
 *
 * Checks all input ports of the block. If an input buffer is a cuda_buffer,
 * it instructs the provided stream to wait for the buffer's DEV_READY event.
 *
 * \param detail The block's detail pointer (e.g. call with this->detail())
 * \param stream The CUDA stream to synchronize with
 */
inline void wait_for_inputs(gr::block_detail_sptr detail, cudaStream_t stream)
{
    int ninputs = detail->ninputs();
    for (int i = 0; i < ninputs; i++) {
        auto buf = detail->input(i)->buffer();
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->wait_device_ready(stream);
        }
    }
}

/*!
 * \brief Mark all output CUDA buffers as ready.
 *
 * Checks all output ports of the block. If an output buffer is a cuda_buffer,
 * it records the DEV_READY event on the provided stream.
 *
 * \param detail The block's detail pointer (e.g. call with this->detail())
 * \param stream The CUDA stream that finished producing data
 */
inline void mark_outputs_ready(gr::block_detail_sptr detail, cudaStream_t stream)
{
    int noutputs = detail->noutputs();
    for (int i = 0; i < noutputs; i++) {
        auto buf = detail->output(i); // Returns buffer_sptr directly
        auto cuda_buf = std::dynamic_pointer_cast<gr::cuda_buffer>(buf);
        if (cuda_buf) {
            cuda_buf->mark_device_ready(stream);
        }
    }
}

} /* namespace cuda */
} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BLOCK_H */

