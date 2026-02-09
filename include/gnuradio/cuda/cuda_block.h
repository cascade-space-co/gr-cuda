/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef _INCLUDED_CUDA_BLOCK_H
#define _INCLUDED_CUDA_BLOCK_H

#include <gnuradio/cuda/cuda_error.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gr {

/*!
 * \brief Base class for GPU blocks using cuda_buffer.
 *
 * Provides a non-blocking CUDA stream and optional kernel launch configuration
 * members.  All GPU blocks should inherit from this class (via multiple
 * inheritance alongside the GR block base) to get consistent stream lifecycle
 * management.
 *
 * \section provided Protected members
 *
 *   - \c d_stream          — non-blocking CUDA stream, created in the
 *                            constructor and destroyed in the destructor.
 *   - \c d_min_grid_size   — minimum grid size for full GPU occupancy
 *                            (populate via cudaOccupancyMaxPotentialBlockSize).
 *   - \c d_block_size      — thread block size for kernel launches
 *                            (populate via cudaOccupancyMaxPotentialBlockSize).
 *
 * \section pattern Standard GPU block pattern
 *
 * \code
 * // my_block_impl.h
 * #include <gnuradio/cuda/cuda_block.h>
 *
 * class my_block_impl : public my_block, public cuda_block {
 *     // ...
 * };
 *
 * // my_block_impl.cc
 * #include <gnuradio/cuda/cuda_buffer.h>
 * #include <gnuradio/cuda/cuda_block_helper.h>
 *
 * my_block_impl::my_block_impl(...)
 *     : gr::sync_block("my_block",
 *           io_signature::make(1, 1, sizeof(float), cuda_buffer::type),
 *           io_signature::make(1, 1, sizeof(float), cuda_buffer::type))
 * {
 *     // d_stream is created automatically by cuda_block.
 *     // Optionally populate launch config:
 *     // get_my_kernel_block_and_grid(&d_min_grid_size, &d_block_size);
 * }
 *
 * int my_block_impl::work(int noutput_items,
 *                         gr_vector_const_void_star& input_items,
 *                         gr_vector_void_star& output_items)
 * {
 *     // 1. Wait for upstream data to be ready on the GPU
 *     gr::cuda::wait_for_inputs(this->detail(), d_stream);
 *
 *     // 2. Launch GPU kernels on d_stream
 *     auto in  = reinterpret_cast<const float*>(input_items[0]);
 *     auto out = reinterpret_cast<float*>(output_items[0]);
 *     my_kernel<<<grid, block, 0, d_stream>>>(in, out, noutput_items);
 *
 *     // 3. Signal outputs ready and inputs consumed
 *     gr::cuda::mark_outputs_ready(this->detail(), d_stream);
 *     return noutput_items;
 * }
 * \endcode
 *
 * The two helper calls (from cuda_block_helper.h) handle all event
 * bookkeeping automatically:
 *   - wait_for_inputs() adds GPU-side waits on each input buffer's
 *     device-ready event so the kernel does not read stale data.
 *   - mark_outputs_ready() records device-ready events on each output
 *     buffer (signalling downstream) AND records read-done events on
 *     each input buffer (signalling the upstream producer that this
 *     consumer is done).
 *
 * \sa cuda_buffer for the underlying synchronization model.
 * \sa cuda_block_helper.h for wait_for_inputs() and mark_outputs_ready().
 */
class cuda_block
{
protected:
    cudaStream_t d_stream;
    int d_min_grid_size;
    int d_block_size;

public:
    cuda_block()
    {
        check_cuda_errors(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));
    }

    virtual ~cuda_block()
    {
        if (d_stream) {
            cudaStreamDestroy(d_stream);
        }
    }
};

} // namespace gr
#endif
