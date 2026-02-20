/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_to_streams_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>

// Forward decls from interleave.cu
void exec_deinterleave(const void* in,
                       void** outputs,
                       int num_streams,
                       int itemsize,
                       int N,
                       int grid_size,
                       int block_size,
                       cudaStream_t stream);

void get_deinterleave_block_and_grid(int* minGrid, int* minBlock);

namespace gr {
namespace cuda {

vector_to_streams::sptr vector_to_streams::make(size_t itemsize, size_t num_streams)
{
    return gnuradio::make_block_sptr<vector_to_streams_impl>(itemsize, num_streams);
}

vector_to_streams_impl::vector_to_streams_impl(size_t itemsize, size_t num_streams)
    : gr::sync_block("vector_to_streams",
                     io_signature::make(1, 1, itemsize * num_streams, cuda_buffer::type),
                     io_signature::make(num_streams, num_streams, itemsize, cuda_buffer::type)),
      d_itemsize(itemsize),
      d_num_streams(num_streams)
{
    get_deinterleave_block_and_grid(&d_min_grid_size, &d_block_size);
    check_cuda_errors(cudaMalloc((void**)&d_output_ptrs_dev, sizeof(void*) * d_num_streams),
                      "vector_to_streams: cudaMalloc output_ptrs", d_logger);
}

vector_to_streams_impl::~vector_to_streams_impl()
{
    if (d_output_ptrs_dev) cudaFree(d_output_ptrs_dev);
}

int vector_to_streams_impl::work(int noutput_items,
                                 gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    gr::cuda::wait_for_inputs(this->detail(), d_stream);

    auto in = static_cast<const void*>(input_items[0]);

    // Copy output pointers to device
    // output_items is vector<void*>
    check_cuda_errors(cudaMemcpyAsync(d_output_ptrs_dev, 
                                      output_items.data(), 
                                      sizeof(void*) * d_num_streams, 
                                      cudaMemcpyHostToDevice, 
                                      d_stream),
                      "vector_to_streams: cudaMemcpyAsync H2D ptrs", d_logger);

    // noutput_items = number of scalars produced per stream (which equals number of input vectors consumed).
    // This kernel deinterleaves data from one input vector stream into N output scalar streams.
    // Equivalent CPU logic:
    // for (i = 0; i < noutput_items; i++) {
    //     for (j = 0; j < nstreams; j++) {
    //         memcpy(output_items[j], in, itemsize);
    //         ...
    //     }
    // }
    // The GPU kernel performs this parallelized across all items.
    int total_scalars = noutput_items * d_num_streams;
    int gridSize = (total_scalars + d_block_size - 1) / d_block_size;

    exec_deinterleave(in,
                      (void**)d_output_ptrs_dev,
                      d_num_streams,
                      d_itemsize,
                      noutput_items,
                      gridSize,
                      d_block_size,
                      d_stream);

    gr::cuda::mark_outputs_ready(this->detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */

