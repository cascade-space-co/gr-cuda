/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "interleave.cuh"
#include "streams_to_vector_impl.h"
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>


namespace gr {
namespace cuda {

streams_to_vector::sptr streams_to_vector::make(size_t itemsize, size_t num_streams)
{
    return gnuradio::make_block_sptr<streams_to_vector_impl>(itemsize, num_streams);
}

streams_to_vector_impl::streams_to_vector_impl(size_t itemsize, size_t num_streams)
    : gr::sync_block(
          "streams_to_vector",
          io_signature::make(num_streams, num_streams, itemsize, cuda_buffer::type),
          io_signature::make(1, 1, itemsize * num_streams, cuda_buffer::type)),
      d_itemsize(itemsize),
      d_num_streams(num_streams)
{
    get_interleave_block_and_grid(&d_min_grid_size, &d_block_size);
    check_cuda_errors(
        cudaMalloc((void**)&d_input_ptrs_dev, sizeof(void*) * d_num_streams),
        "streams_to_vector: cudaMalloc input_ptrs",
        d_logger);
}

streams_to_vector_impl::~streams_to_vector_impl()
{
    if (d_input_ptrs_dev)
        cudaFree(d_input_ptrs_dev);
}

int streams_to_vector_impl::work(int noutput_items,
                                 gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    gr::cuda::wait_for_work(detail(), d_stream);

    auto out = static_cast<void*>(output_items[0]);

    // Copy input pointers to device
    // input_items is vector<const void*>
    check_cuda_errors(cudaMemcpyAsync(d_input_ptrs_dev,
                                      input_items.data(),
                                      sizeof(void*) * d_num_streams,
                                      cudaMemcpyHostToDevice,
                                      d_stream),
                      "streams_to_vector: cudaMemcpyAsync H2D ptrs",
                      d_logger);

    // noutput_items is number of vectors produced.
    // This kernel interleaves data from N input streams into one output vector stream.
    // Equivalent CPU logic:
    // for (i = 0; i < noutput_items; i++) {
    //     for (j = 0; j < nstreams; j++) {
    //         memcpy(out, input_items[j], itemsize);
    //         ...
    //     }
    // }
    // The GPU kernel performs this parallelized across all items.
    int total_scalars = noutput_items * d_num_streams;
    int gridSize = (total_scalars + d_block_size - 1) / d_block_size;

    exec_interleave((const void**)d_input_ptrs_dev,
                    out,
                    d_num_streams,
                    d_itemsize,
                    noutput_items,
                    gridSize,
                    d_block_size,
                    d_stream);

    gr::cuda::mark_work_done(detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
