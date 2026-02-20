/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_to_stream_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>

namespace gr {
namespace cuda {

vector_to_stream::sptr vector_to_stream::make(size_t itemsize, size_t vlen)
{
    return gnuradio::make_block_sptr<vector_to_stream_impl>(itemsize, vlen);
}

vector_to_stream_impl::vector_to_stream_impl(size_t itemsize, size_t vlen)
    : gr::sync_interpolator("vector_to_stream",
                            io_signature::make(1, 1, itemsize * vlen, cuda_buffer::type),
                            io_signature::make(1, 1, itemsize, cuda_buffer::type),
                            vlen),
      d_itemsize(itemsize),
      d_vlen(vlen)
{
}

int vector_to_stream_impl::work(int noutput_items,
                                gr_vector_const_void_star& input_items,
                                gr_vector_void_star& output_items)
{
    // Wait for inputs
    gr::cuda::wait_for_inputs(this->detail(), d_stream);

    auto in = static_cast<const void*>(input_items[0]);
    auto out = static_cast<void*>(output_items[0]);

    // noutput_items is number of scalars.
    // Total bytes = noutput_items * itemsize.
    size_t total_bytes = noutput_items * d_itemsize;

    if (total_bytes > 0) {
        // Perform Device-to-Device copy.
        // This is equivalent to the CPU implementation which does:
        // memcpy(out, in, noutput_items * block_size);
        // Here, noutput_items is the number of scalars produced.
        // Since the input is a stream of vectors, copying this many bytes
        // effectively flattens the vectors into a stream of scalars.
        check_cuda_errors(cudaMemcpyAsync(out, 
                                          in, 
                                          total_bytes, 
                                          cudaMemcpyDeviceToDevice, 
                                          d_stream),
                          "vector_to_stream: cudaMemcpyAsync D2D", d_logger);
    }

    // Mark outputs ready
    gr::cuda::mark_outputs_ready(this->detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */

