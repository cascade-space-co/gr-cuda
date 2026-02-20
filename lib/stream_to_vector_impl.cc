/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "stream_to_vector_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>

namespace gr {
namespace cuda {

stream_to_vector::sptr stream_to_vector::make(size_t itemsize, size_t vlen)
{
    return gnuradio::make_block_sptr<stream_to_vector_impl>(itemsize, vlen);
}

stream_to_vector_impl::stream_to_vector_impl(size_t itemsize, size_t vlen)
    : gr::sync_decimator("stream_to_vector",
                         io_signature::make(1, 1, itemsize, cuda_buffer::type),
                         io_signature::make(1, 1, itemsize * vlen, cuda_buffer::type),
                         vlen),
      d_itemsize(itemsize),
      d_vlen(vlen)
{
}

int stream_to_vector_impl::work(int noutput_items,
                                gr_vector_const_void_star& input_items,
                                gr_vector_void_star& output_items)
{
    // Wait for inputs
    gr::cuda::wait_for_inputs(this->detail(), d_stream);

    auto in = static_cast<const void*>(input_items[0]);
    auto out = static_cast<void*>(output_items[0]);

    // noutput_items is number of vectors.
    // Total bytes = noutput_items * vlen * itemsize.
    size_t total_bytes = noutput_items * d_vlen * d_itemsize;

    if (total_bytes > 0) {
        // Perform Device-to-Device copy.
        // This is equivalent to the CPU implementation which does:
        // memcpy(out, in, noutput_items * block_size);
        // where block_size is the size of the output vector (itemsize * vlen).
        // Since the input is a stream of scalars, copying N vectors worth of bytes
        // effectively groups the scalars into vectors in the output buffer.
        
        check_cuda_errors(cudaMemcpyAsync(out, 
                                          in, 
                                          total_bytes, 
                                          cudaMemcpyDeviceToDevice, 
                                          d_stream),
                          "stream_to_vector: cudaMemcpyAsync D2D", d_logger);
    }

    // Mark outputs ready
    gr::cuda::mark_outputs_ready(this->detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */

