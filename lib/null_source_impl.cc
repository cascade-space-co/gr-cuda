/* -*- c++ -*- */
/*
 * Copyright 2025 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "null_source_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

null_source::sptr null_source::make(size_t sizeof_stream_item, size_t num_outputs,
                                    bool memset)
{
    return gnuradio::make_block_sptr<null_source_impl>(sizeof_stream_item, num_outputs,
                                                       memset);
}

null_source_impl::null_source_impl(size_t sizeof_stream_item, size_t num_outputs,
                                   bool memset)
    : sync_block("null_source",
                 io_signature::make(0, 0, 0),
                 io_signature::make(num_outputs, num_outputs, sizeof_stream_item, cuda_buffer::type)),
      d_itemsize(sizeof_stream_item),
      d_num_outputs(num_outputs),
      d_memset(memset)
{
}

int null_source_impl::work(int noutput_items,
                          gr_vector_const_void_star& input_items,
                          gr_vector_void_star& output_items)
{
    // Ensure output buffer is safe to write (no downstream still reading).
    gr::cuda::wait_for_work(detail(), d_stream);

    // Fill GPU buffer with zeros (mimics GNU Radio's memset behavior).
    // Disabled (d_memset=false) in benchmarks to avoid saturating memory 
    // bandwidth, starving downstream compute kernels that share the same GPU, 
    // and skewing the results.
    if (d_memset) {
        auto out = static_cast<uint8_t*>(output_items[0]);
        check_cuda_errors(cudaMemsetAsync(out, 0, noutput_items * d_itemsize, d_stream),
                          "null_source: cudaMemsetAsync", d_logger);
    }

    // Mark outputs ready
    gr::cuda::mark_work_done(detail(), d_stream);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */


