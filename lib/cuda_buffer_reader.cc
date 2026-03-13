/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/block.h>
#include <gnuradio/block_detail.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_buffer_reader.h>
#include <gnuradio/cuda/cuda_error.h>

namespace gr {

cuda_buffer_reader::cuda_buffer_reader(buffer_sptr buf,
                                       unsigned int read_index,
                                       block_sptr link)
    : buffer_reader(buf, read_index, link)
{
}

void cuda_buffer_reader::set_consumer_stream(cudaStream_t s)
{
    d_consumer_stream = s;
    d_stream_resolved = true;
}

cudaStream_t cuda_buffer_reader::consumer_stream()
{
    if (!d_stream_resolved) {
        d_stream_resolved = true;
        auto* cb = dynamic_cast<cuda_block*>(link().get());
        if (cb)
            d_consumer_stream = cb->get_cuda_stream();
    }
    return d_consumer_stream;
}

void cuda_buffer_reader::update_read_pointer(int nitems)
{
    cudaStream_t cs = consumer_stream();
    if (cs) {
        auto cbuf = std::dynamic_pointer_cast<cuda_buffer>(d_buffer);
        if (cbuf)
            cbuf->mark_read_done(cs);
    }

    buffer_reader::update_read_pointer(nitems);
}

} /* namespace gr */
