/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/block.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_buffer_reader.h>

namespace gr {

cuda_buffer_reader::cuda_buffer_reader(buffer_sptr buf,
                                       unsigned int read_index,
                                       block_sptr link)
    : buffer_reader(buf, read_index, link)
{
    // Resolve the consumer's CUDA stream once, here during single-threaded
    // flowgraph setup, so that consumer_stream() is a lock-free, race-free
    // read at runtime.  It is otherwise called from two different threads:
    // the producer thread (via cuda_buffer::post_work) and the consumer
    // thread (via update_read_pointer).
    //
    // Python GPU blocks are not cuda_block, so they resolve to nullptr here
    // and instead register their stream via set_consumer_stream() in start(),
    // which also runs before the scheduler launches any block threads.
    if (auto* cb = dynamic_cast<cuda_block*>(link.get()))
        d_consumer_stream = cb->get_cuda_stream();

    configure_default_loggers(d_logger, d_debug_logger, "cuda_buffer_reader");
}

void cuda_buffer_reader::set_consumer_stream(cudaStream_t s) { d_consumer_stream = s; }

cudaStream_t cuda_buffer_reader::consumer_stream() const { return d_consumer_stream; }

void cuda_buffer_reader::update_read_pointer(int nitems)
{
    // Auto-sync: mark_read_done; see autosync table in cuda_buffer.h
    cudaStream_t cs = consumer_stream();
    if (cs) {
        auto cbuf = std::dynamic_pointer_cast<cuda_buffer>(d_buffer);
        if (cbuf) {
            cbuf->mark_read_done(cs);
        } else if (!d_sync_error_logged) {
            // A consumer stream resolved (this is a GPU consumer), yet the
            // backing buffer is not a cuda_buffer.  This should be impossible:
            // cuda_buffer_reader is only created by cuda_buffer::create_reader.
            // If it happens, read-done sync is silently skipped and the
            // producer may overwrite device data still being read on the GPU.
            d_sync_error_logged = true;
            d_logger->error("update_read_pointer: consumer stream resolved but "
                            "backing buffer is not a cuda_buffer; read-done "
                            "synchronization skipped (possible data corruption)");
        }
    }

    buffer_reader::update_read_pointer(nitems);
}

} /* namespace gr */
