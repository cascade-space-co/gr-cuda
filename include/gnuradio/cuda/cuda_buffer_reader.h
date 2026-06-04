/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_BUFFER_READER_H
#define INCLUDED_GR_CUDA_BUFFER_READER_H

#include <gnuradio/buffer_reader.h>
#include <gnuradio/cuda/api.h>
#include <cuda_runtime_api.h>

namespace gr {

class cuda_buffer;

/*!
 * \brief Custom reader for cuda_buffer that signals read-done automatically.
 *
 * When the scheduler calls update_read_pointer() after a consumer's
 * general_work(), this reader records a CUDA event on the consumer's
 * stream so the upstream producer knows it is safe to overwrite
 * (see autosync table in cuda_buffer.h).
 *
 * The consumer's CUDA stream is resolved once at construction (during
 * single-threaded flowgraph setup) by attempting a dynamic_cast<cuda_block*>
 * on the consuming block.  If the cast succeeds, get_cuda_stream() provides
 * the stream.  Resolving eagerly keeps consumer_stream() a lock-free read at
 * runtime, where it is called concurrently from the producer thread (via
 * cuda_buffer::post_work) and the consumer thread (via update_read_pointer).
 *
 * For non-GPU consumers (e.g. D2H edges to CPU blocks or Python blocks), the
 * cast returns nullptr and the sync is skipped; Python GPU blocks instead
 * register their stream via set_consumer_stream() in start().
 */
class CUDA_API cuda_buffer_reader : public buffer_reader
{
    friend class cuda_buffer;

public:
    void update_read_pointer(int nitems) override;

    cudaStream_t consumer_stream() const;

    /*!
     * \brief Register the consumer's CUDA stream explicitly.
     *
     * Used by Python GPU blocks that cannot be discovered via
     * dynamic_cast<cuda_block*> at construction.  Called from start()
     * (single-threaded, before the scheduler runs) to overwrite the
     * stream that consumer_stream() returns.
     */
    void set_consumer_stream(cudaStream_t s);

private:
    cuda_buffer_reader(buffer_sptr buf, unsigned int read_index, block_sptr link);

    cudaStream_t d_consumer_stream = nullptr;

    logger_ptr d_logger;
    logger_ptr d_debug_logger;
    bool d_sync_error_logged = false;
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BUFFER_READER_H */
