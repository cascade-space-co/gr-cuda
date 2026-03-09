/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_BUFFER_READER_H
#define INCLUDED_GR_CUDA_BUFFER_READER_H

#include <gnuradio/buffer_reader.h>
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
 * The consumer's CUDA stream is discovered lazily by attempting a
 * dynamic_cast<cuda_block*> on the consuming block (via link()).
 * If the cast succeeds, get_cuda_stream() provides the stream.
 * For non-GPU consumers (e.g. D2H edges to CPU blocks or Python
 * blocks), the cast returns nullptr and the sync is skipped.
 */
class cuda_buffer_reader : public buffer_reader
{
    friend class cuda_buffer;

public:
    void update_read_pointer(int nitems) override;

    cudaStream_t consumer_stream();

private:
    cuda_buffer_reader(buffer_sptr buf, unsigned int read_index, block_sptr link);

    cudaStream_t d_consumer_stream = nullptr;
    bool d_stream_resolved = false;
};

} /* namespace gr */

#endif /* INCLUDED_GR_CUDA_BUFFER_READER_H */
