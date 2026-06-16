/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_SEQ_STRIP_H
#define INCLUDED_CUDA_SEQ_STRIP_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Recover the uint64 sequence number prepended by seq_stamp.
 * \ingroup cuda
 *
 * Splits each (payload_size + 8)-byte GPU input item produced by seq_stamp
 * back into its two parts:
 *
 *   - Output 0 (required): the 8-byte sequence header as a packed uint64.
 *   - Output 1 (optional): the original payload_size-byte payload, with the
 *     sequence header removed.
 *
 * Input:    cuda_buffer items of (payload_size + 8) bytes (GPU).
 * Output 0: cuda_buffer uint64_t items (GPU).
 * Output 1: cuda_buffer payload_size-byte items (GPU); only produced when
 *           the port is connected.
 *
 * Pair with seq_stamp on the transmit side.  Both outputs are GPU
 * cuda_buffers: connect them to CUDA (CuPy) blocks to keep the data on the
 * device, or to ordinary host blocks, in which case gr-cuda copies that
 * edge device->host automatically.  Leaving output 0 connected and the
 * payload unconnected gives the original sequence-number-only behavior.
 */
class CUDA_API seq_strip : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<seq_strip> sptr;

    /*!
     * \param payload_size  Payload size in bytes of the seq_stamp upstream
     *                      (must be >= 1); input items are payload_size + 8 bytes.
     */
    static sptr make(int payload_size);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SEQ_STRIP_H */
