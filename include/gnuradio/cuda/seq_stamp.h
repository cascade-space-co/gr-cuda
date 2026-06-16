/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_SEQ_STAMP_H
#define INCLUDED_CUDA_SEQ_STAMP_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Prepend an incrementing uint64 sequence number to each item.
 * \ingroup cuda
 *
 * A pass-through filter operating on cuda_buffer (GPU) data.  Each input
 * item is copied to the output with an 8-byte little-endian sequence
 * counter prepended, so output items are (payload_size + 8) bytes:
 *
 *     [ uint64 seq ][ original payload_size-byte payload ]
 *
 * The counter increments by one per item across the whole stream.  Pair
 * with seq_strip on the receive side to recover the sequence number.
 */
class CUDA_API seq_stamp : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<seq_stamp> sptr;

    /*!
     * \param payload_size  Input payload size in bytes (must be >= 1).  Output
     *                      items are 8 bytes larger (sequence header prepended).
     */
    static sptr make(int payload_size);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SEQ_STAMP_H */
