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
 * \brief Stamp an incrementing uint64 counter into the first 8 bytes of
 *        each payload-sized item passing through.
 * \ingroup cuda
 *
 * Operates entirely on cuda_buffer (GPU) data.  Only the first 8 bytes
 * of each output item are written; the remaining bytes are left as-is
 * (undefined).  Pair with seq_strip on the receive side.
 */
class CUDA_API seq_stamp : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<seq_stamp> sptr;

    /*!
     * \param payload_size  Item size in bytes (must be >= 8 and a multiple of 8)
     */
    static sptr make(int payload_size);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SEQ_STAMP_H */
