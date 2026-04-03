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
 * \brief Extract the first 8 bytes (uint64 counter) from each
 *        payload-sized GPU item and output it as a host-side uint64.
 * \ingroup cuda
 *
 * Input:  cuda_buffer items of payload_size bytes (GPU).
 * Output: host-buffer uint64_t items (CPU).
 *
 * Pair with seq_stamp on the transmit side.  The output stream is
 * plain host memory, so downstream blocks can be ordinary Python or
 * C++ blocks that operate on uint64_t values.
 */
class CUDA_API seq_strip : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<seq_strip> sptr;

    /*!
     * \param payload_size  Input item size in bytes (must be >= 8 and a multiple of 8)
     */
    static sptr make(int payload_size);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SEQ_STRIP_H */
