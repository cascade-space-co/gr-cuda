/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_NOP_H
#define INCLUDED_CUDA_NOP_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief GPU pass-through that performs no computation or memory operations.
 * \ingroup cuda
 *
 * Data flows through cuda_buffer (triggering H2D/D2H transfers as
 * needed) but the block itself does nothing in work().  Useful for
 * benchmarking transfer overhead in isolation.
 */
class CUDA_API nop : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<nop> sptr;

    /*!
     * \brief Build a GPU nop block.
     *
     * \param sizeof_stream_item Size of a stream item in bytes.
     */
    static sptr make(size_t sizeof_stream_item);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_NOP_H */
