/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_HEAD_H
#define INCLUDED_CUDA_HEAD_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Pass the first N items of a GPU stream, then stop.
 * \ingroup cuda
 *
 * The GPU counterpart of blocks.head.  Items are copied on the GPU, so
 * bounding a GPU chain costs no device-to-host copy.  The output is optional:
 * left unconnected, the block consumes N items without reading them and then
 * reports done.
 */
class CUDA_API head : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<head> sptr;

    /*!
     * \brief Build a GPU head block.
     *
     * \param sizeof_stream_item Size of a stream item in bytes.
     * \param nitems Number of items to pass before stopping.
     */
    static sptr make(size_t sizeof_stream_item, uint64_t nitems);

    /*!
     * \brief Reset the count of items passed so far.
     */
    virtual void reset() = 0;

    /*!
     * \brief Set the number of items to pass before stopping.
     */
    virtual void set_length(uint64_t nitems) = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_HEAD_H */
