/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_SKIPHEAD_H
#define INCLUDED_CUDA_SKIPHEAD_H

#include <gnuradio/block.h>
#include <gnuradio/cuda/api.h>

namespace gr {
namespace cuda {

/*!
 * \brief Drop the first N items of a GPU stream, then pass the rest.
 * \ingroup cuda
 *
 * The GPU counterpart of blocks.skiphead.  Skipped items are never read, and
 * the rest are copied on the GPU, so no device-to-host copy is made.  Tags on
 * skipped items are dropped; later tags are shifted back by N.
 */
class CUDA_API skiphead : virtual public gr::block
{
public:
    typedef std::shared_ptr<skiphead> sptr;

    /*!
     * \brief Build a GPU skiphead block.
     *
     * \param itemsize Size of a stream item in bytes.
     * \param nitems_to_skip Number of items to drop from the start.
     */
    static sptr make(size_t itemsize, uint64_t nitems_to_skip);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_SKIPHEAD_H */
