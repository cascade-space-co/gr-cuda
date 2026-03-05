/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_STREAMS_TO_VECTOR_H
#define INCLUDED_CUDA_STREAMS_TO_VECTOR_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Convert N streams of items into 1 stream of vectors (Interleave)
 * \ingroup cuda
 */
class CUDA_API streams_to_vector : virtual public sync_block
{
public:
    typedef std::shared_ptr<streams_to_vector> sptr;

    /*!
     * \brief Make a streams_to_vector block.
     * \param itemsize The number of bytes per item
     * \param num_streams The number of input streams (and vlen of output)
     */
    static sptr make(size_t itemsize, size_t num_streams);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_STREAMS_TO_VECTOR_H */
