/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_VECTOR_TO_STREAMS_H
#define INCLUDED_CUDA_VECTOR_TO_STREAMS_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief Convert 1 stream of vectors into N streams of items (Deinterleave)
 * \ingroup cuda
 */
class CUDA_API vector_to_streams : virtual public sync_block
{
public:
    typedef std::shared_ptr<vector_to_streams> sptr;

    /*!
     * \brief Make a vector_to_streams block.
     * \param itemsize The number of bytes per item
     * \param num_streams The number of output streams (and vlen of input)
     */
    static sptr make(size_t itemsize, size_t num_streams);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_VECTOR_TO_STREAMS_H */

