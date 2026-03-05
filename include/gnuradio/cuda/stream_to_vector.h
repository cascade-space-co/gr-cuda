/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_STREAM_TO_VECTOR_H
#define INCLUDED_CUDA_STREAM_TO_VECTOR_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_decimator.h>

namespace gr {
namespace cuda {

/*!
 * \brief Convert a stream of items into a stream of vectors of length vlen
 * \ingroup cuda
 */
class CUDA_API stream_to_vector : virtual public sync_decimator
{
public:
    typedef std::shared_ptr<stream_to_vector> sptr;

    /*!
     * \brief Make a stream_to_vector block.
     * \param itemsize The number of bytes per item
     * \param vlen The number of items per output vector
     */
    static sptr make(size_t itemsize, size_t vlen);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_STREAM_TO_VECTOR_H */
