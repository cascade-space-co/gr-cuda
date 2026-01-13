/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_VECTOR_TO_STREAM_H
#define INCLUDED_CUDA_VECTOR_TO_STREAM_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_interpolator.h>

namespace gr {
namespace cuda {

/*!
 * \brief Convert a stream of blocks into a stream of items
 * \ingroup cuda
 */
class CUDA_API vector_to_stream : virtual public sync_interpolator
{
public:
    typedef std::shared_ptr<vector_to_stream> sptr;

    /*!
     * \brief Make a vector_to_stream block.
     * \param itemsize The number of bytes per item
     * \param vlen The number of items per input vector
     */
    static sptr make(size_t itemsize, size_t vlen);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_VECTOR_TO_STREAM_H */

