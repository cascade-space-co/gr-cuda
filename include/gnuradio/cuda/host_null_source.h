/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_HOST_NULL_SOURCE_H
#define INCLUDED_CUDA_HOST_NULL_SOURCE_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief CPU null source whose zero-fill can be disabled
 * \ingroup cuda
 *
 * Like blocks.null_source, but the memset is optional and its byte count
 * doesn't overflow for work() calls of 2 GiB or more.  Used by the transfer
 * benchmark; there is no GRC block.
 */
class CUDA_API host_null_source : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<host_null_source> sptr;

    /*!
     * \param sizeof_stream_item Size of a stream item in bytes
     * \param memset If true (default), zero-fill the output buffer on the CPU
     *               each work() call.  Set to false to only produce items.
     */
    static sptr make(size_t sizeof_stream_item, bool memset = true);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_HOST_NULL_SOURCE_H */
