/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_KEEP_ONE_IN_N_H
#define INCLUDED_CUDA_KEEP_ONE_IN_N_H

#include <gnuradio/block.h>
#include <gnuradio/cuda/api.h>

namespace gr {
namespace cuda {

/*!
 * \brief Keep one item out of every \p n, on CUDA buffers
 * \ingroup cuda
 *
 * GPU equivalent of gr::blocks::keep_one_in_n. Outputs input items n - 1,
 * 2n - 1, ... and keeps its position in the count across work calls, so any
 * \p n works regardless of buffer size. set_n() restarts the count. Each tag
 * moves to the first kept item at or after it.
 */
class CUDA_API keep_one_in_n : virtual public gr::block
{
public:
    typedef std::shared_ptr<keep_one_in_n> sptr;

    /*!
     * \brief Make a keep_one_in_n block.
     * \param itemsize The number of bytes per item
     * \param n Keep one item out of every n
     * \param output_multiple Output batch size requested from the scheduler
     */
    static sptr make(size_t itemsize, int n, int output_multiple = 1);

    /*!
     * \brief Set n and restart the count.
     */
    virtual void set_n(int n) = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_KEEP_ONE_IN_N_H */
