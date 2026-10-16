/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_KEEP_M_IN_N_H
#define INCLUDED_CUDA_KEEP_M_IN_N_H

#include <gnuradio/block.h>
#include <gnuradio/cuda/api.h>

namespace gr {
namespace cuda {

/*!
 * \brief Keep \p m items out of every \p n, on CUDA buffers
 * \ingroup cuda
 *
 * GPU equivalent of gr::blocks::keep_m_in_n. From each block of \p n input
 * items it outputs \p m items. If offset + m <= n, these are
 * block[offset:offset+m]. Otherwise the selection wraps, and the output is
 * block[0:offset+m-n] followed by block[offset:n], as in GNU Radio; kept
 * items always stay in their original order. Tags are propagated by the
 * scheduler at the relative rate m / n.
 */
class CUDA_API keep_m_in_n : virtual public gr::block
{
public:
    typedef std::shared_ptr<keep_m_in_n> sptr;

    /*!
     * \brief Make a keep_m_in_n block.
     * \param itemsize The number of bytes per item
     * \param m Number of items kept per block, 1 <= m <= n
     * \param n Block size in items
     * \param offset Index of the first kept item in each block, 0 <= offset < n
     * \param output_multiple Output batch size requested from the scheduler.
     *        The block always produces whole groups of m, so the scheduler is
     *        given the least common multiple of m and this value.
     */
    static sptr
    make(size_t itemsize, int m, int n, int offset = 0, int output_multiple = 1);

    virtual void set_m(int m) = 0;
    virtual void set_n(int n) = 0;
    virtual void set_offset(int offset) = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_KEEP_M_IN_N_H */
