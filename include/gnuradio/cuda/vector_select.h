/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_VECTOR_SELECT_H
#define INCLUDED_CUDA_VECTOR_SELECT_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <vector>

namespace gr {
namespace cuda {

/*!
 * \brief Select elements of each input vector, on CUDA buffers
 * \ingroup cuda
 *
 * Each input item is a vector of \p vlen elements. Each output item holds
 * input[indices[0]], input[indices[1]], ... in the order given, so one block
 * can pick any channels of a multi-channel stream, reorder them, or repeat
 * them. With a single index the output is a plain stream of elements.
 */
class CUDA_API vector_select : virtual public sync_block
{
public:
    typedef std::shared_ptr<vector_select> sptr;

    /*!
     * \brief Make a vector_select block.
     * \param itemsize The number of bytes per vector element
     * \param vlen The number of elements per input vector
     * \param indices Elements to keep, each in [0, vlen). The output vector
     *        length is indices.size().
     * \param output_multiple Output batch size requested from the scheduler
     */
    static sptr make(size_t itemsize,
                     size_t vlen,
                     const std::vector<int>& indices,
                     int output_multiple = 1);

    /*!
     * \brief Select different elements. The number of indices cannot change.
     */
    virtual void set_indices(const std::vector<int>& indices) = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_VECTOR_SELECT_H */
