/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_ADD_H
#define INCLUDED_CUDA_ADD_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief output = input[0] + input[1] + ... + input[N-1]
 * \ingroup math_operators_blk
 */
template <class T>
class CUDA_API add : virtual public sync_block
{

public:
    // gr::blocks::add::sptr
    typedef std::shared_ptr<add<T>> sptr;

    /*!
     * \brief Create an instance of add
     * \param num_inputs number of input streams
     * \param vlen number of items in vector
     */
    static sptr make(size_t num_inputs, size_t vlen = 1);
};

typedef add<std::int16_t> add_ss;
typedef add<std::int32_t> add_ii;
typedef add<float> add_ff;
typedef add<gr_complex> add_cc;

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_ADD_H */



