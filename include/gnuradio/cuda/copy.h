/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_COPY_H
#define INCLUDED_CUDA_COPY_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace cuda {

/*!
 * \brief GPU memcpy between cuda_buffer endpoints.
 * \ingroup cuda
 */
class CUDA_API copy : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<copy> sptr;

    /*!
     * \brief Build a GPU copy block.
     *
     * \param itemsize Size of a stream item in bytes.
     */
    static sptr make(size_t itemsize);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_COPY_H */
