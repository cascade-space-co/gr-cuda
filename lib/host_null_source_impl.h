/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_HOST_NULL_SOURCE_IMPL_H
#define INCLUDED_CUDA_HOST_NULL_SOURCE_IMPL_H

#include <gnuradio/cuda/host_null_source.h>

namespace gr {
namespace cuda {

class host_null_source_impl : public host_null_source
{
private:
    size_t d_itemsize;
    bool d_memset;

public:
    host_null_source_impl(size_t sizeof_stream_item, bool memset);
    ~host_null_source_impl() override = default;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_HOST_NULL_SOURCE_IMPL_H */
