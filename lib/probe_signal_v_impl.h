/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_PROBE_SIGNAL_V_IMPL_H
#define INCLUDED_CUDA_PROBE_SIGNAL_V_IMPL_H

#include "detail/last_item_probe.h"
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/probe_signal_v.h>

namespace gr {
namespace cuda {

template <class T>
class probe_signal_v_impl : public probe_signal_v<T>, public cuda_block
{
private:
    const size_t d_size;
    detail::last_item_probe d_probe;

public:
    probe_signal_v_impl(size_t size);
    ~probe_signal_v_impl() override = default;

    std::vector<T> level() const override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_PROBE_SIGNAL_V_IMPL_H */
