/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2013 Free Software Foundation, Inc.
 * Copyright 2021 Josh Morman
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_COPY_IMPL_H
#define INCLUDED_CUDA_COPY_IMPL_H

#include <gnuradio/cuda/copy.h>
#include <gnuradio/cuda/cuda_block.h>

namespace gr {
namespace cuda {

class copy_impl : public copy, public cuda_block {
private:
  size_t d_itemsize;
  bool d_passthrough;

public:
  copy_impl(size_t itemsize, bool passthrough);
  ~copy_impl() override = default;

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_COPY_IMPL_H */
