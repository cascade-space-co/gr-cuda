/* -*- c++ -*- */
/*
 * Copyright 2025
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_NULL_SOURCE_IMPL_H
#define INCLUDED_CUDA_NULL_SOURCE_IMPL_H

#include <gnuradio/cuda/null_source.h>
#include <cuda_runtime.h>

namespace gr {
namespace cuda {

class null_source_impl : public null_source {
private:
  size_t d_itemsize;
  size_t d_num_outputs;
  cudaStream_t d_stream;

public:
  null_source_impl(size_t sizeof_stream_item, size_t num_outputs);
  ~null_source_impl() override;

  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_NULL_SOURCE_IMPL_H */


