/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "multiply_const_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>

template <typename T>
void exec_kernel_multiply_const(const T* in,
                                T* out,
                                T k,
                                int grid_size,
                                int block_size,
                                size_t n,
                                cudaStream_t stream);

template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock);


namespace gr {
namespace cuda {

template <class T>
typename multiply_const<T>::sptr multiply_const<T>::make(T k, size_t vlen)
{
    return gnuradio::make_block_sptr<multiply_const_impl<T>>(k, vlen);
}

template <class T>
multiply_const_impl<T>::multiply_const_impl(T k, size_t vlen)
    : gr::sync_block("multiply_const",
                     io_signature::make(1, 1, sizeof(T) * vlen, cuda_buffer::type),
                     io_signature::make(1, 1, sizeof(T) * vlen, cuda_buffer::type)),
      d_k(k),
      d_vlen(vlen)
{
    get_block_and_grid<T>(&d_min_grid_size, &d_block_size);
}

template <class T>
int multiply_const_impl<T>::work(int noutput_items,
                                 gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    // Ensure upstream GPU work is complete before reading inputs.
    gr::cuda::wait_for_inputs(this->detail(), d_stream);
    auto in = static_cast<const T*>(input_items[0]);
    auto out = static_cast<T*>(output_items[0]);
    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    exec_kernel_multiply_const<T>(in,
                                  out,
                                  d_k,
                                  gridSize,
                                  d_block_size,
                                  noutput_items,
                                  d_stream);


    // Notify downstream CUDA buffers that output is ready.
    gr::cuda::mark_outputs_ready(this->detail(), d_stream);
    // Tell runtime system how many output items we produced.
    return noutput_items;
}

template class multiply_const<std::int16_t>;
template class multiply_const<std::int32_t>;
template class multiply_const<float>;
template class multiply_const<gr_complex>;

} /* namespace cuda */
} /* namespace gr */
