/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "add_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>

// Forward declaration of kernel launcher
template <typename T>
void exec_kernel_add(T** inputs,
                     T* out,
                     int num_inputs,
                     int grid_size,
                     int block_size,
                     size_t n,
                     cudaStream_t stream);

template <typename T>
void get_add_block_and_grid(int* minGrid, int* minBlock);


namespace gr {
namespace cuda {

template <class T>
typename add<T>::sptr add<T>::make(size_t num_inputs, size_t vlen)
{
    return gnuradio::make_block_sptr<add_impl<T>>(num_inputs, vlen);
}

template <class T>
add_impl<T>::add_impl(size_t num_inputs, size_t vlen)
    : gr::sync_block("add",
                     io_signature::make(num_inputs, num_inputs, sizeof(T) * vlen, cuda_buffer::type),
                     io_signature::make(1, 1, sizeof(T) * vlen, cuda_buffer::type)),
      d_num_inputs(num_inputs),
      d_vlen(vlen)
{
    get_add_block_and_grid<T>(&d_min_grid_size, &d_block_size);
    check_cuda_errors(cudaStreamCreateWithFlags(&d_stream, cudaStreamNonBlocking));
    
    // Allocate memory for input pointers on device
    check_cuda_errors(cudaMalloc((void**)&d_input_ptrs_dev, sizeof(T*) * d_num_inputs));
}

template <class T>
add_impl<T>::~add_impl()
{
    if (d_input_ptrs_dev) {
        cudaFree(d_input_ptrs_dev);
    }
    check_cuda_errors(cudaStreamDestroy(d_stream));
}

template <class T>
int add_impl<T>::work(int noutput_items,
                      gr_vector_const_void_star& input_items,
                      gr_vector_void_star& output_items)
{
    // Wait for inputs
    gr::cuda::wait_for_inputs(this->detail(), d_stream);

    auto out = static_cast<T*>(output_items[0]);
    
    // Prepare input pointers
    // NOTE: input_items is a std::vector on the CPU holding pointers to GPU buffers.
    // The GPU kernel cannot access this CPU vector directly to iterate over inputs.
    // We must copy these GPU pointers into a device-accessible array (d_input_ptrs_dev)
    // so the kernel can access inputs[i].
    std::vector<T*> host_input_ptrs(d_num_inputs);
    for (size_t i = 0; i < d_num_inputs; i++) {
        host_input_ptrs[i] = (T*)input_items[i]; // const cast
    }
    
    // Copy input pointers to device
    check_cuda_errors(cudaMemcpyAsync(d_input_ptrs_dev, 
                                      host_input_ptrs.data(), 
                                      sizeof(T*) * d_num_inputs, 
                                      cudaMemcpyHostToDevice, 
                                      d_stream));

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    
    exec_kernel_add<T>(d_input_ptrs_dev,
                       out,
                       d_num_inputs,
                       gridSize,
                       d_block_size,
                       noutput_items,
                       d_stream);
    
    // Mark outputs ready
    gr::cuda::mark_outputs_ready(this->detail(), d_stream);

    return noutput_items;
}

// Instantiate templates
template class add<std::int16_t>;
template class add<std::int32_t>;
template class add<float>;
template class add<gr_complex>;

} /* namespace cuda */
} /* namespace gr */

