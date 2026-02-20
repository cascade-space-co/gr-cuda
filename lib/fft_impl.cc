/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/cuda/cuda_error.h>
#include <sstream>
#include <stdexcept>

// Kernel wrappers (implemented in fft.cu)
void exec_kernel_window(const cufftComplex* in,
                        cufftComplex* out,
                        const float* window,
                        size_t total_items,
                        size_t fft_size,
                        cudaStream_t stream);
void exec_kernel_real_window(const float* in,
                             cufftComplex* out,
                             const float* window,
                             size_t total_items,
                             size_t fft_size,
                             cudaStream_t stream);
void exec_kernel_real_to_complex(const float* in,
                                 cufftComplex* out,
                                 size_t total_items,
                                 cudaStream_t stream);
void exec_kernel_ifftshift(const cufftComplex* in,
                           cufftComplex* out,
                           size_t total_items,
                           size_t fft_size,
                           cudaStream_t stream);
void exec_kernel_real_ifftshift(const float* in,
                                cufftComplex* out,
                                size_t total_items,
                                size_t fft_size,
                                cudaStream_t stream);
void exec_kernel_window_ifftshift(const cufftComplex* in,
                                  cufftComplex* out,
                                  const float* window,
                                  size_t total_items,
                                  size_t fft_size,
                                  cudaStream_t stream);
void exec_kernel_real_window_ifftshift(const float* in,
                                       cufftComplex* out,
                                       const float* window,
                                       size_t total_items,
                                       size_t fft_size,
                                       cudaStream_t stream);
void exec_kernel_fftshift(const cufftComplex* in,
                          cufftComplex* out,
                          size_t total_items,
                          size_t fft_size,
                          cudaStream_t stream);

namespace {
inline void check_cufft(cufftResult rc, const char* where)
{
    if (rc != CUFFT_SUCCESS) {
        std::ostringstream msg;
        msg << "cuFFT error in " << where << ": code " << static_cast<int>(rc);
        throw std::runtime_error(msg.str());
    }
}
} // namespace

namespace gr {
namespace cuda {

fft::sptr fft::make(size_t fft_size,
                    bool forward,
                    const std::vector<float>& window,
                    bool shift,
                    bool real_input)
{
    return gnuradio::make_block_sptr<fft_impl>(fft_size, forward, window, shift, real_input);
}

fft_impl::fft_impl(size_t fft_size,
                   bool forward,
                   const std::vector<float>& window,
                   bool shift,
                   bool real_input)
    : gr::sync_block("fft",
                     io_signature::make(1,
                                        1,
                                        (real_input ? sizeof(float) : sizeof(gr_complex)) *
                                            fft_size,
                                        cuda_buffer::type),
                     io_signature::make(1, 1, sizeof(gr_complex) * fft_size, cuda_buffer::type)),
      d_fft_size(fft_size),
      d_forward(forward),
      d_shift(shift),
      d_has_window(!window.empty()),
      d_real_input(real_input),
      d_window_dev(nullptr),
      d_window_size(0),
      d_work_dev(nullptr),
      d_work_items(0)
{
    if (d_fft_size == 0) {
        throw std::invalid_argument("fft_size must be > 0");
    }

    if (d_has_window) {
        if (window.size() != d_fft_size) {
            throw std::invalid_argument("window length must match fft_size");
        }
        d_window_size = window.size();
        check_cuda_errors(cudaMalloc((void**)&d_window_dev, d_window_size * sizeof(float)),
                          "fft: cudaMalloc window", d_logger);
        check_cuda_errors(cudaMemcpyAsync(d_window_dev,
                                          window.data(),
                                          d_window_size * sizeof(float),
                                          cudaMemcpyHostToDevice,
                                          d_stream),
                          "fft: cudaMemcpyAsync H2D window", d_logger);
    }
}

fft_impl::~fft_impl()
{
    for (auto& kv : d_plan_cache) {
        cufftDestroy(kv.second);
    }
    if (d_window_dev) {
        cudaFree(d_window_dev);
    }
    if (d_work_dev) {
        cudaFree(d_work_dev);
    }
}

void fft_impl::ensure_work_buffers(size_t total_items)
{
    if (total_items <= d_work_items) {
        return;
    }
    if (d_work_dev) {
        cudaFree(d_work_dev);
        d_work_dev = nullptr;
        d_work_items = 0;
    }
    check_cuda_errors(cudaMalloc((void**)&d_work_dev, total_items * sizeof(cufftComplex)),
                      "fft: cudaMalloc work buffer", d_logger);
    d_work_items = total_items;
}

cufftHandle fft_impl::get_plan(int batch)
{
    std::lock_guard<std::mutex> lock(d_plan_mutex);
    auto it = d_plan_cache.find(batch);
    // Reuse a cached cuFFT plan for this batch size to avoid costly plan creation.
    if (it != d_plan_cache.end()) {
        return it->second;
    }

    // Cache miss: create a new plan sized for the current batch.
    cufftHandle plan;
    int n[1] = { static_cast<int>(d_fft_size) };
    int inembed[1] = { static_cast<int>(d_fft_size) };
    int onembed[1] = { static_cast<int>(d_fft_size) };
    int istride = 1;
    int ostride = 1;
    int idist = static_cast<int>(d_fft_size);
    int odist = static_cast<int>(d_fft_size);

    check_cufft(cufftPlanMany(&plan,
                              1,
                              n,
                              inembed,
                              istride,
                              idist,
                              onembed,
                              ostride,
                              odist,
                              CUFFT_C2C,
                              batch),
                "cufftPlanMany");

    d_plan_cache.emplace(batch, plan);
    return plan;
}

int fft_impl::work(int noutput_items,
                   gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items)
{
    if (noutput_items <= 0) {
        return 0;
    }

    // Ensure upstream GPU work is complete before reading inputs.
    gr::cuda::wait_for_inputs(detail(), d_stream);

    auto in = reinterpret_cast<const cufftComplex*>(input_items[0]);
    auto out = reinterpret_cast<cufftComplex*>(output_items[0]);

    const size_t total_items = static_cast<size_t>(noutput_items) * d_fft_size;

    // Use a staging buffer when we need to pre/post-process data.
    if (d_real_input || d_has_window || d_shift) {
        ensure_work_buffers(total_items);
    }

    cufftComplex* fft_in = const_cast<cufftComplex*>(in);
    cufftComplex* fft_out = out;

    // Pre-FFT path: optional ifftshift/window/real->complex conversions.
    if (d_shift && !d_forward) {
        if (d_real_input) {
            const auto real_in = reinterpret_cast<const float*>(input_items[0]);
            if (d_has_window) {
                exec_kernel_real_window_ifftshift(real_in,
                                                  d_work_dev,
                                                  d_window_dev,
                                                  total_items,
                                                  d_fft_size,
                                                  d_stream);
            } else {
                exec_kernel_real_ifftshift(
                    real_in, d_work_dev, total_items, d_fft_size, d_stream);
            }
        } else {
            if (d_has_window) {
                exec_kernel_window_ifftshift(
                    in, d_work_dev, d_window_dev, total_items, d_fft_size, d_stream);
            } else {
                exec_kernel_ifftshift(in, d_work_dev, total_items, d_fft_size, d_stream);
            }
        }
        fft_in = d_work_dev;
        fft_out = out;
    } else {
        if (d_real_input) {
            const auto real_in = reinterpret_cast<const float*>(input_items[0]);
            if (d_has_window) {
                exec_kernel_real_window(
                    real_in, d_work_dev, d_window_dev, total_items, d_fft_size, d_stream);
            } else {
                exec_kernel_real_to_complex(real_in, d_work_dev, total_items, d_stream);
            }
            fft_in = d_work_dev;
        } else if (d_has_window) {
            exec_kernel_window(in, d_work_dev, d_window_dev, total_items, d_fft_size, d_stream);
            fft_in = d_work_dev;
        }

        if (d_shift) {
            fft_out = d_work_dev;
        }
    }

    // Execute the cuFFT plan on this block's stream.
    cufftHandle plan = get_plan(noutput_items);
    check_cufft(cufftSetStream(plan, d_stream), "cufftSetStream");
    check_cufft(cufftExecC2C(plan,
                              fft_in,
                              fft_out,
                              d_forward ? CUFFT_FORWARD : CUFFT_INVERSE),
                "cufftExecC2C");

    // Post-FFT path: optional fftshift.
    if (d_shift && d_forward) {
        exec_kernel_fftshift(fft_out, out, total_items, d_fft_size, d_stream);
    }

    // Notify downstream CUDA buffers that output is ready.
    gr::cuda::mark_outputs_ready(detail(), d_stream);
    return noutput_items;
}

} // namespace cuda
} // namespace gr
