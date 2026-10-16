/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gather.cuh"
#include <gnuradio/cuda/cuda_error.h>
#include <algorithm>
#include <cstdint>
#include <utility>

namespace {

constexpr int block_size = 256;
constexpr size_t max_grid_size = 65535;

// Each item is copied as `words` machine words of type W, so one kernel
// handles any item size. One thread copies one word of the output.
template <typename W>
__global__ void kernel_gather(const W* in,
                              W* out,
                              const int* cols,
                              size_t total,
                              size_t ncols,
                              size_t in_stride,
                              size_t words)
{
    // Grid-stride loop: the grid is capped at max_grid_size blocks, so each
    // thread may handle several output words.
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total;
         i += step) {
        // The output is dense, `rows x ncols` items of `words` words each.
        // Split the output word index i into (row, j, word).
        const size_t word = i % words;
        const size_t item = i / words;
        const size_t row = item / ncols;
        const size_t j = item % ncols;
        // Output column j comes from input column cols[j] of the same row.
        // Input rows are in_stride items apart.
        const size_t col = cols ? cols[j] : j;
        out[i] = in[(row * in_stride + col) * words + word];
    }
}

template <typename W>
void launch(const void* in,
            void* out,
            const int* cols,
            size_t rows,
            size_t ncols,
            size_t in_stride,
            size_t itemsize,
            cudaStream_t stream)
{
    const size_t words = itemsize / sizeof(W);
    const size_t total = rows * ncols * words;
    const size_t grid = std::min((total + block_size - 1) / block_size, max_grid_size);
    kernel_gather<W><<<grid, block_size, 0, stream>>>(static_cast<const W*>(in),
                                                      static_cast<W*>(out),
                                                      cols,
                                                      total,
                                                      ncols,
                                                      in_stride,
                                                      words);
    check_cuda_errors(cudaGetLastError(), "gather: kernel launch");
}

} // namespace

void exec_gather(const void* in,
                 void* out,
                 const int* cols,
                 size_t rows,
                 size_t ncols,
                 size_t in_stride,
                 size_t itemsize,
                 cudaStream_t stream)
{
    if (rows == 0 || ncols == 0) {
        return;
    }
    if (itemsize % 8 == 0) {
        launch<uint64_t>(in, out, cols, rows, ncols, in_stride, itemsize, stream);
    } else if (itemsize % 4 == 0) {
        launch<uint32_t>(in, out, cols, rows, ncols, in_stride, itemsize, stream);
    } else if (itemsize % 2 == 0) {
        launch<uint16_t>(in, out, cols, rows, ncols, in_stride, itemsize, stream);
    } else {
        launch<uint8_t>(in, out, cols, rows, ncols, in_stride, itemsize, stream);
    }
}

gather_columns::~gather_columns()
{
    if (d_device) {
        cudaFree(d_device);
    }
}

void gather_columns::set(std::vector<int> cols)
{
    d_host = std::move(cols);
    d_dirty = true;
}

const int* gather_columns::device(cudaStream_t stream)
{
    // The common case: nothing changed since the last upload.
    if (!d_dirty) {
        return d_device;
    }
    // Reallocate only when the table grows; same-size or smaller updates
    // reuse the existing buffer.
    if (d_host.size() > d_capacity) {
        // Earlier kernels on this stream may still be reading the old table.
        check_cuda_errors(cudaStreamSynchronize(stream),
                          "gather_columns: cudaStreamSynchronize");
        check_cuda_errors(cudaFree(d_device), "gather_columns: cudaFree");
        d_device = nullptr;
        d_capacity = 0;
        check_cuda_errors(cudaMalloc(&d_device, d_host.size() * sizeof(int)),
                          "gather_columns: cudaMalloc");
        d_capacity = d_host.size();
    }
    // Copies from pageable memory return once d_host has been staged, so it
    // may be changed again right after this call.
    check_cuda_errors(cudaMemcpyAsync(d_device,
                                      d_host.data(),
                                      d_host.size() * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      stream),
                      "gather_columns: cudaMemcpyAsync H2D");
    d_dirty = false;
    return d_device;
}
