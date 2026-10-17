/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_GR_CUDA_GATHER_CUH
#define INCLUDED_GR_CUDA_GATHER_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

/*! Copy selected items out of each row of a strided item array:
 *  out[r * ncols + j] = in[r * in_stride + cols[j]] for r < rows, j < ncols.
 *  If cols is null, cols[j] = j. Indices and strides count items of itemsize
 *  bytes; in and out must be aligned to itemsize's largest power-of-two
 *  factor (up to 8). */
void exec_gather(const void* in,
                 void* out,
                 const int* cols,
                 size_t rows,
                 size_t ncols,
                 size_t in_stride,
                 size_t itemsize,
                 cudaStream_t stream);

/*! Column table for exec_gather, kept on the host and mirrored on the device.
 *
 *  set() only stores the new table on the host; device() uploads it on the
 *  block's stream the next time the block runs. The upload is then ordered
 *  after kernels already launched with the old table and before the next
 *  one, so a kernel never sees a partly written table. Setters call set() and
 *  work functions call device(), both under the block's d_setlock. */
class gather_columns
{
public:
    gather_columns() = default;
    gather_columns(const gather_columns&) = delete;
    gather_columns& operator=(const gather_columns&) = delete;
    ~gather_columns();

    //! Replace the table. Does no GPU work.
    void set(std::vector<int> cols);

    //! Return the device copy, first uploading on \p stream if set() was called.
    const int* device(cudaStream_t stream);

private:
    std::vector<int> d_host; // latest table from set()
    int* d_device = nullptr; // device copy, d_capacity entries
    size_t d_capacity = 0;   // grown on demand, never shrunk
    bool d_dirty = false;    // d_host changed since the last upload
};

#endif /* INCLUDED_GR_CUDA_GATHER_CUH */
