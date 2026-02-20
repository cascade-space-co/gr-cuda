/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pybind11/pybind11.h>
#include <gnuradio/cuda/cuda_block_helper.h>
#include <gnuradio/block.h>

namespace py = pybind11;

// Helper to access protected detail() method
// This trick relies on the fact that we can cast a pointer to a base class
// to a pointer to a derived class to access protected members, 
// provided the access specifiers allow it in the derived class.
struct block_accessor : public gr::block {
    static gr::block_detail_sptr get_detail(gr::block* b) {
        // static_cast is safe here because we only use it to access a protected member
        // defined in the base class, and we don't dereference the pointer as a block_accessor
        // in a way that depends on block_accessor's specific layout (which is identical to block's).
        return static_cast<block_accessor*>(b)->detail();
    }
};

void wait_for_work_wrapper(std::shared_ptr<gr::block> block, size_t stream_ptr) {
    auto detail = block_accessor::get_detail(block.get());
    // CuPy exposes stream.ptr as a Python int; pybind11 passes it as size_t.
    // cudaStream_t is a pointer type, so reinterpret_cast is required.
    gr::cuda::wait_for_work(detail, reinterpret_cast<cudaStream_t>(stream_ptr));
}

void mark_work_done_wrapper(std::shared_ptr<gr::block> block, size_t stream_ptr) {
    auto detail = block_accessor::get_detail(block.get());
    gr::cuda::mark_work_done(detail, reinterpret_cast<cudaStream_t>(stream_ptr));
}

void bind_cuda_helpers(py::module& m) {
    m.def("wait_for_work", &wait_for_work_wrapper, "Wait for input CUDA buffers to be ready",
        py::arg("block"), py::arg("stream_ptr"));
    m.def("mark_work_done", &mark_work_done_wrapper, "Mark GPU work as done (outputs ready, inputs consumed)",
        py::arg("block"), py::arg("stream_ptr"));
}
