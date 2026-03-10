/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/block.h>
#include <gnuradio/block_detail.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_buffer_reader.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Helper to access protected detail() method
// This trick relies on the fact that we can cast a pointer to a base class
// to a pointer to a derived class to access protected members,
// provided the access specifiers allow it in the derived class.
struct block_accessor : public gr::block {
    static gr::block_detail_sptr get_detail(gr::block* b)
    {
        // static_cast is safe here because we only use it to access a protected member
        // defined in the base class, and we don't dereference the pointer as a
        // block_accessor in a way that depends on block_accessor's specific layout
        // (which is identical to block's).
        return static_cast<block_accessor*>(b)->detail();
    }
};

void register_cuda_stream_wrapper(std::shared_ptr<gr::block> block, size_t stream_ptr)
{
    auto detail = block_accessor::get_detail(block.get());
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Register as producer on all output buffers
    int noutputs = detail->noutputs();
    for (int i = 0; i < noutputs; i++) {
        auto cbuf = std::dynamic_pointer_cast<gr::cuda_buffer>(detail->output(i));
        if (cbuf)
            cbuf->set_producer_stream(stream);
    }

    // Register as consumer on all input buffer readers
    int ninputs = detail->ninputs();
    for (int i = 0; i < ninputs; i++) {
        auto* cbr = dynamic_cast<gr::cuda_buffer_reader*>(detail->input(i).get());
        if (cbr)
            cbr->set_consumer_stream(stream);
    }
}

void bind_cuda_helpers(py::module& m)
{
    m.def("register_cuda_stream",
          &register_cuda_stream_wrapper,
          "Register a CUDA stream with all cuda_buffers on this block for auto-sync",
          py::arg("block"),
          py::arg("stream_ptr"));
}
