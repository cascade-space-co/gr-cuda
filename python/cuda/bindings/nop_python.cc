/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/cuda/nop.h>

void bind_nop(py::module& m)
{
    using nop = ::gr::cuda::nop;

    py::class_<nop, gr::sync_block, gr::block, gr::basic_block, std::shared_ptr<nop>>(
        m, "nop")

        .def(py::init(&nop::make), py::arg("sizeof_stream_item"));
}
