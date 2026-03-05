/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/cuda/tee.h>

void bind_tee(py::module& m)
{
    using tee = ::gr::cuda::tee;

    py::class_<tee, gr::sync_block, gr::block, gr::basic_block, std::shared_ptr<tee>>(
        m, "tee")

        .def(py::init(&tee::make), py::arg("itemsize"), py::arg("gpu") = true);
}
