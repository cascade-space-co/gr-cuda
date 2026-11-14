/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/cuda/skiphead.h>

void bind_skiphead(py::module& m)
{
    using skiphead = ::gr::cuda::skiphead;

    py::class_<skiphead, gr::block, gr::basic_block, std::shared_ptr<skiphead>>(
        m, "skiphead")
        .def(py::init(&skiphead::make),
             py::arg("itemsize"),
             py::arg("nitems_to_skip"));
}
