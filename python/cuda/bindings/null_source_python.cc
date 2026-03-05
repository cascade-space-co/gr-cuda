/*
 * Copyright 2025 Cascade Space.
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

#include <gnuradio/cuda/null_source.h>

void bind_null_source(py::module& m)
{

    using null_source = ::gr::cuda::null_source;

    py::class_<null_source,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<null_source>>(m, "null_source")

        .def(py::init(&null_source::make),
             py::arg("sizeof_stream_item"),
             py::arg("num_outputs") = 1,
             py::arg("memset") = true)

        ;
}
