/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <gnuradio/cuda/host_null_source.h>

void bind_host_null_source(py::module& m)
{

    using host_null_source = ::gr::cuda::host_null_source;

    py::class_<host_null_source,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<host_null_source>>(m, "host_null_source")

        .def(py::init(&host_null_source::make),
             py::arg("sizeof_stream_item"),
             py::arg("memset") = true)

        ;
}
