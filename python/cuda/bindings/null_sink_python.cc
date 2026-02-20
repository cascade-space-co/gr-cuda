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

#include <gnuradio/cuda/null_sink.h>

void bind_null_sink(py::module &m) {

  using null_sink = ::gr::cuda::null_sink;

  py::class_<null_sink, gr::sync_block, gr::block, gr::basic_block,
             std::shared_ptr<null_sink>>(m, "null_sink")

      .def(py::init(&null_sink::make), 
           py::arg("sizeof_stream_item"),
           py::arg("num_inputs") = 1)

      ;
}

