/*
 * Copyright 2025 Free Software Foundation, Inc.
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

#include <gnuradio/cuda/probe_rate.h>

void bind_probe_rate(py::module &m) {
  using probe_rate = ::gr::cuda::probe_rate;

  py::class_<probe_rate, gr::sync_block, gr::block, gr::basic_block,
             std::shared_ptr<probe_rate>>(m, "probe_rate")
      .def(py::init(&probe_rate::make), py::arg("itemsize"),
           py::arg("update_rate_ms") = 500.0, py::arg("alpha") = 0.0001,
           py::arg("name") = "")
      .def("set_alpha", &probe_rate::set_alpha, py::arg("alpha"))
      .def("set_name", &probe_rate::set_name, py::arg("name"))
      .def("rate", &probe_rate::rate);
}


