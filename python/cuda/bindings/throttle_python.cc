/*
 * Copyright 2026 Free Software Foundation, Inc.
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

#include <gnuradio/cuda/throttle.h>

void bind_throttle(py::module &m) {
  using throttle = ::gr::cuda::throttle;

  py::class_<throttle, gr::sync_block, gr::block, gr::basic_block,
             std::shared_ptr<throttle>>(m, "throttle")
      .def(py::init(&throttle::make), py::arg("itemsize"), py::arg("sample_rate"))
      .def("set_sample_rate", &throttle::set_sample_rate, py::arg("rate"));
}


