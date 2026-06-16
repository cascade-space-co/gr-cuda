/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/cuda/seq_strip.h>

void bind_seq_strip(py::module& m)
{
    using seq_strip = ::gr::cuda::seq_strip;

    py::class_<seq_strip,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<seq_strip>>(m, "seq_strip")

        .def(py::init(&seq_strip::make), py::arg("payload_size"))

        ;
}
