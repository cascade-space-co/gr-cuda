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

#include <gnuradio/cuda/seq_stamp.h>

void bind_seq_stamp(py::module& m)
{
    using seq_stamp = ::gr::cuda::seq_stamp;

    py::class_<seq_stamp,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<seq_stamp>>(m, "seq_stamp")

        .def(py::init(&seq_stamp::make), py::arg("payload_size"))

        ;
}
