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

#include <gnuradio/cuda/ibv_source.h>

void bind_ibv_source(py::module& m)
{

    using ibv_source = ::gr::cuda::ibv_source;

    py::class_<ibv_source,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<ibv_source>>(m, "ibv_source")

        .def(py::init(&ibv_source::make),
             py::arg("ibv_device"),
             py::arg("udp_port"),
             py::arg("payload_size"),
             py::arg("mcast_group") = "",
             py::arg("netdev") = "")

        ;
}
