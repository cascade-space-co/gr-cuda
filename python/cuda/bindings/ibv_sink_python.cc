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

#include <gnuradio/cuda/ibv_sink.h>

void bind_ibv_sink(py::module& m)
{

    using ibv_sink = ::gr::cuda::ibv_sink;

    py::class_<ibv_sink,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<ibv_sink>>(m, "ibv_sink")

        .def(py::init(&ibv_sink::make),
             py::arg("ibv_device"),
             py::arg("dst_ip"),
             py::arg("dst_port"),
             py::arg("payload_size"),
             py::arg("dst_mac") = "",
             py::arg("mcast_group") = "",
             py::arg("src_port") = 12345)

        ;
}
