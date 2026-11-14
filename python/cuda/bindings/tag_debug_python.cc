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

#include <gnuradio/cuda/tag_debug.h>

void bind_tag_debug(py::module& m)
{
    using tag_debug = ::gr::cuda::tag_debug;

    py::class_<tag_debug,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<tag_debug>>(m, "tag_debug")
        .def(py::init(&tag_debug::make),
             py::arg("sizeof_stream_item"),
             py::arg("name"),
             py::arg("key_filter") = "",
             py::arg("num_inputs") = 1)
        .def("current_tags", &tag_debug::current_tags)
        .def("num_tags", &tag_debug::num_tags)
        .def("set_display", &tag_debug::set_display, py::arg("d"))
        .def("set_save_all", &tag_debug::set_save_all, py::arg("s"))
        .def("set_key_filter", &tag_debug::set_key_filter, py::arg("key_filter"))
        .def("key_filter", &tag_debug::key_filter);
}
