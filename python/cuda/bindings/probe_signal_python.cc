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

#include <gnuradio/cuda/probe_signal.h>
#include <gnuradio/cuda/probe_signal_v.h>

template <typename T>
void bind_probe_signal_template(py::module& m, const char* classname)
{
    using probe_signal = gr::cuda::probe_signal<T>;

    py::class_<probe_signal,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<probe_signal>>(m, classname)
        .def(py::init(&probe_signal::make))
        .def("level", &probe_signal::level, py::call_guard<py::gil_scoped_release>());
}

template <typename T>
void bind_probe_signal_v_template(py::module& m, const char* classname)
{
    using probe_signal_v = gr::cuda::probe_signal_v<T>;

    py::class_<probe_signal_v,
               gr::sync_block,
               gr::block,
               gr::basic_block,
               std::shared_ptr<probe_signal_v>>(m, classname)
        .def(py::init(&probe_signal_v::make), py::arg("size"))
        .def("level", &probe_signal_v::level, py::call_guard<py::gil_scoped_release>());
}

void bind_probe_signal(py::module& m)
{
    bind_probe_signal_template<std::uint8_t>(m, "probe_signal_b");
    bind_probe_signal_template<std::int16_t>(m, "probe_signal_s");
    bind_probe_signal_template<std::int32_t>(m, "probe_signal_i");
    bind_probe_signal_template<float>(m, "probe_signal_f");
    bind_probe_signal_template<gr_complex>(m, "probe_signal_c");

    bind_probe_signal_v_template<std::uint8_t>(m, "probe_signal_vb");
    bind_probe_signal_v_template<std::int16_t>(m, "probe_signal_vs");
    bind_probe_signal_v_template<std::int32_t>(m, "probe_signal_vi");
    bind_probe_signal_v_template<float>(m, "probe_signal_vf");
    bind_probe_signal_v_template<gr_complex>(m, "probe_signal_vc");
}
