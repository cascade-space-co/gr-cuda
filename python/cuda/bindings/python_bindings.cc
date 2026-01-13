/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/pybind11.h>
#include <gnuradio/cuda/cuda_buffer.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Headers for binding functions
/**************************************/
// The following comment block is used for
// gr_modtool to insert function prototypes
// Please do not delete
/**************************************/
// BINDING_FUNCTION_PROTOTYPES(
    void bind_add(py::module& m);
    void bind_copy(py::module &m);
    void bind_multiply_const(py::module& m);
    void bind_load(py::module& m);
    void bind_cuda_helpers(py::module& m);
    void bind_null_source(py::module& m);
    void bind_null_sink(py::module& m);
    void bind_probe_rate(py::module& m);
    void bind_throttle(py::module& m);
// ) END BINDING_FUNCTION_PROTOTYPES


// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(cuda_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Allow access to base block methods
    py::module::import("gnuradio.gr");

    // Bind cuda_buffer to expose the 'type' static member.
    // Note: We intentionally omit the base class gr::buffer_single_mapped from the template
    // arguments because it is not exposed in the standard GNU Radio python bindings.
    // Specifying it would cause an "unknown base type" error in pybind11.
    // Since we only need access to the static 'type' member and don't need upcasting
    // in Python, binding it as a standalone class (held by shared_ptr) is sufficient.
    py::class_<gr::cuda_buffer, std::shared_ptr<gr::cuda_buffer>>(m, "cuda_buffer")
        .def_readonly_static("type", &gr::cuda_buffer::type);

    /**************************************/
    // The following comment block is used for
    // gr_modtool to insert binding function calls
    // Please do not delete
    /**************************************/
    // BINDING_FUNCTION_CALLS(
    bind_add(m);
    bind_copy(m);
    bind_multiply_const(m);
    bind_load(m);
    bind_cuda_helpers(m);
    bind_null_source(m);
    bind_null_sink(m);
    bind_probe_rate(m);
    bind_throttle(m);
    // ) END BINDING_FUNCTION_CALLS
}