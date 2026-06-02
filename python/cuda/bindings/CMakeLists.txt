# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT cuda_sources)
    message(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT cuda_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW("pygccxml" "import pygccxml" PYGCCXML_FOUND)

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(
    APPEND cuda_python_files
    add_python.cc
    copy_python.cc
    fft_python.cc
    multiply_const_python.cc
    load_python.cc
    nop_python.cc
    cuda_helpers_python.cc
    null_source_python.cc
    null_sink_python.cc
    probe_rate_python.cc
    throttle_python.cc
    stream_to_vector_python.cc
    vector_to_stream_python.cc
    streams_to_vector_python.cc
    vector_to_streams_python.cc
    tee_python.cc
    python_bindings.cc
)

GR_PYBIND_MAKE_OOT(cuda ../../.. gr::cuda "${cuda_python_files}")

# copy in bindings .so file for use in QA test module
add_custom_target(
    copy_bindings_for_tests
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/*.so"
        ${CMAKE_BINARY_DIR}/test_modules/gnuradio/cuda/
    DEPENDS
        cuda_python
)

install(
    TARGETS
        cuda_python
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/cuda
    COMPONENT pythonapi
)
