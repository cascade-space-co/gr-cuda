/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/cuda/cuda_error.h>

#include <sstream>
#include <stdexcept>

void check_cuda_errors(cudaError_t rc)
{
    if (rc) {
        std::cerr << "Operation returned code " << int(rc) << ": " << cudaGetErrorName(rc)
                  << " -- " << cudaGetErrorString(rc) << std::endl;
    }
}

void throw_on_cuda_error(const char* context, cudaError_t rc)
{
    std::ostringstream msg;
    msg << context << ": " << cudaGetErrorName(rc) << " -- " << cudaGetErrorString(rc);
    throw std::runtime_error(msg.str());
}
