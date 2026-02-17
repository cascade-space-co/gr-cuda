/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/logger.h>

#include <sstream>
#include <stdexcept>

namespace {
std::string format_cuda_error(const char* context, cudaError_t rc)
{
    std::ostringstream msg;
    msg << context << ": " << cudaGetErrorName(rc) << " -- " << cudaGetErrorString(rc);
    return msg.str();
}
} // namespace

void check_cuda_errors(cudaError_t rc, const char* context)
{
    if (!rc) {
        return;
    }
    throw std::runtime_error(format_cuda_error(context, rc));
}

void check_cuda_errors(cudaError_t rc,
                       const char* context,
                       const std::shared_ptr<gr::logger>& logger)
{
    if (!rc) {
        return;
    }

    const std::string msg = format_cuda_error(context, rc);
    if (logger) {
        logger->error("{}", msg);
    }
    throw std::runtime_error(msg);
}
