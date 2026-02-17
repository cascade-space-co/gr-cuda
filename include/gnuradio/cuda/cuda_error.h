/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef _INCLUDED_GR_CUDA_ERROR
#define _INCLUDED_GR_CUDA_ERROR

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>

namespace gr {
class logger;
}
/*!
 * \brief Throw on CUDA failure.
 *
 * If \p rc indicates a CUDA failure, throws std::runtime_error with
 * the supplied \p context, CUDA error name, and CUDA error description.
 * Returns silently when \p rc is cudaSuccess.
 *
 * \param rc       The CUDA error code.
 * \param context  Human-readable description of the failed operation.
 * \throws std::runtime_error
 */
void check_cuda_errors(cudaError_t rc, const char* context = "CUDA operation failed");

/*!
 * \brief Log via GNU Radio logger and throw on CUDA failure.
 *
 * If \p rc indicates a CUDA failure, logs the formatted message when
 * \p logger is non-null, then throws std::runtime_error.
 */
void check_cuda_errors(cudaError_t rc,
                       const char* context,
                       const std::shared_ptr<gr::logger>& logger);

#endif