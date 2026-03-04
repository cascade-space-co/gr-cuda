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
 * \brief Log (optionally) and throw on CUDA failure.
 *
 * If \p rc indicates a CUDA failure, logs the formatted message when
 * \p logger is non-null, then throws std::runtime_error with
 * the supplied \p context, CUDA error name, and CUDA error description.
 * Returns silently when \p rc is cudaSuccess.
 *
 * \param rc       The CUDA error code.
 * \param context  Human-readable description of the failed operation.
 * \param logger   Optional GNU Radio logger for error logging before throw.
 * \throws std::runtime_error
 */
void check_cuda_errors(cudaError_t rc,
                       const char* context = "CUDA operation failed",
                       const std::shared_ptr<gr::logger>& logger = nullptr);

/*!
 * \brief Log (optionally) and throw on CUDA Driver API failure.
 *
 * Same behaviour as the runtime-API overload, but for CUresult codes
 * returned by the CUDA Driver API (cuMem*, cuDevice*, etc.).
 */
void check_cuda_errors(CUresult res,
                       const char* context = "CUDA driver operation failed",
                       const std::shared_ptr<gr::logger>& logger = nullptr);

#endif