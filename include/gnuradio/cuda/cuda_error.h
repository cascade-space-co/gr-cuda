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
#include <iostream>

/*!
 * \brief Log a CUDA error to stderr (does not throw).
 *
 * Prints the error code, name, and description to stderr if \p rc
 * indicates a failure.  Used as a lightweight check wrapper throughout
 * the codebase.
 */
void check_cuda_errors(cudaError_t rc);

/*!
 * \brief Format a CUDA error with context and throw std::runtime_error.
 *
 * Always throws — call only when \p rc indicates a failure.
 * The exception message includes \p context, the CUDA error name,
 * and the CUDA error description string.
 *
 * \param context  Human-readable description of the operation that failed.
 * \param rc       The CUDA error code.
 * \throws std::runtime_error
 */
[[noreturn]] void throw_on_cuda_error(const char* context, cudaError_t rc);

#endif