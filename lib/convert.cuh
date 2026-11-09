/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_CONVERT_CUH
#define INCLUDED_CUDA_CONVERT_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Element-wise type conversions used by the type converter blocks. All counts
// are in scalars (complex values for the complex functions), all pointers are
// device pointers, and every launch is asynchronous on `stream`.

// out[i] = in[i] / scale, for T = int8_t, uint8_t, int16_t or int32_t.
template <typename T>
void exec_int_to_float(
    const T* in, float* out, size_t n, float scale, cudaStream_t stream);

// out[i] = in[i] * scale + bias, rounded to nearest (ties to even) and
// saturated to T, for T = int8_t, uint8_t, int16_t or int32_t. NaN gives 0.
template <typename T>
void exec_float_to_int(
    const float* in, T* out, size_t n, float scale, float bias, cudaStream_t stream);

// out[i] = in[i] * 256 (the char becomes the high byte).
void exec_char_to_short(const int8_t* in, int16_t* out, size_t n, cudaStream_t stream);

// out[i] = in[i] >> 8 (the high byte of the short).
void exec_short_to_char(const int16_t* in, int8_t* out, size_t n, cudaStream_t stream);

enum class complex_part { real, imag, mag, mag_squared, arg };

void exec_complex_to_part(
    const float2* in, float* out, size_t n, complex_part part, cudaStream_t stream);

// Splits into real and imaginary parts; im may be null.
void exec_complex_to_float(
    const float2* in, float* re, float* im, size_t n, cudaStream_t stream);

void exec_complex_to_magphase(
    const float2* in, float* mag, float* phase, size_t n, cudaStream_t stream);

// out[i] = re[i] + j*im[i]; im may be null for a zero imaginary part.
void exec_float_to_complex(
    const float* re, const float* im, float2* out, size_t n, cudaStream_t stream);

void exec_magphase_to_complex(
    const float* mag, const float* phase, float2* out, size_t n, cudaStream_t stream);

// out[i] = (in[2i] + j*in[2i+1]) / scale, with I and Q exchanged if swap is
// set, for T = int8_t or int16_t.
template <typename T>
void exec_interleaved_to_complex(
    const T* in, float2* out, size_t n, float scale, bool swap, cudaStream_t stream);

#endif /* INCLUDED_CUDA_CONVERT_CUH */
