/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "convert.cuh"
#include <gnuradio/cuda/cuda_error.h>
#include <algorithm>
#include <climits>

namespace {

constexpr int block_size = 256;
constexpr int items_per_thread = 4;
constexpr size_t max_grid_size = 65535;

// Conversions are memory bound. Each thread loads items_per_thread elements
// before storing any of them so the loads are in flight together; within a
// pass, consecutive threads touch consecutive elements so accesses coalesce.
// An Op provides value load(i) and store(i, value).
template <typename Op>
__global__ void kernel_map(Op op, size_t n)
{
    const size_t tile = static_cast<size_t>(blockDim.x) * items_per_thread;
    const size_t step = static_cast<size_t>(gridDim.x) * tile;
    for (size_t base = static_cast<size_t>(blockIdx.x) * tile + threadIdx.x; base < n;
         base += step) {
        typename Op::value v[items_per_thread];
#pragma unroll
        for (int k = 0; k < items_per_thread; k++) {
            const size_t i = base + static_cast<size_t>(k) * blockDim.x;
            if (i < n) {
                v[k] = op.load(i);
            }
        }
#pragma unroll
        for (int k = 0; k < items_per_thread; k++) {
            const size_t i = base + static_cast<size_t>(k) * blockDim.x;
            if (i < n) {
                op.store(i, v[k]);
            }
        }
    }
}

template <typename Op>
void launch(const Op& op, size_t n, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    const size_t tile = static_cast<size_t>(block_size) * items_per_thread;
    const size_t grid = std::min((n + tile - 1) / tile, max_grid_size);
    kernel_map<<<grid, block_size, 0, stream>>>(op, n);
    check_cuda_errors(cudaGetLastError(), "convert: kernel launch");
}

// Integer-to-float scales are applied as a multiplication by 1/scale, like
// VOLK's SIMD kernels.
template <typename T>
struct int_to_float_op {
    using value = float;
    const T* __restrict__ in;
    float* __restrict__ out;
    float inv_scale;

    __device__ value load(size_t i) const
    {
        return static_cast<float>(in[i]) * inv_scale;
    }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

template <typename T>
struct int_range;
template <>
struct int_range<int8_t> {
    static constexpr int lo = -128, hi = 127;
};
template <>
struct int_range<uint8_t> {
    static constexpr int lo = 0, hi = 255;
};
template <>
struct int_range<int16_t> {
    static constexpr int lo = -32768, hi = 32767;
};
template <>
struct int_range<int32_t> {
    static constexpr int lo = INT_MIN, hi = INT_MAX;
};

template <typename T>
struct float_to_int_op {
    using value = T;
    const float* __restrict__ in;
    T* __restrict__ out;
    float scale;
    float bias;

    __device__ value load(size_t i) const
    {
        // __float2int_rn is undefined for NaN and out-of-range values, so
        // saturate before converting. float(INT_MAX) rounds up to 2^31, and
        // every float below it fits in an int.
        const float x = in[i] * scale + bias;
        if (isnan(x))
            return T(0);
        if (x >= static_cast<float>(int_range<T>::hi))
            return static_cast<T>(int_range<T>::hi);
        if (x <= static_cast<float>(int_range<T>::lo))
            return static_cast<T>(int_range<T>::lo);
        return static_cast<T>(__float2int_rn(x));
    }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

struct char_to_short_op {
    using value = int16_t;
    const int8_t* __restrict__ in;
    int16_t* __restrict__ out;

    __device__ value load(size_t i) const { return static_cast<int16_t>(in[i] * 256); }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

struct short_to_char_op {
    using value = int8_t;
    const int16_t* __restrict__ in;
    int8_t* __restrict__ out;

    __device__ value load(size_t i) const { return static_cast<int8_t>(in[i] >> 8); }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

template <complex_part Part>
struct complex_to_part_op {
    using value = float;
    const float2* __restrict__ in;
    float* __restrict__ out;

    __device__ value load(size_t i) const
    {
        const float2 z = in[i];
        switch (Part) {
        case complex_part::real:
            return z.x;
        case complex_part::imag:
            return z.y;
        case complex_part::mag:
            return hypotf(z.x, z.y);
        case complex_part::mag_squared:
            return z.x * z.x + z.y * z.y;
        default:
            return atan2f(z.y, z.x);
        }
    }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

struct complex_to_float_op {
    using value = float2;
    const float2* __restrict__ in;
    float* __restrict__ re;
    float* __restrict__ im;

    __device__ value load(size_t i) const { return in[i]; }
    __device__ void store(size_t i, value v) const
    {
        re[i] = v.x;
        im[i] = v.y;
    }
};

struct complex_to_magphase_op {
    using value = float2;
    const float2* __restrict__ in;
    float* __restrict__ mag;
    float* __restrict__ phase;

    __device__ value load(size_t i) const
    {
        const float2 z = in[i];
        return make_float2(hypotf(z.x, z.y), atan2f(z.y, z.x));
    }
    __device__ void store(size_t i, value v) const
    {
        mag[i] = v.x;
        phase[i] = v.y;
    }
};

struct float_to_complex_op {
    using value = float2;
    const float* __restrict__ re;
    const float* __restrict__ im;
    float2* __restrict__ out;

    __device__ value load(size_t i) const { return make_float2(re[i], im[i]); }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

struct real_to_complex_op {
    using value = float;
    const float* __restrict__ re;
    float2* __restrict__ out;

    __device__ value load(size_t i) const { return re[i]; }
    __device__ void store(size_t i, value v) const { out[i] = make_float2(v, 0.0f); }
};

struct magphase_to_complex_op {
    using value = float2;
    const float* __restrict__ mag;
    const float* __restrict__ phase;
    float2* __restrict__ out;

    __device__ value load(size_t i) const
    {
        float s, c;
        sincosf(phase[i], &s, &c);
        const float m = mag[i];
        return make_float2(m * c, m * s);
    }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

template <typename T, bool Swap>
struct interleaved_to_complex_op {
    using value = float2;
    const T* __restrict__ in;
    float2* __restrict__ out;
    float inv_scale;

    __device__ value load(size_t i) const
    {
        const float a = static_cast<float>(in[2 * i]) * inv_scale;
        const float b = static_cast<float>(in[2 * i + 1]) * inv_scale;
        return Swap ? make_float2(b, a) : make_float2(a, b);
    }
    __device__ void store(size_t i, value v) const { out[i] = v; }
};

float reciprocal(float scale) { return static_cast<float>(1.0 / scale); }

} // namespace

template <typename T>
void exec_int_to_float(
    const T* in, float* out, size_t n, float scale, cudaStream_t stream)
{
    launch(int_to_float_op<T>{ in, out, reciprocal(scale) }, n, stream);
}

template <typename T>
void exec_float_to_int(
    const float* in, T* out, size_t n, float scale, float bias, cudaStream_t stream)
{
    launch(float_to_int_op<T>{ in, out, scale, bias }, n, stream);
}

void exec_char_to_short(const int8_t* in, int16_t* out, size_t n, cudaStream_t stream)
{
    launch(char_to_short_op{ in, out }, n, stream);
}

void exec_short_to_char(const int16_t* in, int8_t* out, size_t n, cudaStream_t stream)
{
    launch(short_to_char_op{ in, out }, n, stream);
}

void exec_complex_to_part(
    const float2* in, float* out, size_t n, complex_part part, cudaStream_t stream)
{
    switch (part) {
    case complex_part::real:
        launch(complex_to_part_op<complex_part::real>{ in, out }, n, stream);
        break;
    case complex_part::imag:
        launch(complex_to_part_op<complex_part::imag>{ in, out }, n, stream);
        break;
    case complex_part::mag:
        launch(complex_to_part_op<complex_part::mag>{ in, out }, n, stream);
        break;
    case complex_part::mag_squared:
        launch(complex_to_part_op<complex_part::mag_squared>{ in, out }, n, stream);
        break;
    case complex_part::arg:
        launch(complex_to_part_op<complex_part::arg>{ in, out }, n, stream);
        break;
    }
}

void exec_complex_to_float(
    const float2* in, float* re, float* im, size_t n, cudaStream_t stream)
{
    if (im) {
        launch(complex_to_float_op{ in, re, im }, n, stream);
    } else {
        launch(complex_to_part_op<complex_part::real>{ in, re }, n, stream);
    }
}

void exec_complex_to_magphase(
    const float2* in, float* mag, float* phase, size_t n, cudaStream_t stream)
{
    launch(complex_to_magphase_op{ in, mag, phase }, n, stream);
}

void exec_float_to_complex(
    const float* re, const float* im, float2* out, size_t n, cudaStream_t stream)
{
    if (im) {
        launch(float_to_complex_op{ re, im, out }, n, stream);
    } else {
        launch(real_to_complex_op{ re, out }, n, stream);
    }
}

void exec_magphase_to_complex(
    const float* mag, const float* phase, float2* out, size_t n, cudaStream_t stream)
{
    launch(magphase_to_complex_op{ mag, phase, out }, n, stream);
}

template <typename T>
void exec_interleaved_to_complex(
    const T* in, float2* out, size_t n, float scale, bool swap, cudaStream_t stream)
{
    const float k = reciprocal(scale);
    if (swap) {
        launch(interleaved_to_complex_op<T, true>{ in, out, k }, n, stream);
    } else {
        launch(interleaved_to_complex_op<T, false>{ in, out, k }, n, stream);
    }
}

template void
exec_int_to_float<int8_t>(const int8_t*, float*, size_t, float, cudaStream_t);
template void
exec_int_to_float<uint8_t>(const uint8_t*, float*, size_t, float, cudaStream_t);
template void
exec_int_to_float<int16_t>(const int16_t*, float*, size_t, float, cudaStream_t);
template void
exec_int_to_float<int32_t>(const int32_t*, float*, size_t, float, cudaStream_t);

template void
exec_float_to_int<int8_t>(const float*, int8_t*, size_t, float, float, cudaStream_t);
template void
exec_float_to_int<uint8_t>(const float*, uint8_t*, size_t, float, float, cudaStream_t);
template void
exec_float_to_int<int16_t>(const float*, int16_t*, size_t, float, float, cudaStream_t);
template void
exec_float_to_int<int32_t>(const float*, int32_t*, size_t, float, float, cudaStream_t);

template void exec_interleaved_to_complex<int8_t>(
    const int8_t*, float2*, size_t, float, bool, cudaStream_t);
template void exec_interleaved_to_complex<int16_t>(
    const int16_t*, float2*, size_t, float, bool, cudaStream_t);
