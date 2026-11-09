/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_TYPE_CONVERTERS_IMPL_H
#define INCLUDED_CUDA_TYPE_CONVERTERS_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/type_converters.h>
#include <atomic>

namespace gr {
namespace cuda {

// Scales and swap flags are atomic because setters run on the caller's thread
// while work() runs on the scheduler's.

#define CUDA_CONVERTER_WORK                          \
    int work(int noutput_items,                      \
             gr_vector_const_void_star& input_items, \
             gr_vector_void_star& output_items) override

#define CUDA_SCALED_CONVERTER_IMPL(name)                          \
    class name##_impl : public name, public cuda_block            \
    {                                                             \
        const size_t d_vlen;                                      \
        std::atomic<float> d_scale;                               \
                                                                  \
    public:                                                       \
        name##_impl(size_t vlen, float scale);                    \
        float scale() const override { return d_scale; }          \
        void set_scale(float scale) override { d_scale = scale; } \
        CUDA_CONVERTER_WORK;                                      \
    }

#define CUDA_PLAIN_CONVERTER_IMPL(name)                \
    class name##_impl : public name, public cuda_block \
    {                                                  \
        const size_t d_vlen;                           \
                                                       \
    public:                                            \
        explicit name##_impl(size_t vlen);             \
        CUDA_CONVERTER_WORK;                           \
    }

CUDA_SCALED_CONVERTER_IMPL(char_to_float);
CUDA_SCALED_CONVERTER_IMPL(uchar_to_float);
CUDA_SCALED_CONVERTER_IMPL(short_to_float);
CUDA_SCALED_CONVERTER_IMPL(int_to_float);
CUDA_SCALED_CONVERTER_IMPL(float_to_char);
CUDA_SCALED_CONVERTER_IMPL(float_to_short);
CUDA_SCALED_CONVERTER_IMPL(float_to_int);

CUDA_PLAIN_CONVERTER_IMPL(char_to_short);
CUDA_PLAIN_CONVERTER_IMPL(short_to_char);
CUDA_PLAIN_CONVERTER_IMPL(float_to_complex);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_float);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_real);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_imag);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_mag);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_mag_squared);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_arg);
CUDA_PLAIN_CONVERTER_IMPL(complex_to_magphase);
CUDA_PLAIN_CONVERTER_IMPL(magphase_to_complex);

class float_to_uchar_impl : public float_to_uchar, public cuda_block
{
    const size_t d_vlen;
    std::atomic<float> d_scale;
    std::atomic<float> d_bias;

public:
    float_to_uchar_impl(size_t vlen, float scale, float bias);
    float scale() const override { return d_scale; }
    float bias() const override { return d_bias; }
    void set_scale(float scale) override { d_scale = scale; }
    void set_bias(float bias) override { d_bias = bias; }
    CUDA_CONVERTER_WORK;
};

#define CUDA_TO_INTERLEAVED_IMPL(name)                                 \
    class name##_impl : public name, public cuda_block                 \
    {                                                                  \
        const bool d_vector;                                           \
        std::atomic<float> d_scale_factor;                             \
                                                                       \
    public:                                                            \
        name##_impl(bool vector, float scale_factor);                  \
        float scale_factor() const override { return d_scale_factor; } \
        void set_scale_factor(float scale_factor) override             \
        {                                                              \
            d_scale_factor = scale_factor;                             \
        }                                                              \
        CUDA_CONVERTER_WORK;                                           \
    }

CUDA_TO_INTERLEAVED_IMPL(complex_to_interleaved_char);
CUDA_TO_INTERLEAVED_IMPL(complex_to_interleaved_short);

#define CUDA_FROM_INTERLEAVED_IMPL(name)                               \
    class name##_impl : public name, public cuda_block                 \
    {                                                                  \
        std::atomic<float> d_scale_factor;                             \
        std::atomic<bool> d_swap;                                      \
                                                                       \
    public:                                                            \
        name##_impl(bool vector_input, float scale_factor, bool swap); \
        float scale_factor() const override { return d_scale_factor; } \
        bool swap() const override { return d_swap; }                  \
        void set_scale_factor(float scale_factor) override             \
        {                                                              \
            d_scale_factor = scale_factor;                             \
        }                                                              \
        void set_swap(bool swap) override { d_swap = swap; }           \
        CUDA_CONVERTER_WORK;                                           \
    }

CUDA_FROM_INTERLEAVED_IMPL(interleaved_char_to_complex);
CUDA_FROM_INTERLEAVED_IMPL(interleaved_short_to_complex);

#undef CUDA_FROM_INTERLEAVED_IMPL
#undef CUDA_TO_INTERLEAVED_IMPL
#undef CUDA_PLAIN_CONVERTER_IMPL
#undef CUDA_SCALED_CONVERTER_IMPL
#undef CUDA_CONVERTER_WORK

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_TYPE_CONVERTERS_IMPL_H */
