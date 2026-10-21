/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "convert.cuh"
#include "type_converters_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

namespace {

io_signature::sptr sig(int ports, size_t itemsize)
{
    return io_signature::make(ports, ports, itemsize, cuda_buffer::type);
}

io_signature::sptr sig(int min_ports, int max_ports, size_t itemsize)
{
    return io_signature::make(min_ports, max_ports, itemsize, cuda_buffer::type);
}

template <typename T>
const T* in_ptr(gr_vector_const_void_star& items, size_t port = 0)
{
    return static_cast<const T*>(items[port]);
}

template <typename T>
T* out_ptr(gr_vector_void_star& items, size_t port = 0)
{
    return static_cast<T*>(items[port]);
}

} // namespace

#define CUDA_MAKE_VLEN(name)                                 \
    name::sptr name::make(size_t vlen)                       \
    {                                                        \
        return gnuradio::make_block_sptr<name##_impl>(vlen); \
    }

#define CUDA_MAKE_SCALED(name)                                      \
    name::sptr name::make(size_t vlen, float scale)                 \
    {                                                               \
        return gnuradio::make_block_sptr<name##_impl>(vlen, scale); \
    }

// One input and one output of vlen-element vectors.
#define CUDA_SCALED_CTOR(name, in_t, out_t)                                      \
    name##_impl::name##_impl(size_t vlen, float scale)                           \
        : gr::sync_block(                                                        \
              #name, sig(1, vlen * sizeof(in_t)), sig(1, vlen * sizeof(out_t))), \
          d_vlen(vlen),                                                          \
          d_scale(scale)                                                         \
    {                                                                            \
    }

#define CUDA_PLAIN_CTOR(name, in_ports, in_t, out_ports, out_t)    \
    name##_impl::name##_impl(size_t vlen)                          \
        : gr::sync_block(#name,                                    \
                         sig(1, in_ports, vlen * sizeof(in_t)),    \
                         sig(1, out_ports, vlen * sizeof(out_t))), \
          d_vlen(vlen)                                             \
    {                                                              \
    }

#define CUDA_WORK(name)                                           \
    int name##_impl::work(int noutput_items,                      \
                          gr_vector_const_void_star& input_items, \
                          gr_vector_void_star& output_items)

// ---------------------------------------------------------------------------
// Integer to float

CUDA_MAKE_SCALED(char_to_float)
CUDA_SCALED_CTOR(char_to_float, int8_t, float)
CUDA_WORK(char_to_float)
{
    exec_int_to_float(in_ptr<int8_t>(input_items),
                      out_ptr<float>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      d_stream);
    return noutput_items;
}

CUDA_MAKE_SCALED(uchar_to_float)
CUDA_SCALED_CTOR(uchar_to_float, uint8_t, float)
CUDA_WORK(uchar_to_float)
{
    exec_int_to_float(in_ptr<uint8_t>(input_items),
                      out_ptr<float>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      d_stream);
    return noutput_items;
}

CUDA_MAKE_SCALED(short_to_float)
CUDA_SCALED_CTOR(short_to_float, int16_t, float)
CUDA_WORK(short_to_float)
{
    exec_int_to_float(in_ptr<int16_t>(input_items),
                      out_ptr<float>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      d_stream);
    return noutput_items;
}

CUDA_MAKE_SCALED(int_to_float)
CUDA_SCALED_CTOR(int_to_float, int32_t, float)
CUDA_WORK(int_to_float)
{
    exec_int_to_float(in_ptr<int32_t>(input_items),
                      out_ptr<float>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      d_stream);
    return noutput_items;
}

// ---------------------------------------------------------------------------
// Float to integer

CUDA_MAKE_SCALED(float_to_char)
CUDA_SCALED_CTOR(float_to_char, float, int8_t)
CUDA_WORK(float_to_char)
{
    exec_float_to_int(in_ptr<float>(input_items),
                      out_ptr<int8_t>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      0.0f,
                      d_stream);
    return noutput_items;
}

CUDA_MAKE_SCALED(float_to_short)
CUDA_SCALED_CTOR(float_to_short, float, int16_t)
CUDA_WORK(float_to_short)
{
    exec_float_to_int(in_ptr<float>(input_items),
                      out_ptr<int16_t>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      0.0f,
                      d_stream);
    return noutput_items;
}

CUDA_MAKE_SCALED(float_to_int)
CUDA_SCALED_CTOR(float_to_int, float, int32_t)
CUDA_WORK(float_to_int)
{
    exec_float_to_int(in_ptr<float>(input_items),
                      out_ptr<int32_t>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      0.0f,
                      d_stream);
    return noutput_items;
}

float_to_uchar::sptr float_to_uchar::make(size_t vlen, float scale, float bias)
{
    return gnuradio::make_block_sptr<float_to_uchar_impl>(vlen, scale, bias);
}

float_to_uchar_impl::float_to_uchar_impl(size_t vlen, float scale, float bias)
    : gr::sync_block("float_to_uchar",
                     sig(1, vlen * sizeof(float)),
                     sig(1, vlen * sizeof(uint8_t))),
      d_vlen(vlen),
      d_scale(scale),
      d_bias(bias)
{
}

CUDA_WORK(float_to_uchar)
{
    exec_float_to_int(in_ptr<float>(input_items),
                      out_ptr<uint8_t>(output_items),
                      noutput_items * d_vlen,
                      d_scale,
                      d_bias,
                      d_stream);
    return noutput_items;
}

// ---------------------------------------------------------------------------
// Integer to integer

CUDA_MAKE_VLEN(char_to_short)
CUDA_PLAIN_CTOR(char_to_short, 1, int8_t, 1, int16_t)
CUDA_WORK(char_to_short)
{
    exec_char_to_short(in_ptr<int8_t>(input_items),
                       out_ptr<int16_t>(output_items),
                       noutput_items * d_vlen,
                       d_stream);
    return noutput_items;
}

CUDA_MAKE_VLEN(short_to_char)
CUDA_PLAIN_CTOR(short_to_char, 1, int16_t, 1, int8_t)
CUDA_WORK(short_to_char)
{
    exec_short_to_char(in_ptr<int16_t>(input_items),
                       out_ptr<int8_t>(output_items),
                       noutput_items * d_vlen,
                       d_stream);
    return noutput_items;
}

// ---------------------------------------------------------------------------
// Complex and float

CUDA_MAKE_VLEN(float_to_complex)
CUDA_PLAIN_CTOR(float_to_complex, 2, float, 1, gr_complex)
CUDA_WORK(float_to_complex)
{
    exec_float_to_complex(in_ptr<float>(input_items),
                          input_items.size() > 1 ? in_ptr<float>(input_items, 1)
                                                 : nullptr,
                          out_ptr<float2>(output_items),
                          noutput_items * d_vlen,
                          d_stream);
    return noutput_items;
}

CUDA_MAKE_VLEN(complex_to_float)
CUDA_PLAIN_CTOR(complex_to_float, 1, gr_complex, 2, float)
CUDA_WORK(complex_to_float)
{
    exec_complex_to_float(in_ptr<float2>(input_items),
                          out_ptr<float>(output_items),
                          output_items.size() > 1 ? out_ptr<float>(output_items, 1)
                                                  : nullptr,
                          noutput_items * d_vlen,
                          d_stream);
    return noutput_items;
}

#define CUDA_COMPLEX_PART(name, part)                      \
    CUDA_MAKE_VLEN(name)                                   \
    CUDA_PLAIN_CTOR(name, 1, gr_complex, 1, float)         \
    CUDA_WORK(name)                                        \
    {                                                      \
        exec_complex_to_part(in_ptr<float2>(input_items),  \
                             out_ptr<float>(output_items), \
                             noutput_items* d_vlen,        \
                             part,                         \
                             d_stream);                    \
        return noutput_items;                              \
    }

CUDA_COMPLEX_PART(complex_to_real, complex_part::real)
CUDA_COMPLEX_PART(complex_to_imag, complex_part::imag)
CUDA_COMPLEX_PART(complex_to_mag, complex_part::mag)
CUDA_COMPLEX_PART(complex_to_mag_squared, complex_part::mag_squared)
CUDA_COMPLEX_PART(complex_to_arg, complex_part::arg)

CUDA_MAKE_VLEN(complex_to_magphase)
CUDA_PLAIN_CTOR(complex_to_magphase, 1, gr_complex, 2, float)
CUDA_WORK(complex_to_magphase)
{
    exec_complex_to_magphase(in_ptr<float2>(input_items),
                             out_ptr<float>(output_items, 0),
                             out_ptr<float>(output_items, 1),
                             noutput_items * d_vlen,
                             d_stream);
    return noutput_items;
}

CUDA_MAKE_VLEN(magphase_to_complex)
CUDA_PLAIN_CTOR(magphase_to_complex, 2, float, 1, gr_complex)
CUDA_WORK(magphase_to_complex)
{
    exec_magphase_to_complex(in_ptr<float>(input_items, 0),
                             in_ptr<float>(input_items, 1),
                             out_ptr<float2>(output_items),
                             noutput_items * d_vlen,
                             d_stream);
    return noutput_items;
}

// ---------------------------------------------------------------------------
// Interleaved I/Q

#define CUDA_TO_INTERLEAVED(name, out_t)                                          \
    name::sptr name::make(bool vector, float scale_factor)                        \
    {                                                                             \
        return gnuradio::make_block_sptr<name##_impl>(vector, scale_factor);      \
    }                                                                             \
    name##_impl::name##_impl(bool vector, float scale_factor)                     \
        : gr::sync_interpolator(#name,                                            \
                                sig(1, sizeof(gr_complex)),                       \
                                sig(1, (vector ? 2 : 1) * sizeof(out_t)),         \
                                vector ? 1 : 2),                                  \
          d_vector(vector),                                                       \
          d_scale_factor(scale_factor)                                            \
    {                                                                             \
    }                                                                             \
    CUDA_WORK(name)                                                               \
    {                                                                             \
        const size_t n = static_cast<size_t>(noutput_items) * (d_vector ? 2 : 1); \
        exec_float_to_int(in_ptr<float>(input_items),                             \
                          out_ptr<out_t>(output_items),                           \
                          n,                                                      \
                          d_scale_factor,                                         \
                          0.0f,                                                   \
                          d_stream);                                              \
        return noutput_items;                                                     \
    }

CUDA_TO_INTERLEAVED(complex_to_interleaved_char, int8_t)
CUDA_TO_INTERLEAVED(complex_to_interleaved_short, int16_t)

#define CUDA_FROM_INTERLEAVED(name, in_t)                                      \
    name##_impl::name##_impl(bool vector_input, float scale_factor, bool swap) \
        : gr::sync_decimator(#name,                                            \
                             sig(1, (vector_input ? 2 : 1) * sizeof(in_t)),    \
                             sig(1, sizeof(gr_complex)),                       \
                             vector_input ? 1 : 2),                            \
          d_scale_factor(scale_factor),                                        \
          d_swap(swap)                                                         \
    {                                                                          \
    }                                                                          \
    CUDA_WORK(name)                                                            \
    {                                                                          \
        exec_interleaved_to_complex(in_ptr<in_t>(input_items),                 \
                                    out_ptr<float2>(output_items),             \
                                    noutput_items,                             \
                                    d_scale_factor,                            \
                                    d_swap,                                    \
                                    d_stream);                                 \
        return noutput_items;                                                  \
    }

interleaved_char_to_complex::sptr
interleaved_char_to_complex::make(bool vector_input, float scale_factor, bool swap)
{
    return gnuradio::make_block_sptr<interleaved_char_to_complex_impl>(
        vector_input, scale_factor, swap);
}
CUDA_FROM_INTERLEAVED(interleaved_char_to_complex, int8_t)

interleaved_short_to_complex::sptr
interleaved_short_to_complex::make(bool vector_input, bool swap, float scale_factor)
{
    return gnuradio::make_block_sptr<interleaved_short_to_complex_impl>(
        vector_input, scale_factor, swap);
}
CUDA_FROM_INTERLEAVED(interleaved_short_to_complex, int16_t)

} /* namespace cuda */
} /* namespace gr */
