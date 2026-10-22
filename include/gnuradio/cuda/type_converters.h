/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_CUDA_TYPE_CONVERTERS_H
#define INCLUDED_CUDA_TYPE_CONVERTERS_H

#include <gnuradio/cuda/api.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/sync_decimator.h>
#include <gnuradio/sync_interpolator.h>

/*!
 * \file
 * GPU equivalents of the gr::blocks type converters, operating on CUDA
 * buffers. Each block takes the same arguments as its gr::blocks counterpart
 * and produces the same results, except where noted.
 *
 * Float-to-integer conversions are rounded to nearest (ties to even) and
 * saturated to the output type. Integer-to-float scales divide;
 * float-to-integer scales multiply.
 */

namespace gr {
namespace cuda {

/*!
 * \brief Convert signed chars to floats: out = in / scale
 * \ingroup cuda
 */
class CUDA_API char_to_float : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<char_to_float> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Convert signed chars to shorts: out = in * 256
 * \ingroup cuda
 *
 * The char becomes the high byte of the short, as in gr::blocks.
 */
class CUDA_API char_to_short : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<char_to_short> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Convert unsigned chars to floats: out = in / scale
 * \ingroup cuda
 *
 * Unlike gr::blocks::uchar_to_float, supports vectors and a scale.
 */
class CUDA_API uchar_to_float : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<uchar_to_float> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Convert shorts to signed chars: out = in >> 8
 * \ingroup cuda
 *
 * Keeps the high byte of each short, as in gr::blocks.
 */
class CUDA_API short_to_char : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<short_to_char> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Convert shorts to floats: out = in / scale
 * \ingroup cuda
 */
class CUDA_API short_to_float : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<short_to_float> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Convert 32-bit ints to floats: out = in / scale
 * \ingroup cuda
 */
class CUDA_API int_to_float : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<int_to_float> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Convert floats to signed chars: out = saturate(round(in * scale))
 * \ingroup cuda
 */
class CUDA_API float_to_char : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<float_to_char> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Convert floats to unsigned chars:
 * out = saturate(round(in * scale + bias))
 * \ingroup cuda
 */
class CUDA_API float_to_uchar : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<float_to_uchar> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f, float bias = 0.0f);
    virtual float scale() const = 0;
    virtual float bias() const = 0;
    virtual void set_scale(float scale) = 0;
    virtual void set_bias(float bias) = 0;
};

/*!
 * \brief Convert floats to shorts: out = saturate(round(in * scale))
 * \ingroup cuda
 */
class CUDA_API float_to_short : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<float_to_short> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Convert floats to 32-bit ints: out = saturate(round(in * scale))
 * \ingroup cuda
 */
class CUDA_API float_to_int : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<float_to_int> sptr;
    static sptr make(size_t vlen = 1, float scale = 1.0f);
    virtual float scale() const = 0;
    virtual void set_scale(float scale) = 0;
};

/*!
 * \brief Combine real and optional imaginary floats into complex
 * \ingroup cuda
 *
 * With only the first input connected, the imaginary part is zero.
 */
class CUDA_API float_to_complex : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<float_to_complex> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Split complex into real and optional imaginary floats
 * \ingroup cuda
 */
class CUDA_API complex_to_float : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_float> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Real part of complex
 * \ingroup cuda
 */
class CUDA_API complex_to_real : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_real> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Imaginary part of complex
 * \ingroup cuda
 */
class CUDA_API complex_to_imag : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_imag> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Magnitude of complex
 * \ingroup cuda
 *
 * Uses hypotf, so unlike gr::blocks it does not overflow to inf or underflow
 * to 0 when re^2 + im^2 is outside the float range but the magnitude is not.
 */
class CUDA_API complex_to_mag : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_mag> sptr;
    static sptr make(size_t vlen = 1);
};
/*!
 * \brief Squared magnitude of complex
 * \ingroup cuda
 */
class CUDA_API complex_to_mag_squared : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_mag_squared> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Phase of complex, in radians
 * \ingroup cuda
 *
 * Uses the exact atan2 rather than the approximation in gr::blocks.
 */
class CUDA_API complex_to_arg : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_arg> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Magnitude and phase of complex, on two outputs
 * \ingroup cuda
 *
 * Uses the exact atan2 rather than the approximation in gr::blocks, and
 * hypotf for the magnitude, as in complex_to_mag.
 */
class CUDA_API complex_to_magphase : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex_to_magphase> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Combine magnitude and phase inputs into complex
 * \ingroup cuda
 */
class CUDA_API magphase_to_complex : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<magphase_to_complex> sptr;
    static sptr make(size_t vlen = 1);
};

/*!
 * \brief Convert complex to interleaved I/Q signed chars:
 * out = saturate(round(in * scale_factor))
 * \ingroup cuda
 *
 * Outputs two chars per complex, or one vector of two if \p vector is true.
 */
class CUDA_API complex_to_interleaved_char : virtual public gr::sync_interpolator
{
public:
    typedef std::shared_ptr<complex_to_interleaved_char> sptr;
    static sptr make(bool vector = false, float scale_factor = 1.0f);
    virtual float scale_factor() const = 0;
    virtual void set_scale_factor(float scale_factor) = 0;
};

/*!
 * \brief Convert complex to interleaved I/Q shorts:
 * out = saturate(round(in * scale_factor))
 * \ingroup cuda
 *
 * Outputs two shorts per complex, or one vector of two if \p vector is true.
 */
class CUDA_API complex_to_interleaved_short : virtual public gr::sync_interpolator
{
public:
    typedef std::shared_ptr<complex_to_interleaved_short> sptr;
    static sptr make(bool vector = false, float scale_factor = 1.0f);
    virtual float scale_factor() const = 0;
    virtual void set_scale_factor(float scale_factor) = 0;
};

/*!
 * \brief Convert interleaved I/Q signed chars to complex:
 * out = (I + jQ) / scale_factor
 * \ingroup cuda
 *
 * Reads two chars per complex, or one vector of two if \p vector_input is
 * true. If \p swap is true, the first char of each pair is Q. Unlike
 * gr::blocks, supports swap.
 */
class CUDA_API interleaved_char_to_complex : virtual public gr::sync_decimator
{
public:
    typedef std::shared_ptr<interleaved_char_to_complex> sptr;
    static sptr
    make(bool vector_input = false, float scale_factor = 1.0f, bool swap = false);
    virtual float scale_factor() const = 0;
    virtual bool swap() const = 0;
    virtual void set_scale_factor(float scale_factor) = 0;
    virtual void set_swap(bool swap) = 0;
};

/*!
 * \brief Convert interleaved I/Q shorts to complex:
 * out = (I + jQ) / scale_factor
 * \ingroup cuda
 *
 * Reads two shorts per complex, or one vector of two if \p vector_input is
 * true. If \p swap is true, the first short of each pair is Q.
 */
class CUDA_API interleaved_short_to_complex : virtual public gr::sync_decimator
{
public:
    typedef std::shared_ptr<interleaved_short_to_complex> sptr;
    static sptr
    make(bool vector_input = false, bool swap = false, float scale_factor = 1.0f);
    virtual float scale_factor() const = 0;
    virtual bool swap() const = 0;
    virtual void set_scale_factor(float scale_factor) = 0;
    virtual void set_swap(bool swap) = 0;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_TYPE_CONVERTERS_H */
