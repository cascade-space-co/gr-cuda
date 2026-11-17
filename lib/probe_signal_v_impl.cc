/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "probe_signal_v_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <stdexcept>

namespace gr {
namespace cuda {

template <class T>
typename probe_signal_v<T>::sptr probe_signal_v<T>::make(size_t size)
{
    if (size == 0) {
        throw std::invalid_argument("probe_signal_v: size must be positive");
    }
    return gnuradio::make_block_sptr<probe_signal_v_impl<T>>(size);
}

template <class T>
probe_signal_v_impl<T>::probe_signal_v_impl(size_t size)
    : sync_block("probe_signal_v",
                 io_signature::make(1, 1, size * sizeof(T), cuda_buffer::type),
                 io_signature::make(0, 0, 0)),
      d_size(size),
      d_probe(size * sizeof(T), this->d_logger)
{
}

template <class T>
std::vector<T> probe_signal_v_impl<T>::level() const
{
    std::vector<T> level(d_size);
    d_probe.read(level.data());
    return level;
}

template <class T>
int probe_signal_v_impl<T>::work(int noutput_items,
                                 gr_vector_const_void_star& input_items,
                                 gr_vector_void_star& output_items)
{
    // Only the last vector is copied, queued on d_stream before work()
    // returns, so cuda_buffer keeps it from being overwritten until it lands.
    if (noutput_items > 0) {
        const auto in = static_cast<const T*>(input_items[0]);
        d_probe.capture(in + (noutput_items - 1) * d_size, d_stream);
    }
    return noutput_items;
}

template class probe_signal_v<std::uint8_t>;
template class probe_signal_v<std::int16_t>;
template class probe_signal_v<std::int32_t>;
template class probe_signal_v<float>;
template class probe_signal_v<gr_complex>;

} /* namespace cuda */
} /* namespace gr */
