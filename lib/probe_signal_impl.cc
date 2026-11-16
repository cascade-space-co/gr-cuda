/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "probe_signal_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>

namespace gr {
namespace cuda {

template <class T>
typename probe_signal<T>::sptr probe_signal<T>::make()
{
    return gnuradio::make_block_sptr<probe_signal_impl<T>>();
}

template <class T>
probe_signal_impl<T>::probe_signal_impl()
    : sync_block("probe_signal",
                 io_signature::make(1, 1, sizeof(T), cuda_buffer::type),
                 io_signature::make(0, 0, 0)),
      d_probe(sizeof(T), this->d_logger)
{
}

template <class T>
T probe_signal_impl<T>::level() const
{
    T level;
    d_probe.read(&level);
    return level;
}

template <class T>
int probe_signal_impl<T>::work(int noutput_items,
                               gr_vector_const_void_star& input_items,
                               gr_vector_void_star& output_items)
{
    // Only the last sample is copied, queued on d_stream before work()
    // returns, so cuda_buffer keeps it from being overwritten until it lands.
    if (noutput_items > 0) {
        const auto in = static_cast<const T*>(input_items[0]);
        d_probe.capture(in + noutput_items - 1, d_stream);
    }
    return noutput_items;
}

template class probe_signal<std::uint8_t>;
template class probe_signal<std::int16_t>;
template class probe_signal<std::int32_t>;
template class probe_signal<float>;
template class probe_signal<gr_complex>;

} /* namespace cuda */
} /* namespace gr */
