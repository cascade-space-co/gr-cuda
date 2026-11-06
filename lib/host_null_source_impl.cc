/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "host_null_source_impl.h"
#include <gnuradio/io_signature.h>
#include <cstring>

namespace gr {
namespace cuda {

host_null_source::sptr host_null_source::make(size_t sizeof_stream_item, bool memset)
{
    return gnuradio::make_block_sptr<host_null_source_impl>(sizeof_stream_item, memset);
}

host_null_source_impl::host_null_source_impl(size_t sizeof_stream_item, bool memset)
    : sync_block("host_null_source",
                 io_signature::make(0, 0, 0),
                 io_signature::make(1, 1, sizeof_stream_item)),
      d_itemsize(sizeof_stream_item),
      d_memset(memset)
{
}

int host_null_source_impl::work(int noutput_items,
                                gr_vector_const_void_star& input_items,
                                gr_vector_void_star& output_items)
{
    if (d_memset)
        std::memset(
            output_items[0], 0, static_cast<size_t>(noutput_items) * d_itemsize);

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
