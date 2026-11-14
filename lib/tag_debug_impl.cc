/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "tag_debug_impl.h"
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace gr {
namespace cuda {

tag_debug::sptr tag_debug::make(size_t sizeof_stream_item,
                                const std::string& name,
                                const std::string& key_filter,
                                int num_inputs)
{
    return gnuradio::make_block_sptr<tag_debug_impl>(
        sizeof_stream_item, name, key_filter, num_inputs);
}

tag_debug_impl::tag_debug_impl(size_t sizeof_stream_item,
                               const std::string& name,
                               const std::string& key_filter,
                               int num_inputs)
    : sync_block("tag_debug",
                 io_signature::make(
                     num_inputs, num_inputs, sizeof_stream_item, cuda_buffer::type),
                 io_signature::make(0, 0, 0)),
      d_name(name),
      d_display(true),
      d_save_all(false)
{
    set_key_filter(key_filter);
}

std::vector<tag_t> tag_debug_impl::current_tags()
{
    gr::thread::scoped_lock l(d_mutex);
    return d_save_all ? d_tags : d_work_tags;
}

// Counts the tags input 0's buffer still holds, not every tag ever received:
// GNU Radio prunes tags that fall more than a buffer's length behind the
// producer.  Use set_save_all(true) and current_tags() for a running total.
int tag_debug_impl::num_tags()
{
    std::vector<tag_t> t;
    get_tags_in_range(t, 0, 0, nitems_read(0));
    return static_cast<int>(t.size());
}

void tag_debug_impl::set_display(bool d)
{
    gr::thread::scoped_lock l(d_mutex);
    d_display = d;
}

// Turning save_all off discards the tags stored so far.
void tag_debug_impl::set_save_all(bool s)
{
    gr::thread::scoped_lock l(d_mutex);
    if (d_save_all && !s) {
        d_tags.clear();
    }
    d_save_all = s;
}

// An empty key is stored as PMT_NIL, which work() takes to mean "no filter".
void tag_debug_impl::set_key_filter(const std::string& key_filter)
{
    gr::thread::scoped_lock l(d_mutex);
    d_filter = key_filter.empty() ? pmt::PMT_NIL : pmt::intern(key_filter);
}

std::string tag_debug_impl::key_filter() const
{
    gr::thread::scoped_lock l(d_mutex);
    return pmt::is_null(d_filter) ? "" : pmt::symbol_to_string(d_filter);
}

int tag_debug_impl::work(int noutput_items,
                         gr_vector_const_void_star& input_items,
                         gr_vector_void_star& output_items)
{
    // Tags are host-side metadata, so the samples in GPU memory are never read
    // and the input edge needs no device-to-host copy.
    gr::thread::scoped_lock l(d_mutex);

    std::stringstream sout;
    if (d_display) {
        sout << std::endl
             << "----------------------------------------------------------------------"
             << std::endl
             << "Tag Debug (CUDA): " << d_name << std::endl;
    }

    // Collect the tags of every input; blocks.tag_debug keeps only the last
    // input's.  get_tags_in_range() clears `tags` before filling it.
    d_work_tags.clear();
    std::vector<tag_t> tags;
    for (size_t i = 0; i < input_items.size(); i++) {
        // A sync block consumes noutput_items from every input, so this call
        // covers [start, end) on each of them.
        const uint64_t start = nitems_read(i);
        const uint64_t end = start + static_cast<uint64_t>(noutput_items);
        if (pmt::is_null(d_filter)) {
            get_tags_in_range(tags, i, start, end);
        } else {
            get_tags_in_range(tags, i, start, end, d_filter);
        }

        if (d_display) {
            sout << "Input Stream: " << std::setw(2) << std::setfill('0') << i
                 << std::setfill(' ') << std::endl;
            for (const auto& tag : tags) {
                sout << std::setw(10) << "Offset: " << tag.offset << std::setw(10)
                     << "Source: "
                     << (pmt::is_symbol(tag.srcid) ? pmt::symbol_to_string(tag.srcid)
                                                   : "n/a")
                     << std::setw(10) << "Key: " << pmt::symbol_to_string(tag.key)
                     << std::setw(10) << "Value: " << tag.value << std::endl;
            }
        }

        d_work_tags.insert(d_work_tags.end(), tags.begin(), tags.end());
    }

    if (d_save_all) {
        d_tags.insert(d_tags.end(), d_work_tags.begin(), d_work_tags.end());
    }

    // Print only when there is something to show, so tag-free calls stay quiet.
    if (d_display && !d_work_tags.empty()) {
        sout << "----------------------------------------------------------------------"
             << std::endl;
        std::cout << sout.str();
    }

    return noutput_items;
}

} /* namespace cuda */
} /* namespace gr */
