/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_CUDA_TAG_DEBUG_IMPL_H
#define INCLUDED_CUDA_TAG_DEBUG_IMPL_H

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/tag_debug.h>
#include <gnuradio/thread/thread.h>

namespace gr {
namespace cuda {

class tag_debug_impl : public tag_debug, public cuda_block
{
private:
    const std::string d_name;
    std::vector<tag_t> d_work_tags;
    std::vector<tag_t> d_tags;
    bool d_display;
    bool d_save_all;
    pmt::pmt_t d_filter;
    // Guards the members above: the setters and getters run on the caller's
    // thread (Python, GRC callbacks) while work() runs on the scheduler's.
    mutable gr::thread::mutex d_mutex;

public:
    tag_debug_impl(size_t sizeof_stream_item,
                   const std::string& name,
                   const std::string& key_filter,
                   int num_inputs);
    ~tag_debug_impl() override = default;

    std::vector<tag_t> current_tags() override;
    int num_tags() override;
    void set_display(bool d) override;
    void set_save_all(bool s) override;
    void set_key_filter(const std::string& key_filter) override;
    std::string key_filter() const override;

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace cuda
} // namespace gr

#endif /* INCLUDED_CUDA_TAG_DEBUG_IMPL_H */
