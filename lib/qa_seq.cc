/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Unit tests for the seq_stamp / seq_strip blocks.
 *
 * seq_stamp writes an incrementing 64-bit sequence number into the first
 * 8 bytes of each (payload_size-strided) output item; seq_strip extracts
 * those 8 bytes back into a uint64 stream.  They are GPU-backed sync
 * blocks but need no IBV/NIC, so we drive them through the GR scheduler
 * with a tiny GPU source and a host sink.
 */

#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/cuda/seq_stamp.h>
#include <gnuradio/cuda/seq_strip.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/top_block.h>

#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// Finite GPU source: emits `count` items of `item_size` bytes on a
// cuda_buffer output. Content is irrelevant (seq_stamp overwrites the
// sequence-number bytes), but we zero it so the stream carries real work.
class byte_source : public gr::sync_block, public gr::cuda_block
{
    int item_size_;
    long remaining_;

public:
    byte_source(int item_size, long count)
        : gr::sync_block(
              "byte_source",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(1, 1, item_size, gr::cuda_buffer::type)),
          item_size_(item_size),
          remaining_(count)
    {
    }

    int work(int noutput_items,
             gr_vector_const_void_star& /*input_items*/,
             gr_vector_void_star& output_items) override
    {
        if (remaining_ <= 0)
            return WORK_DONE;
        int n = static_cast<int>(std::min<long>(remaining_, noutput_items));
        check_cuda_errors(
            cudaMemsetAsync(
                output_items[0], 0, static_cast<size_t>(n) * item_size_, d_stream),
            "byte_source: cudaMemsetAsync");
        remaining_ -= n;
        return n;
    }
};

// Host sink collecting a uint64 stream.
class u64_sink : public gr::sync_block
{
    std::vector<uint64_t> data_;

public:
    u64_sink()
        : gr::sync_block("u64_sink",
                         gr::io_signature::make(1, 1, sizeof(uint64_t)),
                         gr::io_signature::make(0, 0, 0))
    {
    }

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& /*output_items*/) override
    {
        auto in = static_cast<const uint64_t*>(input_items[0]);
        data_.insert(data_.end(), in, in + noutput_items);
        return noutput_items;
    }

    const std::vector<uint64_t>& data() const { return data_; }
};

// Run src -> seq_stamp -> seq_strip -> sink and check the recovered
// sequence is 0,1,2,...,count-1.
void run_roundtrip(int payload_size, long count)
{
    auto tb = gr::make_top_block("qa_seq");
    auto src = std::make_shared<byte_source>(payload_size, count);
    auto stamp = gr::cuda::seq_stamp::make(payload_size);
    auto strip = gr::cuda::seq_strip::make(payload_size);
    auto snk = std::make_shared<u64_sink>();

    tb->connect(src, 0, stamp, 0);
    tb->connect(stamp, 0, strip, 0);
    tb->connect(strip, 0, snk, 0);
    tb->run();

    const auto& d = snk->data();
    BOOST_REQUIRE_EQUAL(d.size(), static_cast<size_t>(count));
    long mismatch = 0;
    for (long i = 0; i < count; i++)
        if (d[i] != static_cast<uint64_t>(i))
            mismatch++;
    BOOST_CHECK_EQUAL(mismatch, 0);
}

} // namespace

// count > MAX_BATCH (16384) forces several work() calls, so this also
// checks the counter continues correctly across batches.
BOOST_AUTO_TEST_CASE(seq_roundtrip_multibatch)
{
    run_roundtrip(/*payload_size=*/8, /*count=*/40000);
}

BOOST_AUTO_TEST_CASE(seq_roundtrip_wide_payload)
{
    run_roundtrip(/*payload_size=*/64, /*count=*/5000);
}

// payload_size must be >= 8 and a multiple of 8.
BOOST_AUTO_TEST_CASE(seq_stamp_ctor_validation)
{
    using gr::cuda::seq_stamp;
    BOOST_CHECK_NO_THROW(seq_stamp::make(8));
    BOOST_CHECK_NO_THROW(seq_stamp::make(64));
    BOOST_CHECK_THROW(seq_stamp::make(7), std::runtime_error);  // < 8
    BOOST_CHECK_THROW(seq_stamp::make(12), std::runtime_error); // not multiple of 8
}

BOOST_AUTO_TEST_CASE(seq_strip_ctor_validation)
{
    using gr::cuda::seq_strip;
    BOOST_CHECK_NO_THROW(seq_strip::make(8));
    BOOST_CHECK_NO_THROW(seq_strip::make(64));
    BOOST_CHECK_THROW(seq_strip::make(7), std::runtime_error);
    BOOST_CHECK_THROW(seq_strip::make(12), std::runtime_error);
}
