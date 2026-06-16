/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * Unit tests for the seq_stamp / seq_strip blocks.
 *
 * seq_stamp prepends an incrementing 64-bit sequence number to each item
 * (output grows by 8 bytes, payload passes through); seq_strip recovers
 * that 8-byte header on output 0 and, optionally, the original payload on
 * output 1.
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
// cuda_buffer output. Content is irrelevant to the counter assertion
// (seq_strip only recovers the sequence header), but we zero it so the
// stream carries real work for seq_stamp to pass through.
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

// Finite GPU source emitting a deterministic per-byte pattern so the payload
// can be verified end-to-end: byte j of global item g is (g + j) & 0xFF.
class pattern_source : public gr::sync_block, public gr::cuda_block
{
    int item_size_;
    long remaining_;
    long produced_ = 0;
    std::vector<uint8_t> host_;

public:
    pattern_source(int item_size, long count)
        : gr::sync_block(
              "pattern_source",
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

        host_.resize(static_cast<size_t>(n) * item_size_);
        for (int i = 0; i < n; i++) {
            uint8_t* item = host_.data() + static_cast<size_t>(i) * item_size_;
            long g = produced_ + i;
            for (int j = 0; j < item_size_; j++)
                item[j] = static_cast<uint8_t>((g + j) & 0xFF);
        }

        check_cuda_errors(cudaMemcpyAsync(output_items[0],
                                          host_.data(),
                                          static_cast<size_t>(n) * item_size_,
                                          cudaMemcpyHostToDevice,
                                          d_stream),
                          "pattern_source: cudaMemcpyAsync");
        // host_ is reused on the next call, so wait for the copy to drain it.
        check_cuda_errors(cudaStreamSynchronize(d_stream),
                          "pattern_source: cudaStreamSynchronize");

        produced_ += n;
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

// Host sink collecting a raw byte stream (the recovered payload).
class byte_sink : public gr::sync_block
{
    int item_size_;
    std::vector<uint8_t> data_;

public:
    byte_sink(int item_size)
        : gr::sync_block("byte_sink",
                         gr::io_signature::make(1, 1, item_size),
                         gr::io_signature::make(0, 0, 0)),
          item_size_(item_size)
    {
    }

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& /*output_items*/) override
    {
        auto in = static_cast<const uint8_t*>(input_items[0]);
        data_.insert(
            data_.end(), in, in + static_cast<size_t>(noutput_items) * item_size_);
        return noutput_items;
    }

    const std::vector<uint8_t>& data() const { return data_; }
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

    // Cap seq_stamp's per-call batch so `count` spans several work() calls,
    // exercising the counter's continuation across calls (seq_stamp no longer
    // has an internal batch cap now that the counter is generated by a kernel).
    stamp->set_max_noutput_items(8192);

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

// Run pattern_source -> seq_stamp -> seq_strip with BOTH outputs connected,
// checking that port 0 recovers the sequence 0..count-1 and port 1 reproduces
// the original payload bytes exactly (byte j of item i == (i + j) & 0xFF).
void run_payload_split(int payload_size, long count)
{
    auto tb = gr::make_top_block("qa_seq_split");
    auto src = std::make_shared<pattern_source>(payload_size, count);
    auto stamp = gr::cuda::seq_stamp::make(payload_size);
    auto strip = gr::cuda::seq_strip::make(payload_size);
    auto seq_snk = std::make_shared<u64_sink>();
    auto pay_snk = std::make_shared<byte_sink>(payload_size);

    // Force several work() calls so both ports are exercised across batches.
    stamp->set_max_noutput_items(8192);

    tb->connect(src, 0, stamp, 0);
    tb->connect(stamp, 0, strip, 0);
    tb->connect(strip, 0, seq_snk, 0); // port 0: sequence numbers
    tb->connect(strip, 1, pay_snk, 0); // port 1: stripped payload
    tb->run();

    const auto& seq = seq_snk->data();
    BOOST_REQUIRE_EQUAL(seq.size(), static_cast<size_t>(count));
    long seq_mismatch = 0;
    for (long i = 0; i < count; i++)
        if (seq[i] != static_cast<uint64_t>(i))
            seq_mismatch++;
    BOOST_CHECK_EQUAL(seq_mismatch, 0);

    const auto& pay = pay_snk->data();
    BOOST_REQUIRE_EQUAL(pay.size(), static_cast<size_t>(count) * payload_size);
    long pay_mismatch = 0;
    for (long i = 0; i < count; i++)
        for (int j = 0; j < payload_size; j++)
            if (pay[static_cast<size_t>(i) * payload_size + j] !=
                static_cast<uint8_t>((i + j) & 0xFF))
                pay_mismatch++;
    BOOST_CHECK_EQUAL(pay_mismatch, 0);
}

} // namespace

// count well over the 8192 per-call cap forces several work() calls, so
// this also checks the counter continues correctly across batches.
BOOST_AUTO_TEST_CASE(seq_roundtrip_multibatch)
{
    run_roundtrip(/*payload_size=*/8, /*count=*/40000);
}

BOOST_AUTO_TEST_CASE(seq_roundtrip_wide_payload)
{
    run_roundtrip(/*payload_size=*/64, /*count=*/5000);
}

// Exercises seq_strip's optional payload port (output 1): the sequence number
// is recovered on port 0 while the original payload is reproduced on port 1.
// count > the 8192 cap also spans several work() calls.
BOOST_AUTO_TEST_CASE(seq_payload_split)
{
    run_payload_split(/*payload_size=*/16, /*count=*/20000);
}

// payload_size must be >= 1 (the 8-byte sequence header is added on top,
// so no multiple-of-8 requirement on the payload itself).
BOOST_AUTO_TEST_CASE(seq_stamp_ctor_validation)
{
    using gr::cuda::seq_stamp;
    BOOST_CHECK_NO_THROW(seq_stamp::make(1));
    BOOST_CHECK_NO_THROW(seq_stamp::make(7));
    BOOST_CHECK_NO_THROW(seq_stamp::make(64));
    BOOST_CHECK_THROW(seq_stamp::make(0), std::exception);
    BOOST_CHECK_THROW(seq_stamp::make(-8), std::exception);
}

BOOST_AUTO_TEST_CASE(seq_strip_ctor_validation)
{
    using gr::cuda::seq_strip;
    BOOST_CHECK_NO_THROW(seq_strip::make(1));
    BOOST_CHECK_NO_THROW(seq_strip::make(7));
    BOOST_CHECK_NO_THROW(seq_strip::make(64));
    BOOST_CHECK_THROW(seq_strip::make(0), std::exception);
    BOOST_CHECK_THROW(seq_strip::make(-8), std::exception);
}
