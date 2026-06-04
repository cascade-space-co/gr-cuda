/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * C++ QA for the auto-sync contract on `general_work()` blocks.
 *
 * The Python QA suite (qa_general_work_sync, qa_autosync) exercises
 * the same contract via the CuPy wrapper, but every Python call
 * crosses pybind11 and serialises on the GIL. This test file builds
 * native C++ general_work() blocks and runs them through the GR
 * scheduler directly, so the producer/consumer thread interleaving
 * is more aggressive (no GIL) and the buffer hooks (`post_work`,
 * `update_read_pointer`, `write_pointer`) are reached from genuine
 * native `produce()` / `consume_each()` calls.
 *
 * Test blocks are kept inline (no separate impl/factory/binding) so
 * the QA stays a single self-contained translation unit. All "work"
 * is done via `cudaMemcpyAsync`, which avoids needing a `.cu` file
 * for tiny kernels — what we are exercising is the auto-sync, not
 * the math.
 */

#include <gnuradio/block.h>
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cuda_buffer.h>
#include <gnuradio/cuda/cuda_error.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/top_block.h>

#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace {

constexpr int F = static_cast<int>(sizeof(float));

// ---------------------------------------------------------------------------
// Inline CPU source / sink (no gr-blocks dependency).
// ---------------------------------------------------------------------------

class fvec_source : public gr::sync_block
{
    std::vector<float> data_;
    size_t pos_ = 0;

public:
    explicit fvec_source(std::vector<float> data)
        : gr::sync_block("fvec_source",
                         gr::io_signature::make(0, 0, 0),
                         gr::io_signature::make(1, 1, F)),
          data_(std::move(data))
    {
    }

    int work(int noutput_items,
             gr_vector_const_void_star& /*input_items*/,
             gr_vector_void_star& output_items) override
    {
        size_t remaining = data_.size() - pos_;
        if (remaining == 0)
            return WORK_DONE;
        int n = std::min(static_cast<int>(remaining), noutput_items);
        std::memcpy(output_items[0], data_.data() + pos_, static_cast<size_t>(n) * F);
        pos_ += n;
        return n;
    }
};

class fvec_sink : public gr::sync_block
{
    std::vector<float> data_;

public:
    fvec_sink()
        : gr::sync_block("fvec_sink",
                         gr::io_signature::make(1, 1, F),
                         gr::io_signature::make(0, 0, 0))
    {
    }

    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& /*output_items*/) override
    {
        auto in = static_cast<const float*>(input_items[0]);
        data_.insert(data_.end(), in, in + noutput_items);
        return noutput_items;
    }

    const std::vector<float>& data() const { return data_; }
};

// ---------------------------------------------------------------------------
// GPU general_work blocks.
//
// Each block follows the same pattern:
//   1. Enqueue all GPU work on d_stream (cudaMemcpyAsync).
//   2. Call consume_each() / consume() and produce() — AFTER the
//      memcpy is enqueued, so the buffer hooks see a stream position
//      that already includes the work.
//   3. Return WORK_CALLED_PRODUCE.
// ---------------------------------------------------------------------------

/// 1×1 identity (D2D copy).
class gw_copy : public gr::block, public gr::cuda_block
{
public:
    gw_copy()
        : gr::block("gw_copy",
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type),
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type))
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = noutput_items;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n = std::min(noutput_items, ninput_items[0]);
        if (n <= 0)
            return 0;

        auto in = static_cast<const float*>(input_items[0]);
        auto out = static_cast<float*>(output_items[0]);
        check_cuda_errors(cudaMemcpyAsync(out,
                                          in,
                                          static_cast<size_t>(n) * F,
                                          cudaMemcpyDeviceToDevice,
                                          d_stream),
                          "gw_copy: D2D");

        consume_each(n);
        produce(0, n);
        return WORK_CALLED_PRODUCE;
    }
};

/// 1×1 interpolate-by-N: zero-stuffs N-1 zeros between input samples.
class gw_interp : public gr::block, public gr::cuda_block
{
    const int interp_;

public:
    explicit gw_interp(int interp = 2)
        : gr::block("gw_interp",
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type),
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type)),
          interp_(interp)
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = (noutput_items + interp_ - 1) / interp_;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n_in = std::min(ninput_items[0], noutput_items / interp_);
        if (n_in <= 0)
            return 0;
        int n_out = n_in * interp_;

        auto in = static_cast<const float*>(input_items[0]);
        auto out = static_cast<float*>(output_items[0]);
        // Zero the output region first, then drop input samples at every
        // interp_-th offset via a strided 2D copy.
        check_cuda_errors(
            cudaMemsetAsync(out, 0, static_cast<size_t>(n_out) * F, d_stream),
            "gw_interp: memset");
        check_cuda_errors(cudaMemcpy2DAsync(out,
                                            static_cast<size_t>(interp_) * F,
                                            in,
                                            F,
                                            F,
                                            static_cast<size_t>(n_in),
                                            cudaMemcpyDeviceToDevice,
                                            d_stream),
                          "gw_interp: D2D 2D");

        consume_each(n_in);
        produce(0, n_out);
        return WORK_CALLED_PRODUCE;
    }
};

/// 1×1 decimate-by-N via strided cudaMemcpy2DAsync.
class gw_decim : public gr::block, public gr::cuda_block
{
    const int decim_;

public:
    explicit gw_decim(int decim = 2)
        : gr::block("gw_decim",
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type),
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type)),
          decim_(decim)
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = noutput_items * decim_;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n = std::min(ninput_items[0] / decim_, noutput_items);
        if (n <= 0)
            return 0;

        auto in = static_cast<const float*>(input_items[0]);
        auto out = static_cast<float*>(output_items[0]);
        check_cuda_errors(cudaMemcpy2DAsync(out,
                                            F,
                                            in,
                                            static_cast<size_t>(decim_) * F,
                                            F,
                                            static_cast<size_t>(n),
                                            cudaMemcpyDeviceToDevice,
                                            d_stream),
                          "gw_decim: D2D 2D");

        consume_each(n * decim_);
        produce(0, n);
        return WORK_CALLED_PRODUCE;
    }
};

/// 2×1 combiner. Output is a D2D copy of input 0; input 1 is
/// consumed but not used — sufficient to exercise multi-input
/// auto-sync wiring.
class gw_combine_2to1 : public gr::block, public gr::cuda_block
{
public:
    gw_combine_2to1()
        : gr::block("gw_combine_2to1",
                    gr::io_signature::make(2, 2, F, gr::cuda_buffer::type),
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type))
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = noutput_items;
        ninput_items_required[1] = noutput_items;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n = std::min({ noutput_items, ninput_items[0], ninput_items[1] });
        if (n <= 0)
            return 0;

        auto in0 = static_cast<const float*>(input_items[0]);
        auto out = static_cast<float*>(output_items[0]);
        check_cuda_errors(cudaMemcpyAsync(out,
                                          in0,
                                          static_cast<size_t>(n) * F,
                                          cudaMemcpyDeviceToDevice,
                                          d_stream),
                          "gw_combine_2to1: D2D");

        consume(0, n);
        consume(1, n);
        produce(0, n);
        return WORK_CALLED_PRODUCE;
    }
};

/// 2×1 combiner with **asymmetric** input rates: consumes 2×N on
/// port 0 and N on port 1, produces N. The output is a copy of
/// input 1 (input 0 is consumed but unused). Exercises per-input
/// `mark_read_done` event ordering with different rates per port.
class gw_asym_combine_2to1 : public gr::block, public gr::cuda_block
{
public:
    gw_asym_combine_2to1()
        : gr::block("gw_asym_combine_2to1",
                    gr::io_signature::make(2, 2, F, gr::cuda_buffer::type),
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type))
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = noutput_items * 2;
        ninput_items_required[1] = noutput_items;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n = std::min({ noutput_items, ninput_items[0] / 2, ninput_items[1] });
        if (n <= 0)
            return 0;

        auto in1 = static_cast<const float*>(input_items[1]);
        auto out = static_cast<float*>(output_items[0]);
        check_cuda_errors(cudaMemcpyAsync(out,
                                          in1,
                                          static_cast<size_t>(n) * F,
                                          cudaMemcpyDeviceToDevice,
                                          d_stream),
                          "gw_asym_combine_2to1: D2D");

        consume(0, n * 2);
        consume(1, n);
        produce(0, n);
        return WORK_CALLED_PRODUCE;
    }
};

/// 1×2 splitter. Both outputs receive a D2D copy of the input.
class gw_split_1to2 : public gr::block, public gr::cuda_block
{
public:
    gw_split_1to2()
        : gr::block("gw_split_1to2",
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type),
                    gr::io_signature::make(2, 2, F, gr::cuda_buffer::type))
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = noutput_items;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n = std::min(noutput_items, ninput_items[0]);
        if (n <= 0)
            return 0;

        auto in = static_cast<const float*>(input_items[0]);
        auto out0 = static_cast<float*>(output_items[0]);
        auto out1 = static_cast<float*>(output_items[1]);
        check_cuda_errors(cudaMemcpyAsync(out0,
                                          in,
                                          static_cast<size_t>(n) * F,
                                          cudaMemcpyDeviceToDevice,
                                          d_stream),
                          "gw_split_1to2: D2D out0");
        check_cuda_errors(cudaMemcpyAsync(out1,
                                          in,
                                          static_cast<size_t>(n) * F,
                                          cudaMemcpyDeviceToDevice,
                                          d_stream),
                          "gw_split_1to2: D2D out1");

        consume_each(n);
        produce(0, n);
        produce(1, n);
        return WORK_CALLED_PRODUCE;
    }
};

/// 1×2 splitter with **asymmetric** output rates: produces N on
/// port 0 (full-rate copy) and N/2 on port 1 (decim-by-2 of input).
/// Exercises per-output `mark_device_ready` event ordering with
/// different rates per port.
class gw_asym_split_1to2 : public gr::block, public gr::cuda_block
{
public:
    gw_asym_split_1to2()
        : gr::block("gw_asym_split_1to2",
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type),
                    gr::io_signature::make(2, 2, F, gr::cuda_buffer::type))
    {
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        // Worst case: noutput_items applies to port 0 (full rate),
        // so we need at least that many inputs.
        ninput_items_required[0] = noutput_items;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        // n is the count for port 0; port 1 gets n/2. Round to even
        // so the decim-by-2 leg lines up cleanly.
        int n = std::min(ninput_items[0], noutput_items);
        n &= ~1;
        if (n <= 0)
            return 0;

        auto in = static_cast<const float*>(input_items[0]);
        auto out0 = static_cast<float*>(output_items[0]);
        auto out1 = static_cast<float*>(output_items[1]);

        check_cuda_errors(cudaMemcpyAsync(out0,
                                          in,
                                          static_cast<size_t>(n) * F,
                                          cudaMemcpyDeviceToDevice,
                                          d_stream),
                          "gw_asym_split_1to2: D2D out0");
        check_cuda_errors(cudaMemcpy2DAsync(out1,
                                            F,
                                            in,
                                            2 * F,
                                            F,
                                            static_cast<size_t>(n / 2),
                                            cudaMemcpyDeviceToDevice,
                                            d_stream),
                          "gw_asym_split_1to2: D2D 2D out1");

        consume_each(n);
        produce(0, n);
        produce(1, n / 2);
        return WORK_CALLED_PRODUCE;
    }
};

/// 1×1 identity that bounces the data through a scratch buffer
/// `passes` times before writing it out. Each bounce is queued on
/// `d_stream` ahead of the final out-copy, widening the producer's
/// stream queue so that races between produce() and the actual GPU
/// work are easier to surface.
class gw_copy_stress : public gr::block, public gr::cuda_block
{
    const int passes_;
    float* d_scratch_ = nullptr;
    size_t scratch_bytes_ = 0;

public:
    explicit gw_copy_stress(int passes = 50)
        : gr::block("gw_copy_stress",
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type),
                    gr::io_signature::make(1, 1, F, gr::cuda_buffer::type)),
          passes_(passes)
    {
    }

    ~gw_copy_stress() override
    {
        if (d_scratch_)
            cudaFree(d_scratch_);
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override
    {
        ninput_items_required[0] = noutput_items;
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override
    {
        int n = std::min(noutput_items, ninput_items[0]);
        if (n <= 0)
            return 0;

        size_t bytes = static_cast<size_t>(n) * F;
        if (bytes > scratch_bytes_) {
            if (d_scratch_)
                cudaFree(d_scratch_);
            check_cuda_errors(cudaMalloc(reinterpret_cast<void**>(&d_scratch_), bytes),
                              "gw_copy_stress: cudaMalloc");
            scratch_bytes_ = bytes;
        }

        auto in = static_cast<const float*>(input_items[0]);
        auto out = static_cast<float*>(output_items[0]);

        check_cuda_errors(
            cudaMemcpyAsync(d_scratch_, in, bytes, cudaMemcpyDeviceToDevice, d_stream),
            "gw_copy_stress: in→scratch");
        for (int i = 0; i < passes_; ++i) {
            check_cuda_errors(
                cudaMemcpyAsync(
                    d_scratch_, d_scratch_, bytes, cudaMemcpyDeviceToDevice, d_stream),
                "gw_copy_stress: bounce");
        }
        check_cuda_errors(
            cudaMemcpyAsync(out, d_scratch_, bytes, cudaMemcpyDeviceToDevice, d_stream),
            "gw_copy_stress: scratch→out");

        consume_each(n);
        produce(0, n);
        return WORK_CALLED_PRODUCE;
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<float> ramp(size_t n)
{
    std::vector<float> v(n);
    std::iota(v.begin(), v.end(), 0.0f);
    return v;
}

void require_equal(const std::vector<float>& actual, const std::vector<float>& expected)
{
    BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        BOOST_REQUIRE_EQUAL(actual[i], expected[i]);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(single_gw_copy)
{
    auto data = ramp(50000);
    auto tb = gr::make_top_block("single_gw_copy");
    auto src = std::make_shared<fvec_source>(data);
    auto b = std::make_shared<gw_copy>();
    auto snk = std::make_shared<fvec_sink>();
    tb->connect(src, 0, b, 0);
    tb->connect(b, 0, snk, 0);
    tb->run();
    require_equal(snk->data(), data);
}

BOOST_AUTO_TEST_CASE(serial_gw_chain)
{
    auto data = ramp(50000);
    auto tb = gr::make_top_block("serial_gw_chain");
    auto src = std::make_shared<fvec_source>(data);
    auto b1 = std::make_shared<gw_copy>();
    auto b2 = std::make_shared<gw_copy>();
    auto b3 = std::make_shared<gw_copy>();
    auto snk = std::make_shared<fvec_sink>();
    tb->connect(src, 0, b1, 0);
    tb->connect(b1, 0, b2, 0);
    tb->connect(b2, 0, b3, 0);
    tb->connect(b3, 0, snk, 0);
    tb->run();
    require_equal(snk->data(), data);
}

BOOST_AUTO_TEST_CASE(decim_chain)
{
    const size_t N = 49152; // divisible by 4 so two decim-by-2 stages line up cleanly
    auto data = ramp(N);
    auto tb = gr::make_top_block("decim_chain");
    auto src = std::make_shared<fvec_source>(data);
    auto d1 = std::make_shared<gw_decim>(2);
    auto d2 = std::make_shared<gw_decim>(2);
    auto snk = std::make_shared<fvec_sink>();
    tb->connect(src, 0, d1, 0);
    tb->connect(d1, 0, d2, 0);
    tb->connect(d2, 0, snk, 0);
    tb->run();

    std::vector<float> expected;
    expected.reserve(N / 4);
    for (size_t i = 0; i < N; i += 4)
        expected.push_back(data[i]);
    require_equal(snk->data(), expected);
}

BOOST_AUTO_TEST_CASE(interp_chain)
{
    // Two interp-by-2 stages: N → 2N → 4N. After zero-stuffing twice,
    // every 4th output sample equals the original input; the rest are 0.
    const size_t N = 12288;
    auto data = ramp(N);
    auto tb = gr::make_top_block("interp_chain");
    auto src = std::make_shared<fvec_source>(data);
    auto i1 = std::make_shared<gw_interp>(2);
    auto i2 = std::make_shared<gw_interp>(2);
    auto snk = std::make_shared<fvec_sink>();
    tb->connect(src, 0, i1, 0);
    tb->connect(i1, 0, i2, 0);
    tb->connect(i2, 0, snk, 0);
    tb->run();

    std::vector<float> expected(N * 4, 0.0f);
    for (size_t i = 0; i < N; ++i)
        expected[i * 4] = data[i];
    require_equal(snk->data(), expected);
}

BOOST_AUTO_TEST_CASE(asym_combiner)
{
    // Combine block consumes 2N on port 0 and N on port 1, produces N
    // (a copy of input 1). Input 0 carries [0..2N) and input 1 carries
    // [0..N); we expect the sink to receive [0..N).
    const size_t N = 50000;
    auto data1 = ramp(2 * N);
    auto data2 = ramp(N);
    auto tb = gr::make_top_block("asym_combiner");
    auto src1 = std::make_shared<fvec_source>(data1);
    auto src2 = std::make_shared<fvec_source>(data2);
    auto a = std::make_shared<gw_copy>();
    auto b = std::make_shared<gw_copy>();
    auto c = std::make_shared<gw_asym_combine_2to1>();
    auto snk = std::make_shared<fvec_sink>();
    tb->connect(src1, 0, a, 0);
    tb->connect(src2, 0, b, 0);
    tb->connect(a, 0, c, 0);
    tb->connect(b, 0, c, 1);
    tb->connect(c, 0, snk, 0);
    tb->run();
    require_equal(snk->data(), data2);
}

BOOST_AUTO_TEST_CASE(asym_splitter)
{
    // Splitter produces full-rate on port 0 and decim-by-2 on port 1.
    // Sink 0 should match the input verbatim; sink 1 should match the
    // even-indexed input samples.
    const size_t N = 49152;
    auto data = ramp(N);
    auto tb = gr::make_top_block("asym_splitter");
    auto src = std::make_shared<fvec_source>(data);
    auto a = std::make_shared<gw_copy>();
    auto s = std::make_shared<gw_asym_split_1to2>();
    auto c0 = std::make_shared<gw_copy>();
    auto c1 = std::make_shared<gw_copy>();
    auto snk0 = std::make_shared<fvec_sink>();
    auto snk1 = std::make_shared<fvec_sink>();
    tb->connect(src, 0, a, 0);
    tb->connect(a, 0, s, 0);
    tb->connect(s, 0, c0, 0);
    tb->connect(s, 1, c1, 0);
    tb->connect(c0, 0, snk0, 0);
    tb->connect(c1, 0, snk1, 0);
    tb->run();

    require_equal(snk0->data(), data);
    std::vector<float> decim;
    decim.reserve(N / 2);
    for (size_t i = 0; i < N; i += 2)
        decim.push_back(data[i]);
    require_equal(snk1->data(), decim);
}

BOOST_AUTO_TEST_CASE(stress_diamond)
{
    // Diamond topology (split → two branches → combine) with stress blocks
    // that queue many bounce-memcpys on d_stream ahead of produce(). If
    // auto-sync were mis-ordered, consumers would race past the bouncing
    // memcpys and read stale data.
    auto data = ramp(50000);
    auto tb = gr::make_top_block("stress_diamond");
    auto src = std::make_shared<fvec_source>(data);
    auto entry = std::make_shared<gw_copy_stress>(50);
    auto split = std::make_shared<gw_split_1to2>();
    auto br_a = std::make_shared<gw_copy_stress>(50);
    auto br_b = std::make_shared<gw_copy_stress>(50);
    auto combine = std::make_shared<gw_combine_2to1>();
    auto snk = std::make_shared<fvec_sink>();
    tb->connect(src, 0, entry, 0);
    tb->connect(entry, 0, split, 0);
    tb->connect(split, 0, br_a, 0);
    tb->connect(split, 1, br_b, 0);
    tb->connect(br_a, 0, combine, 0);
    tb->connect(br_b, 0, combine, 1);
    tb->connect(combine, 0, snk, 0);
    tb->run();
    require_equal(snk->data(), data);
}
