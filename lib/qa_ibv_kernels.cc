/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * On-device unit tests for the IBV sink/source CUDA kernels, exercised
 * through their plain C++ entry points (exec_build_frames_kernel /
 * exec_strip_headers_kernel).  These need a GPU but no ibverbs/NIC, so
 * they validate the frame assembly/disassembly math independently of the
 * ibv_sink / ibv_source blocks.  Compiled only when ENABLE_IBV is set
 * (that is what builds the .cu files these entry points live in).
 */

#include "ibv_sink.cuh"
#include "ibv_source.cuh"

#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <vector>

namespace {

constexpr int HDR = 42; // L2/L3/L4 header length used by the blocks

// Minimal device-buffer helper.
struct dbuf {
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit dbuf(size_t bytes) : n(bytes)
    {
        BOOST_REQUIRE_EQUAL(cudaMalloc(&p, bytes), cudaSuccess);
    }
    ~dbuf()
    {
        if (p)
            cudaFree(p);
    }
    dbuf(const dbuf&) = delete;
    dbuf& operator=(const dbuf&) = delete;

    void fill(int v) { BOOST_REQUIRE_EQUAL(cudaMemset(p, v, n), cudaSuccess); }
    void put(const void* src, size_t bytes)
    {
        BOOST_REQUIRE_EQUAL(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice),
                            cudaSuccess);
    }
    void get(void* dst, size_t bytes) const
    {
        BOOST_REQUIRE_EQUAL(cudaMemcpy(dst, p, bytes, cudaMemcpyDeviceToHost),
                            cudaSuccess);
    }
};

} // namespace

// Build N frames into a landing buffer and verify each slot is
// header_template ++ payload, with correct modulo slot-wrapping and
// untouched trailing bytes.  Odd payload_size exercises the kernel's
// head/bulk/tail alignment phases.
BOOST_AUTO_TEST_CASE(build_frames_layout_and_wrap)
{
    const int payload_size = 100; // not a multiple of 16 -> head+bulk+tail
    const uint32_t num_slots = 8;
    const int slot_size = HDR + payload_size + 8; // +8 trailing guard bytes
    const int num_frames = 5;
    const uint32_t first_slot = 6; // 6,7,0,1,2 -> wraps past num_slots

    std::vector<uint8_t> header(HDR);
    for (int i = 0; i < HDR; i++)
        header[i] = static_cast<uint8_t>(i + 1);

    std::vector<uint8_t> payload(num_frames * payload_size);
    for (size_t i = 0; i < payload.size(); i++)
        payload[i] = static_cast<uint8_t>((i * 7 + 3) & 0xff);

    dbuf d_landing(num_slots * slot_size);
    dbuf d_header(HDR);
    dbuf d_payload(payload.size());
    d_landing.fill(0xAA);
    d_header.put(header.data(), header.size());
    d_payload.put(payload.data(), payload.size());

    exec_build_frames_kernel(d_landing.p,
                             slot_size,
                             first_slot,
                             num_slots,
                             d_header.p,
                             HDR,
                             d_payload.p,
                             payload_size,
                             num_frames,
                             /*block_size=*/64,
                             /*stream=*/0);
    BOOST_REQUIRE_EQUAL(cudaStreamSynchronize(0), cudaSuccess);

    std::vector<uint8_t> landing(num_slots * slot_size);
    d_landing.get(landing.data(), landing.size());

    int hdr_mismatch = 0, pay_mismatch = 0, guard_mismatch = 0;
    for (int f = 0; f < num_frames; f++) {
        uint32_t slot = (first_slot + f) % num_slots;
        const uint8_t* s = landing.data() + slot * slot_size;
        for (int i = 0; i < HDR; i++)
            if (s[i] != header[i])
                hdr_mismatch++;
        for (int k = 0; k < payload_size; k++)
            if (s[HDR + k] != payload[f * payload_size + k])
                pay_mismatch++;
        for (int t = HDR + payload_size; t < slot_size; t++)
            if (s[t] != 0xAA)
                guard_mismatch++;
    }
    BOOST_CHECK_EQUAL(hdr_mismatch, 0);
    BOOST_CHECK_EQUAL(pay_mismatch, 0);
    BOOST_CHECK_EQUAL(guard_mismatch, 0); // only header+payload region written
}

// num_frames <= 0 must be a no-op (landing buffer untouched).
BOOST_AUTO_TEST_CASE(build_frames_zero_is_noop)
{
    const int payload_size = 64;
    const uint32_t num_slots = 4;
    const int slot_size = HDR + payload_size;

    dbuf d_landing(num_slots * slot_size);
    dbuf d_header(HDR);
    dbuf d_payload(payload_size);
    d_landing.fill(0xAA);
    d_header.fill(0);
    d_payload.fill(0);

    exec_build_frames_kernel(d_landing.p,
                             slot_size,
                             0,
                             num_slots,
                             d_header.p,
                             HDR,
                             d_payload.p,
                             payload_size,
                             /*num_frames=*/0,
                             64,
                             0);
    BOOST_REQUIRE_EQUAL(cudaStreamSynchronize(0), cudaSuccess);

    std::vector<uint8_t> landing(num_slots * slot_size);
    d_landing.get(landing.data(), landing.size());
    int touched = 0;
    for (uint8_t b : landing)
        if (b != 0xAA)
            touched++;
    BOOST_CHECK_EQUAL(touched, 0);
}

// Strip payloads out of a landing buffer; verify extraction + slot wrap.
BOOST_AUTO_TEST_CASE(strip_headers_extracts_payload)
{
    const int payload_size = 100;
    const uint32_t num_slots = 8;
    const int slot_size = HDR + payload_size + 8;
    const int num_packets = 5;
    const uint32_t first_slot = 6;

    // Lay out the landing buffer: each slot = junk header ++ known payload.
    std::vector<uint8_t> landing(num_slots * slot_size, 0xEE);
    std::vector<uint8_t> expect(num_packets * payload_size);
    for (int f = 0; f < num_packets; f++) {
        uint32_t slot = (first_slot + f) % num_slots;
        uint8_t* s = landing.data() + slot * slot_size;
        for (int k = 0; k < payload_size; k++) {
            uint8_t v = static_cast<uint8_t>((f * 31 + k) & 0xff);
            s[HDR + k] = v;
            expect[f * payload_size + k] = v;
        }
    }

    dbuf d_landing(landing.size());
    dbuf d_out(num_packets * payload_size);
    d_landing.put(landing.data(), landing.size());
    d_out.fill(0);

    exec_strip_headers_kernel(d_landing.p,
                              slot_size,
                              first_slot,
                              num_slots,
                              d_out.p,
                              HDR,
                              payload_size,
                              num_packets,
                              64,
                              0);
    BOOST_REQUIRE_EQUAL(cudaStreamSynchronize(0), cudaSuccess);

    std::vector<uint8_t> out(num_packets * payload_size);
    d_out.get(out.data(), out.size());
    BOOST_CHECK(out == expect);
}

// build_frames followed by strip_headers over the same slots is identity
// on the payload.
BOOST_AUTO_TEST_CASE(build_then_strip_roundtrip)
{
    const int payload_size = 1500;
    const uint32_t num_slots = 16;
    const int slot_size = HDR + payload_size;
    const int num_frames = 10;
    const uint32_t first_slot = 12; // wraps

    std::vector<uint8_t> header(HDR, 0x5a);
    std::vector<uint8_t> payload(num_frames * payload_size);
    for (size_t i = 0; i < payload.size(); i++)
        payload[i] = static_cast<uint8_t>((i * 13 + 7) & 0xff);

    dbuf d_landing(num_slots * slot_size);
    dbuf d_header(HDR);
    dbuf d_payload(payload.size());
    dbuf d_out(payload.size());
    d_landing.fill(0);
    d_header.put(header.data(), header.size());
    d_payload.put(payload.data(), payload.size());

    exec_build_frames_kernel(d_landing.p,
                             slot_size,
                             first_slot,
                             num_slots,
                             d_header.p,
                             HDR,
                             d_payload.p,
                             payload_size,
                             num_frames,
                             64,
                             0);
    exec_strip_headers_kernel(d_landing.p,
                              slot_size,
                              first_slot,
                              num_slots,
                              d_out.p,
                              HDR,
                              payload_size,
                              num_frames,
                              64,
                              0);
    BOOST_REQUIRE_EQUAL(cudaStreamSynchronize(0), cudaSuccess);

    std::vector<uint8_t> out(payload.size());
    d_out.get(out.data(), out.size());
    BOOST_CHECK(out == payload);
}
