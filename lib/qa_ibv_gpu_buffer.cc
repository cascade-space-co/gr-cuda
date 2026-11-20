/* -*- c++ -*- */
/*
 * Copyright 2026 Cascade Space.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/*
 * On-device unit tests for ibv_gpu_buffer.  They need a GPU and an ibverbs
 * device for the protection domain, but no link and no CAP_NET_RAW.  The test
 * passes without checking anything on a machine with no ibverbs device.
 */

#include "network/ibv_common.h"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <memory>

namespace {

// The first ibverbs device's context and a protection domain on it.
struct first_device {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;

    first_device()
    {
        int n = 0;
        ibv_device** devs = ibv_get_device_list(&n);
        if (devs && n > 0)
            ctx = ibv_open_device(devs[0]);
        if (devs)
            ibv_free_device_list(devs);
        if (ctx)
            pd = ibv_alloc_pd(ctx);
    }
    ~first_device()
    {
        if (pd)
            ibv_dealloc_pd(pd);
        if (ctx)
            ibv_close_device(ctx);
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(registers_sizes_that_are_not_page_aligned)
{
    first_device dev;
    if (!dev.pd) {
        BOOST_TEST_MESSAGE("no ibverbs device; skipping");
        return;
    }

    constexpr size_t PAGE = gr::cuda::ibv_gpu_buffer::GPU_PAGE_SIZE;
    struct {
        size_t requested;
        size_t expected;
    } cases[] = {
        { 1, PAGE },
        { 1000003, PAGE },
        { PAGE, PAGE },
        { PAGE + 1, 2 * PAGE },
        { 7 * 8042 * 1235, 34 * PAGE }, // 69523090 bytes
    };
    for (const auto& c : cases) {
        BOOST_TEST_CONTEXT("requested " << c.requested << " bytes")
        {
            std::unique_ptr<gr::cuda::ibv_gpu_buffer> buf;
            BOOST_REQUIRE_NO_THROW(
                buf = std::make_unique<gr::cuda::ibv_gpu_buffer>(c.requested, dev.pd));
            BOOST_CHECK_EQUAL(buf->size(), c.expected);
            BOOST_REQUIRE(buf->mr() != nullptr);
            BOOST_CHECK_EQUAL(buf->mr()->length, c.expected);
            // The whole rounded size is usable device memory.
            if (buf->is_dmabuf())
                BOOST_CHECK_EQUAL(cudaMemset(buf->data(), 0x5a, buf->size()),
                                  cudaSuccess);
        }
    }
}
