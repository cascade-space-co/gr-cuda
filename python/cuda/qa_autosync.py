#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
QA tests for the automatic CUDA synchronization contract.

These tests cover scenarios NOT already exercised by the existing QA suite
(qa_stress_sync, qa_parallel_sync, qa_cuda_block, qa_multiply_const_cupy, etc).

Covered scenarios:
  - C++ GPU->GPU chains (cuda.copy) with multi-iteration event reuse
  - Mixed C++/Python GPU block boundaries
  - Fan-out with C++ blocks and asymmetric consumers
  - D2H fan-out to multiple CPU sinks
"""

import cupy as cp
import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest


class _gpu_scale(cuda.sync_block):
    """Multiply by a constant on the GPU. No explicit sync calls."""

    def __init__(self, k, dtype=np.float32):
        self.k = dtype(k)
        cuda.sync_block.__init__(self, "gpu_scale", [dtype], [dtype])

    def work(self, input_items, output_items):
        cp.multiply(input_items[0], self.k, out=output_items[0])
        return len(output_items[0])


class qa_autosync(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    # -- C++ GPU -> GPU chain ----------------------------------------------

    def test_cpp_gpu_to_gpu_chain(self):
        """
        Chain of C++ cuda.copy blocks (GPU->GPU edges) with enough data
        to exercise multiple work() calls and event reuse across iterations.
        """
        N = 1_000_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        chain = [cuda.copy(np.dtype(np.float32).itemsize) for _ in range(5)]
        snk = blocks.vector_sink_f()

        self.tb.connect(src, chain[0])
        for i in range(len(chain) - 1):
            self.tb.connect(chain[i], chain[i + 1])
        self.tb.connect(chain[-1], snk)

        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data)

    # -- Mixed C++ / Python boundary ---------------------------------------

    def test_mixed_cpp_cupy_chain(self):
        """
        C++ cuda.copy -> Python CuPy block -> C++ cuda.copy.
        Auto-sync must work across C++/Python block boundaries.
        """
        N = 50_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        c1 = cuda.copy(np.dtype(np.float32).itemsize)
        scale = _gpu_scale(3.0)
        c2 = cuda.copy(np.dtype(np.float32).itemsize)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, c1, scale, c2, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, src_data * 3.0, rtol=1e-6)

    # -- Fan-out -----------------------------------------------------------

    def test_fanout_cpp_gpu_blocks(self):
        """
        Fan-out with C++ GPU blocks only.
        Verifies read-done event chaining across multiple cuda_buffer_readers.
        """
        N = 100_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        copy_in = cuda.copy(np.dtype(np.float32).itemsize)

        branch_a = cuda.copy(np.dtype(np.float32).itemsize)
        branch_b = cuda.copy(np.dtype(np.float32).itemsize)

        snk_a = blocks.vector_sink_f()
        snk_b = blocks.vector_sink_f()

        self.tb.connect(src, copy_in)
        self.tb.connect(copy_in, branch_a, snk_a)
        self.tb.connect(copy_in, branch_b, snk_b)

        self.tb.run()

        result_a = np.array(snk_a.data(), dtype=np.float32)
        result_b = np.array(snk_b.data(), dtype=np.float32)
        np.testing.assert_array_equal(result_a, src_data)
        np.testing.assert_array_equal(result_b, src_data)

    def test_fanout_asymmetric_consumers(self):
        """
        Fan-out where one consumer is fast (copy) and the other is
        computationally heavier (scale chain). read-done must wait
        for the slowest consumer.
        """
        N = 50_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        producer = cuda.copy(np.dtype(np.float32).itemsize)

        fast = cuda.copy(np.dtype(np.float32).itemsize)
        snk_fast = blocks.vector_sink_f()

        s1 = _gpu_scale(2.0)
        s2 = _gpu_scale(3.0)
        s3 = _gpu_scale(1.0 / 6.0)
        snk_slow = blocks.vector_sink_f()

        self.tb.connect(src, producer)
        self.tb.connect(producer, fast, snk_fast)
        self.tb.connect(producer, s1, s2, s3, snk_slow)

        self.tb.run()

        result_fast = np.array(snk_fast.data(), dtype=np.float32)
        result_slow = np.array(snk_slow.data(), dtype=np.float32)
        np.testing.assert_array_equal(result_fast, src_data)
        np.testing.assert_allclose(result_slow, src_data, rtol=1e-4)

    # -- D2H fan-out to multiple CPU sinks ---------------------------------

    def test_d2h_multiple_sinks(self):
        """
        Fan-out at the D2H boundary: GPU producer -> two CPU sinks.
        Each sink gets its own cuda_buffer_reader with independent
        host-ready events.
        """
        N = 50_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        scale = _gpu_scale(2.0)

        snk_a = blocks.vector_sink_f()
        snk_b = blocks.vector_sink_f()

        self.tb.connect(src, scale)
        self.tb.connect(scale, snk_a)
        self.tb.connect(scale, snk_b)

        self.tb.run()

        result_a = np.array(snk_a.data(), dtype=np.float32)
        result_b = np.array(snk_b.data(), dtype=np.float32)
        np.testing.assert_allclose(result_a, src_data * 2.0, rtol=1e-6)
        np.testing.assert_allclose(result_b, src_data * 2.0, rtol=1e-6)


if __name__ == "__main__":
    gr_unittest.run(qa_autosync)
