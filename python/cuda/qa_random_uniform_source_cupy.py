#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA for the CuPy Random Uniform Source.

The RNG differs from the in-tree block, so outputs are compared statistically
(range, mean, variance) against analog.random_uniform_source_x.
"""

import numpy as np
from gnuradio import analog, blocks, cuda, gr, gr_unittest

N = 200000


def _collect_int(tb, src, n):
    head = blocks.head(gr.sizeof_int, n)
    snk = blocks.vector_sink_i()
    tb.connect(src, head, snk)
    tb.run()
    return np.array(snk.data(), dtype=np.int32)


def _collect(tb, src, itemsize, snk, n):
    head = blocks.head(itemsize, n)
    tb.connect(src, head, snk)
    tb.run()
    return np.array(snk.data())


class qa_random_uniform_source_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_range(self):
        lo, hi = -5, 11
        dut = cuda.random_uniform_source_cupy(lo, hi, seed=123, dtype=np.int32)
        result = _collect_int(self.tb, dut, N)
        self.assertEqual(len(result), N)
        self.assertGreaterEqual(result.min(), lo)
        self.assertLess(result.max(), hi)
        # Every value in [lo, hi) should appear for large N.
        self.assertEqual(set(np.unique(result)), set(range(lo, hi)))

    def test_stats_vs_cpu(self):
        lo, hi = 0, 100

        ref_tb = gr.top_block()
        ref = analog.random_uniform_source_i(lo, hi, 42)
        expected = _collect_int(ref_tb, ref, N)

        dut = cuda.random_uniform_source_cupy(lo, hi, seed=7, dtype=np.int32)
        result = _collect_int(self.tb, dut, N)

        # Theoretical mean/var of a discrete uniform on [lo, hi).
        self.assertAlmostEqual(result.mean(), expected.mean(), delta=1.0)
        self.assertAlmostEqual(
            result.std(), expected.std(), delta=0.05 * expected.std()
        )

    # -- Range tests over each integer type, adapted from in-tree QA ---------

    def test_byte_range(self):
        lo, hi = 0, 5
        dut = cuda.random_uniform_source_cupy(lo, hi, seed=3, dtype=np.uint8)
        res = _collect(self.tb, dut, gr.sizeof_char, blocks.vector_sink_b(), 10000)
        self.assertGreaterEqual(res.min(), lo)
        self.assertLess(res.max(), hi)

    def test_short_range(self):
        lo, hi = 42, 1025
        dut = cuda.random_uniform_source_cupy(lo, hi, seed=3, dtype=np.int16)
        res = _collect(self.tb, dut, gr.sizeof_short, blocks.vector_sink_s(), 10000)
        self.assertGreaterEqual(res.min(), lo)
        self.assertLess(res.max(), hi)

    def test_int_range(self):
        lo, hi = 2**12 - 2, 2**17 + 5
        dut = cuda.random_uniform_source_cupy(lo, hi, seed=3, dtype=np.int32)
        res = _collect_int(self.tb, dut, 10000)
        self.assertGreaterEqual(res.min(), lo)
        self.assertLess(res.max(), hi)

    def test_seed_reproducible(self):
        lo, hi = 0, 1000
        a = _collect_int(
            gr.top_block(),
            cuda.random_uniform_source_cupy(lo, hi, seed=99, dtype=np.int32),
            1000,
        )
        b = _collect_int(
            gr.top_block(),
            cuda.random_uniform_source_cupy(lo, hi, seed=99, dtype=np.int32),
            1000,
        )
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    gr_unittest.run(qa_random_uniform_source_cupy)
