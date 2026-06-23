#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest


class qa_complex_to_mag_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_basic(self):
        N = 1000
        src_data = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)

        src = blocks.vector_source_c(src_data, False)
        dut = cuda.complex_to_mag_cupy()
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = np.abs(src_data)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_002_vlen(self):
        N = 500
        vlen = 4
        src_data = (np.random.randn(N * vlen) + 1j * np.random.randn(N * vlen)).astype(
            np.complex64
        )

        src = blocks.vector_source_c(src_data, False, vlen)
        dut = cuda.complex_to_mag_cupy(vlen=vlen)
        snk = blocks.vector_sink_f(vlen)

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = np.abs(src_data)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_003_known_values(self):
        src_data = np.array([1 + 0j, 0 + 1j, 3 + 4j, -3 - 4j], dtype=np.complex64)

        src = blocks.vector_source_c(src_data, False)
        dut = cuda.complex_to_mag_cupy()
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


class qa_complex_to_mag_squared_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_basic(self):
        N = 1000
        src_data = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)

        src = blocks.vector_source_c(src_data, False)
        dut = cuda.complex_to_mag_squared_cupy()
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = np.abs(src_data) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_002_vlen(self):
        N = 500
        vlen = 4
        src_data = (np.random.randn(N * vlen) + 1j * np.random.randn(N * vlen)).astype(
            np.complex64
        )

        src = blocks.vector_source_c(src_data, False, vlen)
        dut = cuda.complex_to_mag_squared_cupy(vlen=vlen)
        snk = blocks.vector_sink_f(vlen)

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = np.abs(src_data) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_003_known_values(self):
        src_data = np.array([1 + 0j, 0 + 1j, 3 + 4j, -3 - 4j], dtype=np.complex64)

        src = blocks.vector_source_c(src_data, False)
        dut = cuda.complex_to_mag_squared_cupy()
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = np.array([1.0, 1.0, 25.0, 25.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


class qa_nlog10_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_10log10(self):
        """Classic 10*log10 — matches upstream nlog10_ff QA."""
        src_data = np.array([1, 10, 100, 1000, 10000, 100000], dtype=np.float32)
        expected = np.array([0, 10, 20, 30, 40, 50], dtype=np.float32)

        src = blocks.vector_source_f(src_data, False)
        dut = cuda.nlog10_cupy(n=10)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_002_20log10(self):
        src_data = np.array([1, 10, 100], dtype=np.float32)
        expected = np.array([0, 20, 40], dtype=np.float32)

        src = blocks.vector_source_f(src_data, False)
        dut = cuda.nlog10_cupy(n=20)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_003_with_offset(self):
        src_data = np.array([1, 10, 100], dtype=np.float32)
        expected = np.array([30, 40, 50], dtype=np.float32)

        src = blocks.vector_source_f(src_data, False)
        dut = cuda.nlog10_cupy(n=10, k=30)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_004_vlen(self):
        N = 500
        vlen = 4
        src_data = np.abs(np.random.randn(N * vlen)).astype(np.float32) + 1e-10

        src = blocks.vector_source_f(src_data, False, vlen)
        dut = cuda.nlog10_cupy(n=10, vlen=vlen)
        snk = blocks.vector_sink_f(vlen)

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = 10 * np.log10(src_data)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_005_clamps_zero(self):
        """Zero input should not produce -inf."""
        src_data = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        src = blocks.vector_source_f(src_data, False)
        dut = cuda.nlog10_cupy(n=10)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    gr_unittest.run(qa_complex_to_mag_cupy)
    gr_unittest.run(qa_complex_to_mag_squared_cupy)
    gr_unittest.run(qa_nlog10_cupy)
