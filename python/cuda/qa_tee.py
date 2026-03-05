#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest


class qa_tee(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_cpu(self):
        N = 10000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        dut = cuda.tee(gr.sizeof_float, gpu=False)
        snk0 = blocks.vector_sink_f()
        snk1 = blocks.vector_sink_f()

        self.tb.connect(src, dut)
        self.tb.connect((dut, 0), snk0)
        self.tb.connect((dut, 1), snk1)
        self.tb.run()

        result0 = np.array(snk0.data(), dtype=np.float32)
        result1 = np.array(snk1.data(), dtype=np.float32)
        np.testing.assert_array_equal(result0, src_data)
        np.testing.assert_array_equal(result1, src_data)

    def test_002_gpu(self):
        N = 10000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        dut = cuda.tee(gr.sizeof_float, gpu=True)
        snk0 = blocks.vector_sink_f()
        snk1 = blocks.vector_sink_f()

        self.tb.connect(src, dut)
        self.tb.connect((dut, 0), snk0)
        self.tb.connect((dut, 1), snk1)
        self.tb.run()

        result0 = np.array(snk0.data(), dtype=np.float32)
        result1 = np.array(snk1.data(), dtype=np.float32)
        np.testing.assert_array_equal(result0, src_data)
        np.testing.assert_array_equal(result1, src_data)

    def test_003_gpu_complex(self):
        N = 10000
        src_data = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)

        src = blocks.vector_source_c(src_data, False)
        dut = cuda.tee(gr.sizeof_gr_complex, gpu=True)
        snk0 = blocks.vector_sink_c()
        snk1 = blocks.vector_sink_c()

        self.tb.connect(src, dut)
        self.tb.connect((dut, 0), snk0)
        self.tb.connect((dut, 1), snk1)
        self.tb.run()

        result0 = np.array(snk0.data(), dtype=np.complex64)
        result1 = np.array(snk1.data(), dtype=np.complex64)
        np.testing.assert_array_equal(result0, src_data)
        np.testing.assert_array_equal(result1, src_data)

    def test_004_gpu_vlen(self):
        N = 1000
        vlen = 16
        src_data = np.random.randn(N * vlen).astype(np.float32)

        src = blocks.vector_source_f(src_data, False, vlen)
        dut = cuda.tee(gr.sizeof_float * vlen, gpu=True)
        snk0 = blocks.vector_sink_f(vlen)
        snk1 = blocks.vector_sink_f(vlen)

        self.tb.connect(src, dut)
        self.tb.connect((dut, 0), snk0)
        self.tb.connect((dut, 1), snk1)
        self.tb.run()

        result0 = np.array(snk0.data(), dtype=np.float32)
        result1 = np.array(snk1.data(), dtype=np.float32)
        np.testing.assert_array_equal(result0, src_data)
        np.testing.assert_array_equal(result1, src_data)


if __name__ == "__main__":
    gr_unittest.run(qa_tee)
