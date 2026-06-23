#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA for the CuPy Constant Source, compared against analog.sig_source_x."""

import numpy as np
from gnuradio import analog, blocks, cuda, gr, gr_unittest

N = 5000


class qa_const_source_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_complex(self):
        const = 3.0 - 1.5j

        ref_tb = gr.top_block()
        ref_src = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, const)
        ref_head = blocks.head(gr.sizeof_gr_complex, N)
        ref_snk = blocks.vector_sink_c()
        ref_tb.connect(ref_src, ref_head, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.complex64)

        dut = cuda.const_source_cupy(const, dtype=np.complex64)
        head = blocks.head(gr.sizeof_gr_complex, N)
        snk = blocks.vector_sink_c()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.complex64)

        self.assertEqual(len(result), N)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_float(self):
        const = 2.5

        ref_tb = gr.top_block()
        ref_src = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, const)
        ref_head = blocks.head(gr.sizeof_float, N)
        ref_snk = blocks.vector_sink_f()
        ref_tb.connect(ref_src, ref_head, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.float32)

        dut = cuda.const_source_cupy(const, dtype=np.float32)
        head = blocks.head(gr.sizeof_float, N)
        snk = blocks.vector_sink_f()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)

        self.assertEqual(len(result), N)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_int(self):
        const = 7
        dut = cuda.const_source_cupy(const, dtype=np.int32)
        head = blocks.head(gr.sizeof_int, N)
        snk = blocks.vector_sink_i()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.int32)
        self.assertEqual(len(result), N)
        np.testing.assert_array_equal(result, np.full(N, const, dtype=np.int32))


if __name__ == "__main__":
    gr_unittest.run(qa_const_source_cupy)
