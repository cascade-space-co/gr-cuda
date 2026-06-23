#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA for the CuPy Vector Source, compared against blocks.vector_source_x."""

import numpy as np
import pmt
from gnuradio import blocks, cuda, gr, gr_unittest


class qa_vector_source_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_complex_no_repeat(self):
        data = (np.random.randn(317) + 1j * np.random.randn(317)).astype(np.complex64)

        ref = blocks.vector_source_c(data, False)
        ref_snk = blocks.vector_sink_c()
        ref_tb = gr.top_block()
        ref_tb.connect(ref, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.complex64)

        dut = cuda.vector_source_cupy(data, False, dtype=np.complex64)
        snk = blocks.vector_sink_c()
        self.tb.connect(dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.complex64)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_float_repeat(self):
        data = np.random.randn(123).astype(np.float32)
        n = 1000

        ref = blocks.vector_source_f(data, True)
        ref_head = blocks.head(gr.sizeof_float, n)
        ref_snk = blocks.vector_sink_f()
        ref_tb = gr.top_block()
        ref_tb.connect(ref, ref_head, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.float32)

        dut = cuda.vector_source_cupy(data, True, dtype=np.float32)
        head = blocks.head(gr.sizeof_float, n)
        snk = blocks.vector_sink_f()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)

        self.assertEqual(len(result), n)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_float_vlen(self):
        vlen = 4
        data = np.random.randn(vlen * 50).astype(np.float32)

        ref = blocks.vector_source_f(data, False, vlen)
        ref_snk = blocks.vector_sink_f(vlen)
        ref_tb = gr.top_block()
        ref_tb.connect(ref, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.float32)

        dut = cuda.vector_source_cupy(data, False, vlen, dtype=np.float32)
        snk = blocks.vector_sink_f(vlen)
        self.tb.connect(dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_tags_repeat(self):
        data = np.arange(8, dtype=np.float32)
        tag = gr.tag_t()
        tag.offset = 2
        tag.key = pmt.string_to_symbol("k")
        tag.value = pmt.to_pmt(42)
        tag.srcid = pmt.to_pmt("src")

        n = 32  # 4 full copies of the 8-sample vector

        ref = blocks.vector_source_f(data, True, 1, [tag])
        ref_head = blocks.head(gr.sizeof_float, n)
        ref_snk = blocks.vector_sink_f()
        ref_tb = gr.top_block()
        ref_tb.connect(ref, ref_head, ref_snk)
        ref_tb.run()
        ref_offsets = sorted(t.offset for t in ref_snk.tags())

        dut = cuda.vector_source_cupy(data, True, 1, [tag], dtype=np.float32)
        head = blocks.head(gr.sizeof_float, n)
        snk = blocks.vector_sink_f()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        dut_offsets = sorted(t.offset for t in snk.tags())

        self.assertEqual(dut_offsets, ref_offsets)


if __name__ == "__main__":
    gr_unittest.run(qa_vector_source_cupy)
