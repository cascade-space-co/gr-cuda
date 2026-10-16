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


def make_tag(offset):
    tag = gr.tag_t()
    tag.offset = offset
    tag.key = pmt.string_to_symbol("k")
    tag.value = pmt.to_pmt(offset)
    tag.srcid = pmt.to_pmt("src")
    return tag


def run_tag_offsets(source, vlen, num_items, max_noutput_items):
    """Run the first num_items items of a float source; return sorted tag offsets."""
    tb = gr.top_block()
    head = blocks.head(gr.sizeof_float * vlen, num_items)
    sink = blocks.vector_sink_f(vlen)
    tb.connect(source, head, sink)
    if max_noutput_items is None:
        tb.run()
    else:
        tb.run(max_noutput_items=max_noutput_items)
    return sorted(t.offset for t in sink.tags())


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
        tag = make_tag(1)
        n = 32  # items to run; 4 or 8 copies of the data vector

        # max_noutput_items that is not a multiple of the copy length makes
        # calls start mid-copy, which tag placement must handle.
        for vlen in (1, 2):
            for max_noutput_items in (None, 5, 3):
                with self.subTest(vlen=vlen, max_noutput_items=max_noutput_items):
                    ref = blocks.vector_source_f(data, True, vlen, [tag])
                    dut = cuda.vector_source_cupy(
                        data, True, vlen, [tag], dtype=np.float32
                    )
                    self.assertEqual(
                        run_tag_offsets(dut, vlen, n, max_noutput_items),
                        run_tag_offsets(ref, vlen, n, None),
                    )

    def test_set_data_tags_follow_new_length(self):
        dut = cuda.vector_source_cupy(
            np.arange(8), True, 1, [make_tag(2)], dtype=np.float32
        )
        dut.set_data(np.arange(6), [make_tag(1)])
        # Tag placement does not depend on the output multiple, so set_data()
        # never changes the block's scheduling constraints.
        self.assertEqual(dut.output_multiple(), 1)
        self.assertEqual(run_tag_offsets(dut, 1, 24, 5), [1, 7, 13, 19])

        dut.set_data(np.arange(5))
        self.assertEqual(run_tag_offsets(dut, 1, 10, 5), [])


if __name__ == "__main__":
    gr_unittest.run(qa_vector_source_cupy)
