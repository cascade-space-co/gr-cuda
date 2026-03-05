#!/usr/bin/env python
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Tests that stream tags survive transit through GPU blocks.

import numpy as np
import pmt
from gnuradio import blocks, gr, gr_unittest
from gnuradio.cuda import copy, multiply_const_ff


def make_tag(key, value, offset, srcid=None):
    tag = gr.tag_t()
    tag.key = pmt.string_to_symbol(key)
    tag.value = pmt.to_pmt(value)
    tag.offset = offset
    if srcid is not None:
        tag.srcid = pmt.to_pmt(srcid)
    return tag


def tags_equal(a, b):
    return (
        a.offset == b.offset and pmt.equal(a.key, b.key) and pmt.equal(a.value, b.value)
    )


class qa_tag_propagation(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_single_tag_through_copy(self):
        """One tag at offset 0 should pass through a GPU copy block."""
        src_data = [float(x) for x in range(1000)]
        src_tags = [make_tag("key", "val", 0, "src")]

        src = blocks.vector_source_f(src_data, repeat=False, tags=src_tags)
        op = copy(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, op, snk)
        self.tb.run()

        self.assertEqual(list(snk.data()), src_data)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), 1)
        self.assertTrue(tags_equal(src_tags[0], result_tags[0]))

    def test_002_multiple_tags(self):
        """Multiple tags at different offsets through GPU copy."""
        N = 10_000
        src_data = [float(x) for x in range(N)]
        src_tags = [
            make_tag("start", "begin", 0),
            make_tag("mid", 42, N // 2),
            make_tag("end", "done", N - 1),
        ]

        src = blocks.vector_source_f(src_data, repeat=False, tags=src_tags)
        op = copy(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, op, snk)
        self.tb.run()

        self.assertEqual(list(snk.data()), src_data)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), 3)
        for expected, actual in zip(src_tags, result_tags, strict=True):
            self.assertTrue(
                tags_equal(expected, actual),
                f"Tag mismatch at offset {expected.offset}",
            )

    def test_003_tags_through_chained_copies(self):
        """Tags through 3 cascaded GPU copy blocks."""
        src_data = [float(x) for x in range(5000)]
        src_tags = [
            make_tag("a", 1, 0),
            make_tag("b", 2, 1000),
            make_tag("c", 3, 4999),
        ]

        src = blocks.vector_source_f(src_data, repeat=False, tags=src_tags)
        copies = [copy(gr.sizeof_float) for _ in range(3)]
        snk = blocks.vector_sink_f()

        self.tb.connect(src, copies[0])
        self.tb.connect(copies[0], copies[1])
        self.tb.connect(copies[1], copies[2])
        self.tb.connect(copies[2], snk)

        self.tb.run()

        self.assertEqual(list(snk.data()), src_data)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), 3)
        for expected, actual in zip(src_tags, result_tags, strict=True):
            self.assertTrue(
                tags_equal(expected, actual),
                f"Tag mismatch at offset {expected.offset}",
            )

    def test_004_tags_through_multiply_const(self):
        """Tags through a GPU compute block (multiply_const)."""
        N = 10_000
        src_data = np.arange(N, dtype=np.float32)
        src_tags = [
            make_tag("rate", 1e6, 0),
            make_tag("burst", True, 5000),
        ]

        src = blocks.vector_source_f(src_data, repeat=False, tags=src_tags)
        op = multiply_const_ff(2.0)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, op, snk)
        self.tb.run()

        expected_data = src_data * 2.0
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, expected_data, rtol=1e-5)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), 2)
        for expected, actual in zip(src_tags, result_tags, strict=True):
            self.assertTrue(
                tags_equal(expected, actual),
                f"Tag mismatch at offset {expected.offset}",
            )

    def test_005_tags_with_repeat(self):
        """Tags should repeat when source repeats (through GPU copy)."""
        length = 100
        total = 2 * length
        src_data = [float(x) for x in range(length)]
        src_tags = [make_tag("pkt", 0, 0)]
        expected_tags = [
            make_tag("pkt", 0, 0),
            make_tag("pkt", 0, length),
        ]

        src = blocks.vector_source_f(src_data, repeat=True, tags=src_tags)
        head = blocks.head(gr.sizeof_float, total)
        op = copy(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, head, op, snk)
        self.tb.run()

        result_data = list(snk.data())
        self.assertEqual(len(result_data), total)
        self.assertEqual(result_data, src_data + src_data)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), 2)
        for expected, actual in zip(expected_tags, result_tags, strict=True):
            self.assertTrue(
                tags_equal(expected, actual),
                f"Tag mismatch at offset {expected.offset}",
            )

    def test_006_many_tags_large_buffer(self):
        """100 tags spread across 1M samples through GPU copy."""
        N = 1_000_000
        src_data = np.arange(N, dtype=np.float32)
        tag_offsets = list(range(0, N, N // 100))
        src_tags = [make_tag("seq", i, off) for i, off in enumerate(tag_offsets)]

        src = blocks.vector_source_f(src_data, repeat=False, tags=src_tags)
        op = copy(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, op, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), len(src_tags))
        for expected, actual in zip(src_tags, result_tags, strict=True):
            self.assertTrue(
                tags_equal(expected, actual),
                f"Tag mismatch at offset {expected.offset}",
            )

    def test_007_complex_tags_through_copy(self):
        """Tags on complex data through GPU copy."""
        N = 5000
        src_data = (
            np.arange(N, dtype=np.float32) + 1j * np.arange(N, dtype=np.float32)
        ).astype(np.complex64)
        src_tags = [make_tag("freq", 1.42e9, 0)]

        src = blocks.vector_source_c(src_data, repeat=False, tags=src_tags)
        op = copy(gr.sizeof_gr_complex)
        snk = blocks.vector_sink_c()

        self.tb.connect(src, op, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.complex64)
        np.testing.assert_array_equal(result, src_data)

        result_tags = snk.tags()
        self.assertEqual(len(result_tags), 1)
        self.assertTrue(tags_equal(src_tags[0], result_tags[0]))


if __name__ == "__main__":
    gr_unittest.run(qa_tag_propagation)
