#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import pmt
from gnuradio import blocks, gr, gr_unittest
from gnuradio.cuda import copy, tag_debug


def make_tag(key, value, offset):
    tag = gr.tag_t()
    tag.key = pmt.intern(key)
    tag.value = pmt.to_pmt(value)
    tag.offset = offset
    return tag


def summary(tags):
    return sorted((t.offset, pmt.symbol_to_string(t.key)) for t in tags)


class qa_tag_debug(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def make_sink(self, *args, **kwargs):
        snk = tag_debug(gr.sizeof_float, "qa", *args, **kwargs)
        snk.set_display(False)
        snk.set_save_all(True)
        return snk

    def test_001_tags_from_gpu_stream(self):
        n = 100_000
        tags = [make_tag("a", 1, 0), make_tag("b", 2, n // 2), make_tag("c", 3, n - 1)]
        src = blocks.vector_source_f([0.0] * n, False, 1, tags)
        snk = self.make_sink()

        self.tb.connect(src, copy(gr.sizeof_float), snk)
        self.tb.run()

        self.assertEqual(summary(snk.current_tags()), summary(tags))
        self.assertEqual(snk.nitems_read(0), n)

    def test_002_key_filter(self):
        n = 10_000
        tags = [make_tag("keep", i, i * 1000) for i in range(5)]
        tags += [make_tag("drop", i, i * 1000 + 1) for i in range(5)]
        src = blocks.vector_source_f([0.0] * n, False, 1, tags)
        snk = self.make_sink("keep")

        self.tb.connect(src, copy(gr.sizeof_float), snk)
        self.tb.run()

        self.assertEqual(snk.key_filter(), "keep")
        self.assertEqual(
            summary(snk.current_tags()), [(i * 1000, "keep") for i in range(5)]
        )

    def test_003_multiple_inputs(self):
        n = 10_000
        snk = self.make_sink(num_inputs=2)
        for port, key in enumerate(["in0", "in1"]):
            tags = [make_tag(key, port, 10 * (port + 1))]
            src = blocks.vector_source_f([0.0] * n, False, 1, tags)
            self.tb.connect(src, copy(gr.sizeof_float), (snk, port))
        self.tb.run()

        self.assertEqual(summary(snk.current_tags()), [(10, "in0"), (20, "in1")])

    def test_004_empty_key_filter(self):
        snk = self.make_sink()
        self.assertEqual(snk.key_filter(), "")


if __name__ == "__main__":
    gr_unittest.run(qa_tag_debug)
