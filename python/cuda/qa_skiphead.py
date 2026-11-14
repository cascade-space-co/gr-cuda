#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import pmt
from gnuradio import blocks, gr, gr_unittest
from gnuradio.cuda import copy, skiphead


def make_tag(offset):
    tag = gr.tag_t()
    tag.key = pmt.intern("k")
    tag.value = pmt.from_long(offset)
    tag.offset = offset
    return tag


class qa_skiphead(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def run_chain(self, data, nskip, tags=()):
        src = blocks.vector_source_f(data, False, 1, list(tags))
        snk = blocks.vector_sink_f()
        self.tb.connect(
            src,
            copy(gr.sizeof_float),
            skiphead(gr.sizeof_float, nskip),
            copy(gr.sizeof_float),
            snk,
        )
        self.tb.run()
        return snk

    def test_001_skip(self):
        data = [float(x) for x in range(100_000)]
        snk = self.run_chain(data, 12_345)
        self.assertEqual(list(snk.data()), data[12_345:])

    def test_002_skip_nothing(self):
        data = [float(x) for x in range(10_000)]
        snk = self.run_chain(data, 0)
        self.assertEqual(list(snk.data()), data)

    def test_003_skip_everything(self):
        data = [float(x) for x in range(10_000)]
        snk = self.run_chain(data, 20_000)
        self.assertEqual(len(snk.data()), 0)

    def test_004_tags_are_dropped_or_shifted(self):
        n = 100_000
        nskip = 54_321
        offsets = [0, nskip - 1, nskip, nskip + 1, n - 1]
        snk = self.run_chain(
            [float(x) for x in range(n)], nskip, [make_tag(o) for o in offsets]
        )

        result = sorted((t.offset, pmt.to_long(t.value)) for t in snk.tags())
        expected = [(o - nskip, o) for o in offsets if o >= nskip]
        self.assertEqual(result, expected)

    def test_005_host_edges(self):
        data = [float(x) for x in range(10_000)]
        src = blocks.vector_source_f(data, False)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, skiphead(gr.sizeof_float, 100), snk)
        self.tb.run()
        self.assertEqual(list(snk.data()), data[100:])


if __name__ == "__main__":
    gr_unittest.run(qa_skiphead)
