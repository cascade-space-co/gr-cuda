#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import pmt
from gnuradio import blocks, gr, gr_unittest
from gnuradio.cuda import copy, head


class qa_head(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_first_n_items(self):
        data = [float(x) for x in range(100_000)]
        n = 12_345
        src = blocks.vector_source_f(data, False)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, head(gr.sizeof_float, n), snk)
        self.tb.run()

        self.assertEqual(list(snk.data()), data[:n])

    def test_002_gpu_chain_with_tags(self):
        data = [float(x) for x in range(100_000)]
        n = 54_321
        tags = []
        for offset in (0, n - 1, n, n + 100):
            tag = gr.tag_t()
            tag.key = pmt.intern("k")
            tag.value = pmt.from_long(offset)
            tag.offset = offset
            tags.append(tag)
        src = blocks.vector_source_f(data, False, 1, tags)
        snk = blocks.vector_sink_f()

        self.tb.connect(
            src, copy(gr.sizeof_float), head(gr.sizeof_float, n), copy(gr.sizeof_float), snk
        )
        self.tb.run()

        self.assertEqual(list(snk.data()), data[:n])
        self.assertEqual(sorted(t.offset for t in snk.tags()), [0, n - 1])

    def test_003_no_output_stops_endless_source(self):
        n = 1_000_000
        src = blocks.null_source(gr.sizeof_gr_complex)
        hd = head(gr.sizeof_gr_complex, n)

        self.tb.connect(src, copy(gr.sizeof_gr_complex), hd)
        self.tb.run()

        self.assertEqual(hd.nitems_read(0), n)

    def test_004_set_length(self):
        data = [float(x) for x in range(1000)]
        src = blocks.vector_source_f(data, False)
        hd = head(gr.sizeof_float, 10)
        hd.set_length(500)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, hd, snk)
        self.tb.run()

        self.assertEqual(list(snk.data()), data[:500])

    def test_005_zero_items(self):
        src = blocks.null_source(gr.sizeof_float)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, head(gr.sizeof_float, 0), snk)
        self.tb.run()

        self.assertEqual(len(snk.data()), 0)


if __name__ == "__main__":
    gr_unittest.run(qa_head)
