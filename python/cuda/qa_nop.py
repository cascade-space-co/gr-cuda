#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import time

from gnuradio import blocks, gr, gr_unittest

try:
    from gnuradio import cuda
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio import cuda


class qa_nop(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        cuda.nop(gr.sizeof_gr_complex)

    def test_data_flows(self):
        """Verify the pipeline runs without deadlock and items flow."""
        src = blocks.null_source(gr.sizeof_float)
        op = cuda.nop(gr.sizeof_float)
        snk = cuda.null_sink(gr.sizeof_float)

        self.tb.connect(src, op, snk)
        self.tb.start()
        time.sleep(0.2)
        self.tb.stop()
        self.tb.wait()

        self.assertGreater(snk.nitems_read(0), 0)

    def test_h2d_d2h(self):
        """CPU -> nop -> CPU: exercises both H2D and D2H transfers."""
        nsamples = 8192
        src = blocks.null_source(gr.sizeof_float)
        head = blocks.head(gr.sizeof_float, nsamples)
        op = cuda.nop(gr.sizeof_float)
        snk = blocks.null_sink(gr.sizeof_float)

        self.tb.connect(src, head, op, snk)
        self.tb.run()

        self.assertEqual(snk.nitems_read(0), nsamples)


if __name__ == "__main__":
    gr_unittest.run(qa_nop)
