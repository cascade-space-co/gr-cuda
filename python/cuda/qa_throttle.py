#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Test GPU throttle block.
"""

import time
from gnuradio import gr, gr_unittest
from gnuradio import blocks
from gnuradio import cuda

try:
    import pmt
except Exception:  # pragma: no cover
    pmt = None

class test_throttle(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_throttle(self):
        """
        Smoke test: block runs and stops without hanging.
        """
        sample_rate = 1e6  # 1 MS/s
        target_duration = 0.10

        src = cuda.null_source(gr.sizeof_float)
        throttle = cuda.throttle(gr.sizeof_float, sample_rate)
        sink = cuda.null_sink(gr.sizeof_float)

        self.tb.connect(src, throttle)
        self.tb.connect(throttle, sink)

        start = time.time()
        self.tb.start()
        time.sleep(target_duration)
        self.tb.stop()
        self.tb.wait()

        elapsed = time.time() - start
        self.assertGreater(elapsed, 0.0)

    def test_002_rate_accuracy(self):
        """
        Verify rate accuracy using probe_rate messages.

        This test intentionally uses a loose tolerance to avoid flakiness across
        different GPUs/hosts.
        """
        if pmt is None:
            self.skipTest("pmt module not available; cannot parse probe_rate messages")

        sample_rate = 5e6  # items/s
        duration = 0.8
        update_rate_ms = 100.0
        # Higher alpha converges faster so we can keep this test short.
        alpha = 0.2

        src = cuda.null_source(gr.sizeof_gr_complex)
        throttle = cuda.throttle(gr.sizeof_gr_complex, sample_rate)
        probe = cuda.probe_rate(gr.sizeof_gr_complex, update_rate_ms, alpha, "throttle_test")
        dbg = blocks.message_debug(True, gr.log_levels.info)

        self.tb.connect(src, throttle)
        self.tb.connect(throttle, probe)
        self.tb.msg_connect((probe, "rate"), (dbg, "store"))

        self.tb.start()
        time.sleep(duration)
        self.tb.stop()
        self.tb.wait()

        # message_debug API varies by GNU Radio version; handle common cases.
        if not hasattr(dbg, "num_messages") or not hasattr(dbg, "get_message"):
            self.skipTest("message_debug does not expose num_messages/get_message in this GNU Radio build")

        nmsgs = int(dbg.num_messages())
        # Expect multiple updates for a stable estimate.
        self.assertGreaterEqual(nmsgs, 3)

        last = dbg.get_message(nmsgs - 1)
        rate_avg_pmt = pmt.dict_ref(last, pmt.intern("rate_avg"), pmt.PMT_NIL)
        rate_now_pmt = pmt.dict_ref(last, pmt.intern("rate_now"), pmt.PMT_NIL)
        self.assertFalse(pmt.is_null(rate_avg_pmt))
        self.assertFalse(pmt.is_null(rate_now_pmt))

        rate_avg = float(pmt.to_double(rate_avg_pmt))
        rate_now = float(pmt.to_double(rate_now_pmt))

        # Loose tolerance to avoid flakiness across machines (short test duration).
        tol = 0.02  # 2%
        self.assertGreater(rate_avg, 0.0)
        self.assertGreater(rate_now, 0.0)
        self.assertAlmostEqual(rate_avg, sample_rate, delta=sample_rate * tol)

if __name__ == '__main__':
    gr_unittest.run(test_throttle)


