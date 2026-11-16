#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import time

from gnuradio import blocks, cuda, gr, gr_unittest

TYPES = [
    ("b", gr.sizeof_char, [3, 1, 4, 1, 5, 9, 2, 6]),
    ("s", gr.sizeof_short, [-300, 200, -100, 12345]),
    ("i", gr.sizeof_int, [-70000, 5, 1 << 30, -42]),
    ("f", gr.sizeof_float, [0.5, -1.25, 3.0, 2.75]),
    ("c", gr.sizeof_gr_complex, [1 + 2j, -3 - 4j, 0.5j, 7 - 0.25j]),
]


class qa_probe_signal(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_last_sample_each_type(self):
        for suffix, size, data in TYPES:
            with self.subTest(type=suffix):
                tb = gr.top_block()
                src = getattr(blocks, f"vector_source_{suffix}")(data * 1000, False)
                probe = getattr(cuda, f"probe_signal_{suffix}")()
                tb.connect(src, cuda.copy(size), probe)
                tb.run()
                self.assertEqual(probe.level(), data[-1])

    def test_002_zero_before_first_work(self):
        self.assertEqual(cuda.probe_signal_f().level(), 0.0)
        self.assertEqual(cuda.probe_signal_vc(3).level(), [0j, 0j, 0j])

    def test_003_last_vector_each_type(self):
        vlen = 4
        for suffix, size, data in TYPES:
            with self.subTest(type=suffix):
                tb = gr.top_block()
                # 1001 vectors, the last one being the last vlen values of data.
                last = data[-vlen:]
                stream = [0] * (vlen * 1000) + last
                src = getattr(blocks, f"vector_source_{suffix}")(stream, False, vlen)
                probe = getattr(cuda, f"probe_signal_v{suffix}")(vlen)
                tb.connect(src, cuda.copy(size * vlen), probe)
                tb.run()
                self.assertEqual(list(probe.level()), last)

    def test_004_read_while_running(self):
        # Every sample is in [0, 1000), so any value read mid-run must be too.
        src = blocks.vector_source_f([float(x) for x in range(1000)], True)
        probe = cuda.probe_signal_f()
        self.tb.connect(src, cuda.copy(gr.sizeof_float), probe)
        self.tb.start()
        values = set()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            values.add(probe.level())
        self.tb.stop()
        self.tb.wait()

        self.assertTrue(values <= {float(x) for x in range(1000)})
        self.assertGreater(len(values), 1)


if __name__ == "__main__":
    gr_unittest.run(qa_probe_signal)
