#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import threading
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

    def test_005_vectors_not_torn_with_concurrent_readers(self):
        # Every vector is [k, k, k, k], so a vector read while a newer one is
        # being copied in would show mixed values.
        vlen = 4
        stream = [float(k) for k in range(1000) for _ in range(vlen)]
        src = blocks.vector_source_f(stream, True, vlen)
        probe = cuda.probe_signal_vf(vlen)
        self.tb.connect(src, cuda.copy(gr.sizeof_float * vlen), probe)

        torn = []
        values = set()

        def poll():
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                v = list(probe.level())
                if len(set(v)) != 1:
                    torn.append(v)
                values.add(v[0])

        self.tb.start()
        readers = [threading.Thread(target=poll) for _ in range(2)]
        for r in readers:
            r.start()
        for r in readers:
            r.join()
        self.tb.stop()
        self.tb.wait()

        self.assertEqual(torn, [])
        self.assertGreater(len(values), 1)

    def test_006_zero_size_rejected(self):
        with self.assertRaises(ValueError):
            cuda.probe_signal_vf(0)


if __name__ == "__main__":
    gr_unittest.run(qa_probe_signal)
