#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import os
import time
import unittest
import numpy as np
from gnuradio import gr, gr_unittest, blocks, cuda


class slow_check_sink(gr.sync_block):
    """Slow sink that validates a known sequence."""
    def __init__(self, expected, sleep_s=0.0002):
        gr.sync_block.__init__(self, "slow_check_sink", in_sig=[np.complex64], out_sig=[])
        self._expected = expected
        self._offset = 0
        self._sleep_s = sleep_s
        self.mismatch_count = 0
        self.notmismatch_count = 0

    def work(self, input_items, output_items):
        in_data = input_items[0]
        n = len(in_data)
        if n == 0:
            return 0
        expected_slice = self._expected[self._offset:self._offset + n]
        if expected_slice.shape[0] != n:
            raise RuntimeError("Received more items than expected")
        if not np.array_equal(in_data, expected_slice):
            self.mismatch_count += 1
            print("Sequence mismatch (possible buffer sync issue)")
        else:
            self.notmismatch_count += 1
        self._offset += n
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        return n


class qa_cuda_buffer_race(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @unittest.skipUnless(os.environ.get("GR_CUDA_STRESS") == "1",
                         "set GR_CUDA_STRESS=1 to enable stress test")
    def test_001_slow_consumer_sequence(self):
        nitems = int(os.environ.get("GR_CUDA_STRESS_ITEMS", "1000000"))
        x = (np.arange(nitems, dtype=np.float32) +
             1j * np.zeros(nitems, dtype=np.float32)).astype(np.complex64)

        src = blocks.vector_source_c(x, False)
        gpu = cuda.multiply_const_cc(1.0)
        # Encourage small buffers / frequent transfers.
        min_buf = int(os.environ.get("GR_CUDA_STRESS_BUF", "1024"))
        gpu.set_max_output_buffer(min_buf)
        gpu.set_min_output_buffer(min_buf)

        sleep_s = float(os.environ.get("GR_CUDA_STRESS_SLEEP", "0.001"))
        checker = slow_check_sink(x, sleep_s=sleep_s)

        self.tb.connect(src, gpu)
        self.tb.connect(gpu, checker)
        self.tb.run()
        print(checker.notmismatch_count)
        if checker.mismatch_count:
            raise RuntimeError(f"Sequence mismatches: {checker.mismatch_count}")


if __name__ == '__main__':
    gr_unittest.run(qa_cuda_buffer_race)
