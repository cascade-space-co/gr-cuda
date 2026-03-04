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


class wrap_check_sink(gr.sync_block):
    """Sink that validates a repeating known sequence."""
    def __init__(self, expected):
        gr.sync_block.__init__(self, "wrap_check_sink", in_sig=[np.complex64], out_sig=[])
        self._expected = expected
        self._expected_len = len(expected)
        self._offset = 0
        self.mismatch_count = 0
        self.notmismatch_count = 0

    def _expected_slice(self, n):
        start = self._offset % self._expected_len
        result = np.empty(n, dtype=self._expected.dtype)
        written = 0
        pos = start
        while written < n:
            chunk = min(n - written, self._expected_len - pos)
            result[written:written + chunk] = self._expected[pos:pos + chunk]
            written += chunk
            pos = 0
        return result

    def work(self, input_items, output_items):
        in_data = input_items[0]
        n = len(in_data)
        if n == 0:
            return 0
        expected_slice = self._expected_slice(n)
        if not np.array_equal(in_data, expected_slice):
            self.mismatch_count += 1
            print("Sequence mismatch (possible buffer sync issue), count: ", self.mismatch_count, "not mismatch count: ", self.notmismatch_count)
            raise RuntimeError("Sequence mismatch (possible buffer sync issue)")
        else:
            self.notmismatch_count += 1
        self._offset += n
        return n


class qa_long_race_stress(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @unittest.skipUnless(os.environ.get("GR_CUDA_LONG_STRESS") == "1",
                         "set GR_CUDA_LONG_STRESS=1 to enable long stress test")
    def test_001_fanin_fanout_chain_long(self):
        duration_s = float(os.environ.get("GR_CUDA_STRESS_SECONDS", "3600"))
        nitems = int(os.environ.get("GR_CUDA_STRESS_ITEMS", "65536"))
        min_buf = int(os.environ.get("GR_CUDA_STRESS_BUF", "1024"))
        chain_len = int(os.environ.get("GR_CUDA_STRESS_CHAIN", "64"))
        fanout = int(os.environ.get("GR_CUDA_STRESS_FANOUT", "10"))
        cpu_fanout = int(os.environ.get("GR_CUDA_STRESS_CPU_FANOUT", "1"))

        x = (np.arange(nitems, dtype=np.float32) +
             1j * np.zeros(nitems, dtype=np.float32)).astype(np.complex64)

        src = blocks.vector_source_c(x, True)
        add_blk = cuda.add_cc(num_inputs=fanout)
        add_blk.set_max_output_buffer(min_buf)
        add_blk.set_min_output_buffer(min_buf)

        branches = []
        for branch_idx in range(fanout):
            chain = []
            for i in range(chain_len):
                k = 2.0 if i % 2 == 0 else 0.5
                blk = cuda.multiply_const_cc(k)
                blk.set_max_output_buffer(min_buf)
                blk.set_min_output_buffer(min_buf)
                chain.append(blk)
            branches.append(chain)

        # Alternating 2.0 and 0.5 yields net gain of 1.0 for even length,
        # 2.0 for odd length (first multiply is 2.0).
        chain_gain = 2.0 if (chain_len % 2) == 1 else 1.0
        expected = x * (fanout * chain_gain)
        
        # Create multiple CPU checkers for GPU-to-CPU fanout stress test
        checkers = [wrap_check_sink(expected) for _ in range(cpu_fanout)]

        if fanout == 1:
            chain = branches[0]
            if not chain:
                # No GPU chain - connect src directly to all CPU checkers
                for checker in checkers:
                    self.tb.connect(src, checker)
            else:
                self.tb.connect(src, chain[0])
                for i in range(len(chain) - 1):
                    self.tb.connect(chain[i], chain[i + 1])
                # Fan out from last GPU block to all CPU checkers
                for checker in checkers:
                    self.tb.connect(chain[-1], checker)
        else:
            for branch_idx, chain in enumerate(branches):
                if not chain:
                    self.tb.connect(src, (add_blk, branch_idx))
                    continue
                self.tb.connect(src, chain[0])
                for i in range(len(chain) - 1):
                    self.tb.connect(chain[i], chain[i + 1])
                self.tb.connect(chain[-1], (add_blk, branch_idx))

            # Fan out from GPU add block to all CPU checkers
            for checker in checkers:
                self.tb.connect(add_blk, checker)

        self.tb.start()
        start = time.time()
        try:
            while time.time() - start < duration_s:
                time.sleep(0.5)
        finally:
            self.tb.stop()
            self.tb.wait()

        total_mismatches = sum(c.mismatch_count for c in checkers)
        if total_mismatches:
            for i, c in enumerate(checkers):
                if c.mismatch_count:
                    print(f"Checker {i}: mismatches={c.mismatch_count}, ok={c.notmismatch_count}")
        self.assertEqual(total_mismatches, 0, f"Total sequence mismatches: {total_mismatches}")


if __name__ == '__main__':
    gr_unittest.run(qa_long_race_stress)
