#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Zero-copy edges: GPU reads and writes that cross the halfway point of the
host ring.

On integrated GPUs, H2D and D2H edges hand blocks pointers into the host
ring's double-mapped 2N region.  A block's range may start before N and run
past it into the alias of the first half.  Both CUDA copies and kernels must
accept such ranges and see the aliased data.
"""

import os
import threading
import unittest

import cupy as cp
import numpy as np
from gnuradio import blocks, gr, gr_unittest

try:
    from gnuradio import cuda
except ImportError:
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio import cuda

# Fix the ring size so the test wraps it regardless of the user's prefs.
# cuda_buffer reads this once per process, at its first allocation, and the
# environment overrides config files.
RING_BYTES = 8 << 20
os.environ["GR_CONF_CUDA_BUFFER_MIN_BUFFER_BYTES"] = str(RING_BYTES)
RING_ITEMS = RING_BYTES // gr.sizeof_gr_complex

# Primes, so neither the pattern nor the GPU block's calls line up with the ring.
PATTERN_LEN = 1_000_003
CALL_ITEMS = 300_007
TOTAL_ITEMS = 4 * RING_ITEMS + PATTERN_LEN
TIMEOUT_S = 60


def zero_copy_active():
    rt = cp.cuda.runtime
    device = cp.cuda.Device().id
    integrated = rt.deviceGetAttribute(rt.cudaDevAttrIntegrated, device)
    host_page_tables = rt.deviceGetAttribute(
        rt.cudaDevAttrPageableMemoryAccessUsesHostPageTables, device
    )
    host_pointer_ok = rt.deviceGetAttribute(
        rt.cudaDevAttrCanUseHostPointerForRegisteredMem, device
    )
    return bool(integrated and host_page_tables and host_pointer_ok)


@unittest.skipUnless(zero_copy_active(), "zero-copy is only used on integrated GPUs")
class qa_zero_copy_wrap(gr_unittest.TestCase):
    def run_through(self, gpu_block):
        idx = np.arange(PATTERN_LEN, dtype=np.float32)
        pattern = (idx - 1j * idx).astype(np.complex64)

        tb = gr.top_block()
        src = blocks.vector_source_c(pattern.tolist(), repeat=True)
        head = blocks.head(gr.sizeof_gr_complex, TOTAL_ITEMS)
        sink = blocks.vector_sink_c()
        gpu_block.set_max_noutput_items(CALL_ITEMS)
        tb.connect(src, head, gpu_block, sink)

        # A failed CUDA call kills only the GPU block's thread, which can
        # leave tb.run() waiting forever.
        runner = threading.Thread(target=tb.run)
        runner.start()
        runner.join(TIMEOUT_S)
        if runner.is_alive():
            tb.stop()
            runner.join()
            self.fail(f"flowgraph did not finish within {TIMEOUT_S} s")

        out = np.asarray(sink.data(), dtype=np.complex64)
        self.assertEqual(out.size, TOTAL_ITEMS)
        expected = np.resize(pattern, TOTAL_ITEMS)
        mismatches = np.count_nonzero(out != expected)
        self.assertEqual(mismatches, 0, f"{mismatches} samples differ")

    def test_001_copy_crosses_halfway(self):
        """cudaMemcpyAsync on ranges that run past N."""
        self.run_through(cuda.copy(gr.sizeof_gr_complex))

    def test_002_kernel_crosses_halfway(self):
        """A kernel reading and writing ranges that run past N."""
        self.run_through(cuda.multiply_const_cc(1.0))


if __name__ == "__main__":
    gr_unittest.run(qa_zero_copy_wrap)
