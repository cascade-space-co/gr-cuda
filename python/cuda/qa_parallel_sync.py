#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest

try:
    from .multiply_const_cupy import multiply_const_cupy
except ImportError:
    from multiply_const_cupy import multiply_const_cupy


class qa_parallel_sync(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_fanout(self):
        """
        Test fan-out from a single GPU buffer to multiple consumers.
        This tests if the buffer management handles multiple readers correctly
        and if synchronization holds across branches.
        """
        # Parameters
        N = 10000

        # Generate random complex data
        src_data = np.random.randn(N) + 1j * np.random.randn(N)
        src_data = src_data.astype(np.complex64)

        src = blocks.vector_source_c(src_data, False)

        # Move to GPU
        # Note: We use explicit cuda.copy here to verify explicit buffer management.
        # It is not strictly needed; connecting src -> mult1 directly would also work
        # (implicit copy handled by the runtime).
        to_dev = cuda.copy(np.dtype(np.complex64).itemsize)

        # Branch 1: Multiply by 2.0
        mult1 = multiply_const_cupy(2.0, dtype=np.complex64)
        from_dev1 = cuda.copy(np.dtype(np.complex64).itemsize)
        snk1 = blocks.vector_sink_c()

        # Branch 2: Multiply by 3.0
        mult2 = multiply_const_cupy(3.0, dtype=np.complex64)
        from_dev2 = cuda.copy(np.dtype(np.complex64).itemsize)
        snk2 = blocks.vector_sink_c()

        # Connect
        # src -> to_dev -> mult1 -> from_dev1 -> snk1
        #               -> mult2 -> from_dev2 -> snk2

        self.tb.connect(src, to_dev)

        self.tb.connect(to_dev, mult1)
        self.tb.connect(mult1, from_dev1)
        self.tb.connect(from_dev1, snk1)

        self.tb.connect(to_dev, mult2)
        self.tb.connect(mult2, from_dev2)
        self.tb.connect(from_dev2, snk2)

        # Run
        self.tb.run()

        # Verify Branch 1
        result1 = np.array(snk1.data(), dtype=np.complex64)
        expected1 = src_data * 2.0
        np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-5)

        # Verify Branch 2
        result2 = np.array(snk2.data(), dtype=np.complex64)
        expected2 = src_data * 3.0
        np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    gr_unittest.run(qa_parallel_sync)
