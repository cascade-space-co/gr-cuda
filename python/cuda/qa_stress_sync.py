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


class qa_stress_sync(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_chain(self):
        """
        Test a long chain of multiply_const_cupy blocks.
        This verifies that synchronization between multiple GPU blocks
        (each with its own stream) works correctly.
        """
        # Parameters
        chain_length = 100
        N = 100000  # Large enough to trigger multiple work calls

        # Generate random complex data
        src_data = np.random.randn(N) + 1j * np.random.randn(N)
        src_data = src_data.astype(np.complex64)

        src = blocks.vector_source_c(src_data, False)

        # Move to GPU
        # Note: We use explicit cuda.copy here to verify explicit buffer management.
        # It is not strictly needed; connecting src -> chain[0] directly would also work
        # (implicit copy handled by the runtime).
        to_dev = cuda.copy(np.dtype(np.complex64).itemsize)

        # Move from GPU
        from_dev = cuda.copy(np.dtype(np.complex64).itemsize)

        # Create chain of blocks
        # We alternate multiplying by 2.0 and 0.5. Net effect should be 1.0.
        chain = []
        for i in range(chain_length):
            k = 2.0 if i % 2 == 0 else 0.5
            blk = multiply_const_cupy(k, dtype=np.complex64)
            chain.append(blk)

        snk = blocks.vector_sink_c()

        # Connect: src -> to_dev -> chain[0] -> ... -> chain[N-1] -> from_dev -> snk
        self.tb.connect(src, to_dev)
        if chain:
            self.tb.connect(to_dev, chain[0])
            for i in range(len(chain) - 1):
                self.tb.connect(chain[i], chain[i + 1])
            self.tb.connect(chain[-1], from_dev)
        else:
            self.tb.connect(to_dev, from_dev)

        self.tb.connect(from_dev, snk)

        # Run
        self.tb.run()

        # Verify
        result = np.array(snk.data(), dtype=np.complex64)

        # Net effect of alternating *2.0 and *0.5 is identity.
        # Allow for floating point accumulation error.
        np.testing.assert_allclose(result, src_data, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    gr_unittest.run(qa_stress_sync)
