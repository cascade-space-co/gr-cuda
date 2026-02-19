#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda
import cupy as cp

try:
    from .multiply_const_py import multiply_const_py
    from .add_py import add_py
except ImportError:
    from multiply_const_py import multiply_const_py
    from add_py import add_py

class qa_fanin_sync(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_fanin(self):
        """
        Test fan-in from multiple parallel GPU blocks to a single consumer.
        This verifies that the consumer correctly waits for multiple upstream streams.
        """
        N = 10000
        
        # Generate random complex data
        src_data = np.random.randn(N) + 1j * np.random.randn(N)
        src_data = src_data.astype(np.complex64)
        
        src = blocks.vector_source_c(src_data, False)
        
        # Branch 1: Multiply by 2.0
        # This will run on its own stream
        mult1 = multiply_const_py(2.0, dtype=np.complex64)
        
        # Branch 2: Multiply by 3.0
        # This will run on its own stream (different from mult1)
        mult2 = multiply_const_py(3.0, dtype=np.complex64)
        
        # Merge: Add branch 1 and branch 2
        # This runs on yet another stream and must wait for both mult1 and mult2
        add_blk = add_py(num_inputs=2, dtype=np.complex64)
        
        snk = blocks.vector_sink_c()
        
        # Connect
        # src -> mult1 -> add_blk -> snk
        #     -> mult2 -> (port 1 of add_blk)
        # Implicit copies src->mult1, src->mult2, add_blk->snk
        
        self.tb.connect(src, mult1)
        self.tb.connect(src, mult2)
        
        self.tb.connect(mult1, (add_blk, 0))
        self.tb.connect(mult2, (add_blk, 1))
        
        self.tb.connect(add_blk, snk)
        
        # Run
        self.tb.run()
        
        # Verify
        result = np.array(snk.data(), dtype=np.complex64)
        expected = (src_data * 2.0) + (src_data * 3.0) # == src_data * 5.0
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    gr_unittest.run(qa_fanin_sync)
