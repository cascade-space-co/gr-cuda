#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2021 Josh Morman.
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio.cuda import multiply_const_ff
import numpy as np

class qa_multiply_const(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        instance = multiply_const_ff(2.0, 1)

    def test_001_descriptive_test_name(self):
        # set up fg
        input_data = list(range(100000))
        expected_data = [x*2.0*3.0 for x in input_data]
        src = blocks.vector_source_f(input_data, False)
        op1 = multiply_const_ff(2.0)
        op2 = multiply_const_ff(3.0)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, op1, op2, snk)
        self.tb.run()
        self.assertEqual(snk.data(), expected_data)        

    def test_002_float_vlen(self):
        vlen = 4
        N = 10000
        k = 3.0
        input_data = np.random.randn(N * vlen).astype(np.float32)
        expected_data = input_data * k

        src = blocks.vector_source_f(input_data, False, vlen)
        dut = multiply_const_ff(k, vlen)
        snk = blocks.vector_sink_f(vlen)

        self.tb.connect(src, dut, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, expected_data, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    gr_unittest.run(qa_multiply_const)
