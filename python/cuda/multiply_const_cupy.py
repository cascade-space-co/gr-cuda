#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from gnuradio import cuda


class multiply_const_cupy(cuda.sync_block):
    """
    Multiplies input stream by a constant value (k) on the GPU using CuPy.
    """
    def __init__(self, k, dtype=np.complex64, vlen=1):
        self.k = k
        io_dtype = (np.dtype(dtype), vlen) if vlen > 1 else np.dtype(dtype)
        cuda.sync_block.__init__(self, "multiply_const_cupy",
            [io_dtype],
            [io_dtype])

    def work(self, input_items, output_items):
        cp.multiply(input_items[0], self.k, out=output_items[0])
        return len(output_items[0])

    def set_k(self, k):
        self.k = k
