#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import cupy as cp
import numpy as np
from gnuradio import cuda


class nlog10_cupy(cuda.sync_block):
    """Computes output = n * log10(input) + k on the GPU using CuPy."""

    def __init__(self, n: float = 1.0, vlen: int = 1, k: float = 0.0):
        """
        Parameters
        ----------
        n : float
            Scalar multiplicative constant.
        vlen : int
            Vector length.
        k : float
            Scalar additive constant.
        """
        self._n = np.float32(n)
        self._k = np.float32(k)
        io_dtype = (np.float32, vlen) if vlen > 1 else np.float32
        cuda.sync_block.__init__(self, "nlog10_cupy", [io_dtype], [io_dtype])

    def work(self, input_items, output_items):
        x = input_items[0]
        cp.log10(cp.maximum(x, np.finfo(np.float32).tiny), out=output_items[0])
        if self._n != 1.0:
            output_items[0] *= self._n
        if self._k != 0.0:
            output_items[0] += self._k
        return len(output_items[0])
