#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import cupy as cp
import numpy as np
from gnuradio import cuda


class complex_to_mag_squared_cupy(cuda.sync_block):
    """Computes the squared magnitude of complex samples on the GPU using CuPy."""

    def __init__(self, vlen: int = 1):
        """
        Parameters
        ----------
        vlen : int
            Vector length.
        """
        in_dtype = (np.complex64, vlen) if vlen > 1 else np.complex64
        out_dtype = (np.float32, vlen) if vlen > 1 else np.float32
        cuda.sync_block.__init__(
            self, "complex_to_mag_squared_cupy", [in_dtype], [out_dtype]
        )

    def work(self, input_items, output_items):
        x = input_items[0]
        cp.add(x.real**2, x.imag**2, out=output_items[0])
        return len(output_items[0])
