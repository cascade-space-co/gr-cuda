#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import cuda


class add_cupy(cuda.sync_block):
    """Adds N input streams element-wise on the GPU using CuPy."""
    def __init__(self, num_inputs=2, dtype=np.complex64, vlen=1):
        self.num_inputs = num_inputs
        io_dtype = (np.dtype(dtype), vlen) if vlen > 1 else np.dtype(dtype)
        cuda.sync_block.__init__(self, "add_cupy",
            [io_dtype] * num_inputs,
            [io_dtype])

    def work(self, input_items, output_items):
        output_items[0][:] = sum(input_items)
        return len(output_items[0])