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
from gnuradio import gr, cuda


class add_cupy(cuda.sync_block):
    """
    Adds N input streams together on the GPU using CuPy.

    Uses the convenience base class ``cuda.sync_block`` which automatically
    handles CUDA synchronization and CuPy array conversion.
    """
    def __init__(self, num_inputs=2, dtype=np.complex64, vlen=1):
        self.num_inputs = num_inputs
        io_dtype = (np.dtype(dtype), vlen) if vlen > 1 else np.dtype(dtype)
        cuda.sync_block.__init__(self, "add_cupy",
            [io_dtype] * num_inputs,
            [io_dtype])

    def work(self, input_items, output_items):
        output_items[0][:] = sum(input_items)
        return len(output_items[0])


class add_cupy_mixin(cuda.cuda_block, gr.sync_block):
    """
    Adds N input streams together on the GPU using CuPy.

    Uses the ``cuda_block`` mixin directly with ``gr.sync_block``, which
    is equivalent to ``cuda.sync_block`` but shows the inheritance explicitly.
    """
    def __init__(self, num_inputs=2, dtype=np.complex64, vlen=1):
        self.num_inputs = num_inputs
        io_dtype = (np.dtype(dtype), vlen) if vlen > 1 else np.dtype(dtype)
        cuda.cuda_block.__init__(self, "add_cupy_mixin",
            [io_dtype] * num_inputs,
            [io_dtype])

    def work(self, input_items, output_items):
        output_items[0][:] = sum(input_items)
        return len(output_items[0])


class add_cupy_decorator(cuda.cuda_block, gr.sync_block):
    """
    Adds N input streams together on the GPU using CuPy.

    Uses the ``@cuda.gpu_work`` decorator instead of the implicit
    ``__init_subclass__`` wrapping.  The decorator makes the CUDA sync +
    CuPy conversion visible at the call site.
    """
    def __init__(self, num_inputs=2, dtype=np.complex64, vlen=1):
        self.num_inputs = num_inputs
        io_dtype = (np.dtype(dtype), vlen) if vlen > 1 else np.dtype(dtype)
        cuda.cuda_block.__init__(self, "add_cupy_decorator",
            [io_dtype] * num_inputs,
            [io_dtype])

    @cuda.gpu_work
    def work(self, input_items, output_items):
        output_items[0][:] = sum(input_items)
        return len(output_items[0])