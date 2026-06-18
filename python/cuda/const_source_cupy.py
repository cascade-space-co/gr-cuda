#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""GPU constant source (CuPy port of ``analog.const_source_x``)."""

import numpy as np
from gnuradio import cuda


class const_source_cupy(cuda.sync_block):
    """Emits a constant value on the GPU using CuPy.

    GPU counterpart of the in-tree Constant Source (a ``sig_source`` driven
    with ``GR_CONST_WAVE``).
    """

    def __init__(self, const=0, dtype: np.dtype = np.complex64):
        """
        Parameters
        ----------
        const : float or complex
            The constant value to emit.
        dtype : numpy.dtype
            Output sample type.
        """
        self._dtype = np.dtype(dtype)
        self._const = const
        cuda.sync_block.__init__(self, "const_source_cupy", None, [self._dtype])

    def work(self, input_items, output_items):
        out = output_items[0]
        out.fill(self._const)
        return len(out)

    def set_const(self, const):
        self._const = const

    def const(self):
        return self._const
