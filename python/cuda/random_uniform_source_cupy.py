#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""GPU uniform random source (CuPy port of ``analog.random_uniform_source_x``)."""

import cupy as cp
import numpy as np
from gnuradio import cuda

from .noise_source_cupy import _normalize_seed


class random_uniform_source_cupy(cuda.sync_block):
    """Generates uniformly distributed random integers on the GPU using CuPy.

    GPU counterpart of ``analog.random_uniform_source_x``. Produces integer
    samples drawn uniformly from the half-open interval ``[minimum, maximum)``.

    The underlying RNG differs from the in-tree block (CuPy's generator versus
    GNU Radio's xoroshiro128+), so the exact sample sequence is not bit-for-bit
    reproducible against the CPU block; the statistical distribution matches.
    """

    def __init__(
        self,
        minimum: int = 0,
        maximum: int = 2,
        seed: int = 0,
        dtype: np.dtype = np.int32,
    ):
        """
        Parameters
        ----------
        minimum : int
            Inclusive lower bound.
        maximum : int
            Exclusive upper bound.
        seed : int
            RNG seed. 0 selects a nondeterministic seed.
        dtype : numpy.dtype
            Output integer type (int32, int16, uint8). Note the byte variant
            is unsigned, matching the in-tree ``random_uniform_source_b``
            (``std::uint8_t``).
        """
        self._dtype = np.dtype(dtype)
        self._min = int(minimum)
        self._max = int(maximum)
        if self._max <= self._min:
            raise ValueError("maximum must be greater than minimum")
        self._rng = cp.random.RandomState(_normalize_seed(seed))
        cuda.sync_block.__init__(
            self, "random_uniform_source_cupy", None, [self._dtype]
        )

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        if n == 0:
            return 0
        out[:] = self._rng.randint(self._min, self._max, size=n).astype(self._dtype)
        return n
