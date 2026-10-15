#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import math
import operator
import threading

import numpy as np
from gnuradio import cuda


def _integer(value, name, minimum):
    """Return an integer-like value after validating that it is >= minimum."""
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer >= {minimum}") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _check(m, n, offset):
    if m > n:
        raise ValueError(f"m ({m}) must not exceed n ({n})")
    if offset >= n:
        raise ValueError(f"offset ({offset}) must be less than n ({n})")


class keep_m_in_n_cupy(cuda.basic_block):
    """Keep ``m`` input items out of every ``n`` on the GPU.

    This is the CUDA-buffer equivalent of ``blocks.keep_m_in_n``. From each
    block of ``n`` input items it outputs the ``m`` items starting at
    ``offset``. If ``offset + m > n``, the selection wraps to the start of the
    same block; kept items always stay in their original order. Tags are
    propagated by the scheduler at the relative rate ``m / n``.

    Parameters
    ----------
    m : int
        Number of items kept per block. Must satisfy ``1 <= m <= n``.
    n : int
        Block size in items. Must be positive.
    offset : int, optional
        Index of the first kept item in each block. Must satisfy
        ``0 <= offset < n``. Default is 0.
    vlen : int, optional
        Number of scalar values in each stream item. Default is 1.
    dtype : numpy dtype, optional
        Scalar data type. Default is ``numpy.float32``.
    output_multiple : int, optional
        Ask the scheduler for output batches in multiples of this size, which
        can amortize per-call overhead. The block always produces whole groups
        of ``m``, so the scheduler is given the least common multiple of
        ``m`` and this value. Default is 1.
    """

    def __init__(
        self,
        m: int,
        n: int,
        offset: int = 0,
        vlen: int = 1,
        dtype=np.float32,
        output_multiple: int = 1,
    ):
        m = _integer(m, "m", 1)
        n = _integer(n, "n", 1)
        offset = _integer(offset, "offset", 0)
        _check(m, n, offset)
        vlen = _integer(vlen, "vlen", 1)
        output_multiple = _integer(output_multiple, "output_multiple", 1)
        dtype = np.dtype(dtype)
        if dtype.hasobject or dtype.itemsize == 0:
            raise ValueError("dtype must have a fixed, nonzero item size")

        self._m = m
        self._n = n
        self._offset = offset
        self._output_multiple = output_multiple
        self._lock = threading.Lock()

        io_dtype = (dtype, vlen) if vlen > 1 else dtype
        cuda.basic_block.__init__(self, "keep_m_in_n_cupy", [io_dtype], [io_dtype])
        self.set_output_multiple(math.lcm(m, output_multiple))
        self.set_relative_rate(m, n)

    def set_m(self, m):
        m = _integer(m, "m", 1)
        with self._lock:
            _check(m, self._n, self._offset)
            self._m = m
            self.set_output_multiple(math.lcm(m, self._output_multiple))
            self.set_relative_rate(m, self._n)

    def set_n(self, n):
        n = _integer(n, "n", 1)
        with self._lock:
            _check(self._m, n, self._offset)
            self._n = n
            self.set_relative_rate(self._m, n)

    def set_offset(self, offset):
        offset = _integer(offset, "offset", 0)
        with self._lock:
            _check(self._m, self._n, offset)
            self._offset = offset

    def forecast(self, noutput_items, ninputs):
        with self._lock:
            return [self._n * (noutput_items // self._m)] * ninputs

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        with self._lock:
            m, n, offset = self._m, self._n, self._offset

        blocks = min(len(out) // m, len(in0) // n)
        if blocks:
            src = in0[: blocks * n].reshape((blocks, n) + in0.shape[1:])
            dst = out[: blocks * m].reshape((blocks, m) + out.shape[1:])
            excess = offset + m - n
            if excess <= 0:
                dst[:] = src[:, offset : offset + m]
            else:
                dst[:, :excess] = src[:, :excess]
                dst[:, excess:] = src[:, offset:]

        self.consume_each(blocks * n)
        return blocks * m
