#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import operator
import threading

import cupy as cp
import numpy as np
from gnuradio import cuda


def _positive_integer(value, name):
    """Return an integer-like value after validating that it is positive."""
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _indices(indices, vlen):
    """Return ``indices`` as an int64 array after validating it against vlen."""
    try:
        indices = [operator.index(indices)]
    except TypeError:
        pass
    try:
        values = [operator.index(i) for i in indices]
    except TypeError:
        raise ValueError("indices must be a sequence of integers") from None
    if not values:
        raise ValueError("indices must not be empty")
    for i in values:
        if not 0 <= i < vlen:
            raise ValueError(f"index {i} is out of range for vlen {vlen}")
    return np.asarray(values, dtype=np.int64)


class vector_select_cupy(cuda.sync_block):
    """Select elements of each input vector on the GPU.

    Each input item is a vector of ``vlen`` values. Each output item holds
    ``input[indices[0]], input[indices[1]], ...`` in the order given, so one
    block can pick any channels of a multi-channel stream, reorder them, or
    repeat them. With a single index the output is a plain scalar stream.

    Parameters
    ----------
    vlen : int
        Length of each input vector. Must be positive.
    indices : int or sequence of int
        Elements to keep, each in ``[0, vlen)``. The output vector length is
        ``len(indices)``; a single integer is treated as a one-element list.
    dtype : numpy dtype, optional
        Scalar data type. Default is ``numpy.complex64``.
    output_multiple : int, optional
        Ask the scheduler for output batches in multiples of this size, which
        can amortize per-call overhead. Default is 1.
    """

    def __init__(
        self,
        vlen: int,
        indices,
        dtype=np.complex64,
        output_multiple: int = 1,
    ):
        vlen = _positive_integer(vlen, "vlen")
        host_indices = _indices(indices, vlen)
        output_multiple = _positive_integer(output_multiple, "output_multiple")
        dtype = np.dtype(dtype)
        if dtype.hasobject or dtype.itemsize == 0:
            raise ValueError("dtype must have a fixed, nonzero item size")

        self._vlen = vlen
        self._num_selected = host_indices.size
        self._lock = threading.Lock()

        in_dtype = (dtype, vlen) if vlen > 1 else dtype
        out_dtype = (dtype, self._num_selected) if self._num_selected > 1 else dtype
        cuda.sync_block.__init__(self, "vector_select_cupy", [in_dtype], [out_dtype])
        self.set_output_multiple(output_multiple)

        with self.stream:
            self._d_indices = cp.asarray(host_indices)

    def set_indices(self, indices):
        """Select different elements. The number of indices cannot change."""
        host_indices = _indices(indices, self._vlen)
        if host_indices.size != self._num_selected:
            raise ValueError(
                f"set_indices needs {self._num_selected} indices, "
                f"got {host_indices.size}"
            )
        with self.stream:
            d_indices = cp.asarray(host_indices)
        with self._lock:
            self._d_indices = d_indices

    def work(self, input_items, output_items):
        n = len(output_items[0])
        in0 = input_items[0].reshape(n, self._vlen)
        out = output_items[0].reshape(n, self._num_selected)
        with self._lock:
            d_indices = self._d_indices
        cp.take(in0, d_indices, axis=1, out=out)
        return n
