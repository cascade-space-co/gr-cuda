#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import operator
import threading

import numpy as np
from gnuradio import cuda, gr


def _positive_integer(value, name):
    """Return an integer-like value after validating that it is positive."""
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class keep_one_in_n_cupy(cuda.basic_block):
    """Keep one input item out of every ``n`` on the GPU.

    This is the CUDA-buffer equivalent of ``blocks.keep_one_in_n``. It outputs
    input items ``n - 1``, ``2n - 1``, ... and keeps its position in the count
    across work calls, so any ``n`` works regardless of buffer size.
    ``set_n()`` restarts the count. Each input tag is propagated to output
    offset ``floor(offset / n)``.

    Parameters
    ----------
    n : int
        Keep one item out of every ``n``. Must be positive.
    vlen : int, optional
        Number of scalar values in each stream item. Default is 1.
    dtype : numpy dtype, optional
        Scalar data type. Default is ``numpy.float32``.
    output_multiple : int, optional
        Ask the scheduler for output batches in multiples of this size, which
        can amortize per-call overhead. Default is 1.
    """

    def __init__(
        self,
        n: int,
        vlen: int = 1,
        dtype=np.float32,
        output_multiple: int = 1,
    ):
        n = _positive_integer(n, "n")
        vlen = _positive_integer(vlen, "vlen")
        output_multiple = _positive_integer(output_multiple, "output_multiple")
        dtype = np.dtype(dtype)
        if dtype.hasobject or dtype.itemsize == 0:
            raise ValueError("dtype must have a fixed, nonzero item size")

        self._n = n
        self._count = n  # items left until the next kept item, inclusive
        self._lock = threading.Lock()

        io_dtype = (dtype, vlen) if vlen > 1 else dtype
        cuda.basic_block.__init__(self, "keep_one_in_n_cupy", [io_dtype], [io_dtype])
        self.set_output_multiple(output_multiple)
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def set_n(self, n):
        """Set ``n`` and restart the count."""
        n = _positive_integer(n, "n")
        with self._lock:
            self._n = n
            self._count = n

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        ninput_items = len(in0)
        noutput_items = len(out)

        with self._lock:
            n = self._n
            count = self._count

            if count > ninput_items:
                kept = 0
            else:
                kept = min((ninput_items - count) // n + 1, noutput_items)

            if kept == noutput_items:
                # Output is full: stop right after the last kept item.
                consumed = count + (kept - 1) * n
                self._count = n
            else:
                consumed = ninput_items
                self._count = count + kept * n - ninput_items

            if kept:
                first = count - 1
                out[:kept] = in0[first : first + (kept - 1) * n + 1 : n]

            start = self.nitems_read(0)
            for tag in self.get_tags_in_range(0, start, start + consumed):
                self.add_item_tag(0, tag.offset // n, tag.key, tag.value, tag.srcid)

        self.consume_each(consumed)
        return kept
