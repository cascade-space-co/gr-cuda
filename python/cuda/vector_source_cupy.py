#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""GPU vector source (CuPy port of ``blocks.vector_source_x``)."""

import cupy as cp
import numpy as np
from gnuradio import cuda


class vector_source_cupy(cuda.sync_block):
    """Streams a fixed data vector from GPU memory using CuPy.

    GPU counterpart of ``blocks.vector_source_x``. The data vector is held in
    device memory and emitted in order, optionally repeating. Stream tags may
    be attached and are (re)emitted on every repetition, matching the in-tree
    block's semantics.
    """

    def __init__(
        self,
        data,
        repeat: bool = True,
        vlen: int = 1,
        tags=None,
        dtype: np.dtype = np.complex64,
    ):
        """
        Parameters
        ----------
        data : sequence
            Samples to emit. Length must be a multiple of ``vlen``.
        repeat : bool
            If True, loop over the data forever; otherwise stop when exhausted.
        vlen : int
            Vector length (items per output element).
        tags : list[gr.tag_t] or None
            Stream tags to attach, with offsets relative to the data vector.
        dtype : numpy.dtype
            Output sample type.
        """
        self._dtype = np.dtype(dtype)
        self._vlen = int(vlen)
        if self._vlen < 1:
            raise ValueError("vlen must be >= 1")

        host = np.asarray(data, dtype=self._dtype).ravel()
        if host.size % self._vlen != 0:
            raise ValueError(
                f"data length ({host.size}) must be a multiple of vlen ({self._vlen})"
            )

        self._data = cp.asarray(host)
        self._size = int(self._data.size)
        self._repeat = bool(repeat)
        self._offset = 0  # position within the data vector, in items
        self._tags = list(tags) if tags else []

        io_dtype = (self._dtype, self._vlen) if self._vlen > 1 else self._dtype
        cuda.sync_block.__init__(self, "vector_source_cupy", None, [io_dtype])

        # Match in-tree behavior: when tags are present, only emit whole copies
        # of the data vector per work() call so tag offsets stay aligned.
        if self._tags:
            self.set_output_multiple(self._size // self._vlen)

    def _emit_tags(self, base_item):
        """Emit configured tags at absolute output item ``base_item`` + offset."""
        for tag in self._tags:
            self.add_item_tag(0, base_item + tag.offset, tag.key, tag.value, tag.srcid)

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)  # number of output elements (vectors)
        if self._size == 0:
            return -1

        vlen = self._vlen
        written = self.nitems_written(0)

        if self._repeat:
            n_items = n * vlen
            size = self._size
            out_flat = out.reshape(-1)

            # Rotate the data so it starts at the current phase (only allocates
            # the small vector when not already phase-aligned).
            base = self._data
            if self._offset:
                base = cp.concatenate((base[self._offset :], base[: self._offset]))

            # Fill whole periods via a broadcast write straight into the output
            # buffer (no large temporary), then the partial tail.
            full, rem = divmod(n_items, size)
            if full:
                out_flat[: full * size].reshape(full, size)[:] = base
            if rem:
                out_flat[full * size :] = base[:rem]

            if self._tags:
                items_per_copy = size // vlen
                for i in range(0, n, items_per_copy):
                    self._emit_tags(written + i)

            self._offset = (self._offset + n_items) % size
            return n

        # Non-repeating: stop once the data vector is exhausted.
        if self._offset >= self._size:
            return -1

        n_items = min(self._size - self._offset, n * vlen)
        n_produced = n_items // vlen
        chunk = self._data[self._offset : self._offset + n_items]
        out[:n_produced] = chunk.reshape(n_produced, vlen) if vlen > 1 else chunk

        for tag in self._tags:
            if self._offset <= tag.offset < self._offset + n_items:
                self.add_item_tag(0, tag.offset, tag.key, tag.value, tag.srcid)

        self._offset += n_items
        return n_produced

    def rewind(self):
        self._offset = 0

    def set_data(self, data, tags=None):
        host = np.asarray(data, dtype=self._dtype).ravel()
        if host.size % self._vlen != 0:
            raise ValueError(
                f"data length ({host.size}) must be a multiple of vlen ({self._vlen})"
            )
        self._data = cp.asarray(host)
        self._size = int(self._data.size)
        self._tags = list(tags) if tags else []
        self.rewind()
