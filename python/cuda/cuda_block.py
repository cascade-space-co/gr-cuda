#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""
Base classes for CuPy-based GPU blocks.

Drop-in replacements for ``gr.sync_block``, etc. that automatically:

1. Allocate CUDA buffers (via io_signature).
2. Synchronize CUDA buffers before/after ``work()``.
3. Convert input/output arrays to CuPy (zero-copy).

Example::

    class my_multiply(cuda.sync_block):
        def __init__(self, k):
            self.k = k
            cuda.sync_block.__init__(self, "my_multiply",
                [np.complex64],
                [np.complex64])

        def work(self, input_items, output_items):
            output_items[0][:] = input_items[0] * self.k
            return len(output_items[0])
"""

import functools
from collections.abc import Sequence

import cupy as cp
from gnuradio import cuda, gr


def _wrap_work(fn):
    """Wrap ``work()`` / ``general_work()`` with CUDA sync and CuPy conversion.

    If the user calls ``self.produce()`` inside ``general_work()``,
    ``mark_work_done`` is triggered automatically before the first
    ``produce()`` call so that the device-ready event is recorded
    before the write pointer advances.
    """

    @functools.wraps(fn)
    def wrapped(self, input_items, output_items):
        # GPU-side waits: block until upstream data is ready and
        # downstream is done reading our previous output.
        cuda.wait_for_work(self.gateway, self.stream.ptr)

        # Intercept self.produce() so that mark_work_done is called
        # before produce() advances the write pointer.  Without this,
        # produce() triggers post_work() + update_write_pointer()
        # immediately, and for D2D buffers post_work is a no-op, so
        # the downstream block would see a stale device-ready event.
        self._cuda_work_done = False
        orig_produce = self.produce

        def _safe_produce(*args, **kwargs):
            if not self._cuda_work_done:
                cuda.mark_work_done(self.gateway, self.stream.ptr)
                self._cuda_work_done = True
            return orig_produce(*args, **kwargs)

        self.produce = _safe_produce

        try:
            with self.stream:
                cp_in = [cuda.as_cupy(x) for x in input_items]
                cp_out = [cuda.as_cupy(x) for x in output_items]
                result = fn(self, cp_in, cp_out)
        finally:
            # Always restore the original produce, even on exceptions.
            self.produce = orig_produce

        if not self._cuda_work_done:
            # User returned a count instead of calling produce() explicitly;
            # the block_executor will call produce_each() after we return.
            cuda.mark_work_done(self.gateway, self.stream.ptr)

        return result

    wrapped._cuda_wrapped = True
    return wrapped


class cuda_block:
    """Mixin that adds CUDA stream and synchronization to any GR block.

    Transparently wraps ``work()`` and ``general_work()`` so that the
    user receives CuPy arrays and never touches synchronization calls.

    Plain dtype lists passed as ``in_sig`` / ``out_sig`` are
    automatically converted to CUDA io signatures.

    Place ``cuda_block`` **before** the GR base in the inheritance list
    so that ``__init__`` chains correctly::

        class my_block(cuda.cuda_block, gr.basic_block):
            ...
    """

    def __init__(
        self,
        name: str,
        in_sig: Sequence | None,
        out_sig: Sequence | None,
        *args,
        **kwargs,
    ):
        if in_sig is not None:
            n = len(in_sig)
            in_sig = cuda.io_signature_make(n, n, in_sig)
        if out_sig is not None:
            n = len(out_sig)
            out_sig = cuda.io_signature_make(n, n, out_sig)
        super().__init__(name, in_sig, out_sig, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        """Auto-wrap work()/general_work() at class definition time."""
        super().__init_subclass__(**kwargs)
        for name in ("work", "general_work"):
            if name in cls.__dict__:
                fn = cls.__dict__[name]
                if not getattr(fn, "_cuda_wrapped", False):
                    setattr(cls, name, _wrap_work(fn))

    @property
    def stream(self):
        """Non-blocking CuPy CUDA stream, created on first access."""
        try:
            return self._cuda_stream
        except AttributeError:
            self._cuda_stream = cp.cuda.Stream(non_blocking=True)
            return self._cuda_stream


class sync_block(cuda_block, gr.sync_block):
    pass


class decim_block(cuda_block, gr.decim_block):
    pass


class interp_block(cuda_block, gr.interp_block):
    pass


class basic_block(cuda_block, gr.basic_block):
    pass
