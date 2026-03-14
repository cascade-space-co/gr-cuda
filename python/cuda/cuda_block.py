#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""
Base classes for CuPy-based GPU blocks.

Drop-in replacements for ``gr.sync_block``, etc. that automatically:

1. Allocate CUDA buffers (via io_signature).
2. Register the block's CUDA stream with every connected ``cuda_buffer``.
   This tells the buffer which stream the block submits work on, so
   ``cuda_buffer`` can insert GPU-side event waits/records between
   producers and consumers.
3. Convert input/output arrays to CuPy (zero-copy).
4. Wrap ``start()`` overrides so stream registration is never skipped.

Auto-sync contract
------------------

Everything is handled transparently:

- Each block gets its own CUDA stream.
- ``work()`` / ``general_work()`` run inside that stream automatically.
- Stream registration and inter-block synchronization are invisible.
- Custom ``start()`` overrides are safe; registration still happens.

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
from gnuradio.gr.gateway import py_io_signature


def _wrap_work(fn):
    """Wrap ``work()`` / ``general_work()`` with CuPy stream and array conversion."""

    @functools.wraps(fn)
    def wrapped(self, input_items, output_items):
        with self.stream:
            cp_in = [cuda.as_cupy(x) for x in input_items]
            cp_out = [cuda.as_cupy(x) for x in output_items]
            return fn(self, cp_in, cp_out)

    wrapped._cuda_wrapped = True
    return wrapped


def _wrap_start(fn):
    """Ensure stream registration when a subclass defines its own ``start()``.

    Users are allowed to write a custom ``start()`` for their own setup.
    This wrapper runs that code first, then calls ``cuda_block.start()``
    to handle stream registration and GR base-class propagation.
    """

    @functools.wraps(fn)
    def wrapped(self):
        fn(self)
        return cuda_block.start(self)

    wrapped._cuda_start_wrapped = True
    return wrapped


class cuda_block:
    """Mixin that adds CUDA stream and synchronization to any GR block.

    Transparently wraps ``work()`` and ``general_work()`` so that the
    user receives CuPy arrays and never touches synchronization calls.

    ``in_sig`` / ``out_sig`` follow the same convention as GNU Radio's
    Python ``gateway_block``: pass a ``py_io_signature`` (typically built
    via ``cuda.io_signature_make``) to fully control min/max port counts,
    or pass a plain sequence of numpy dtypes for the common fixed-arity
    case (``min == max == len(dtype_list)``).

    Place ``cuda_block`` **before** the GR base in the inheritance list
    so that ``__init__`` chains correctly::

        class my_block(cuda.cuda_block, gr.basic_block):
            ...
    """

    def __init__(
        self,
        name: str,
        in_sig: py_io_signature | Sequence | None,
        out_sig: py_io_signature | Sequence | None,
        *args,
        **kwargs,
    ):
        in_sig = in_sig or ()
        out_sig = out_sig or ()
        if type(in_sig) is not py_io_signature:
            in_sig = cuda.io_signature_make(len(in_sig), len(in_sig), in_sig)
        if type(out_sig) is not py_io_signature:
            out_sig = cuda.io_signature_make(len(out_sig), len(out_sig), out_sig)
        super().__init__(name, in_sig, out_sig, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        """Auto-wrap work()/general_work()/start() at class definition time."""
        super().__init_subclass__(**kwargs)
        for name in ("work", "general_work"):
            if name in cls.__dict__:
                fn = cls.__dict__[name]
                if not getattr(fn, "_cuda_wrapped", False):
                    setattr(cls, name, _wrap_work(fn))
        if "start" in cls.__dict__:
            fn = cls.__dict__["start"]
            if not getattr(fn, "_cuda_start_wrapped", False):
                cls.start = _wrap_start(fn)

    def start(self):
        """Register CUDA stream with all cuda_buffers for auto-sync."""
        cuda.register_cuda_stream(self.gateway, self.stream.ptr)
        return super().start()

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
