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
    """Wrap ``work()`` with CuPy stream and array conversion.

    Used for the return-count path (sync/decim/interp blocks): the user
    returns a count and the scheduler calls ``produce_each()`` -> ``post_work()``
    *after* ``work()`` returns, so every kernel enqueued here is already on the
    stream when the device-ready event is recorded.  No produce/consume
    interception is needed.
    """

    @functools.wraps(fn)
    def wrapped(self, input_items, output_items):
        with self.stream:
            cp_in = [cuda.as_cupy(x) for x in input_items]
            cp_out = [cuda.as_cupy(x) for x in output_items]
            return fn(self, cp_in, cp_out)

    wrapped._cuda_wrapped = True
    return wrapped


# produce/consume calls intercepted and replayed after general_work() returns.
_DEFERRED_METHODS = ("produce", "consume", "consume_each")


def _wrap_general_work(fn):
    """Wrap ``general_work()`` with CuPy conversion, deferring produce/consume.

    Unlike the return-count path, ``produce()`` invokes ``post_work()``
    *synchronously*, which records the device-ready event immediately.  Any
    kernel the user enqueues after ``produce()``/``consume_each()`` would then
    be missed by that event, letting a downstream block read device memory
    before the write completes.

    As a backstop, ``produce()``, ``consume()`` and ``consume_each()`` are
    intercepted and their effects replayed only after the user's
    ``general_work()`` body returns; i.e. once every kernel is enqueued on the
    stream.  This advances the read/write pointers (and thus records
    device-ready) after the user's GPU work, not in the middle of it.

    This is a *safety net, not a feature*: the produce/consume ordering
    contract is identical to C++ (enqueue all stream work before you publish),
    and the C++ path has no such deferral.  Write blocks to the contract so
    they port cleanly; do not rely on calling produce/consume before enqueuing
    the work they publish.  See docs/LIMITATIONS.md.

    Consequence of the deferral: produce/consume side effects (e.g.
    ``nitems_written()``) are not observable *within* the same
    ``general_work()`` call -- they take effect when it returns -- so track
    running offsets in local variables rather than re-querying the framework.
    """

    @functools.wraps(fn)
    def wrapped(self, input_items, output_items):
        deferred = []
        originals = {name: getattr(self, name) for name in _DEFERRED_METHODS}

        def _make_recorder(target):
            def _record(*args, **kwargs):
                deferred.append((target, args, kwargs))

            return _record

        for name, target in originals.items():
            setattr(self, name, _make_recorder(target))

        try:
            with self.stream:
                cp_in = [cuda.as_cupy(x) for x in input_items]
                cp_out = [cuda.as_cupy(x) for x in output_items]
                result = fn(self, cp_in, cp_out)
        finally:
            # Restore class-method lookup (drop the instance shadows).
            for name in _DEFERRED_METHODS:
                delattr(self, name)

        # All kernels are now enqueued on self.stream; replaying produce/consume
        # here makes post_work() record device-ready after the user's work.
        for target, args, kwargs in deferred:
            target(*args, **kwargs)

        return result

    wrapped._cuda_wrapped = True
    return wrapped


def _wrap_start(fn):
    """Ensure stream registration when a subclass defines its own ``start()``.

    Users are allowed to write a custom ``start()`` for their own setup.
    This wrapper runs that code first, then ensures ``cuda_block.start()``
    (stream registration + GR base-class propagation) runs *exactly once*.

    A user start() may or may not call ``super().start()``.  If it does, the
    super() chain already routes through ``cuda_block.start`` (which sets
    ``_cuda_start_done``), so the wrapper must not call it a second time, or
    both stream registration and the GR base ``start()`` would fire twice.
    The per-call guard is reset on every entry so flowgraph restarts still
    re-register the stream.
    """

    @functools.wraps(fn)
    def wrapped(self):
        self._cuda_start_done = False
        result = fn(self)
        if not self._cuda_start_done:
            result = cuda_block.start(self)
        return result

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
        wrappers = {"work": _wrap_work, "general_work": _wrap_general_work}
        for name, wrap in wrappers.items():
            if name in cls.__dict__:
                fn = cls.__dict__[name]
                if not getattr(fn, "_cuda_wrapped", False):
                    setattr(cls, name, wrap(fn))
        if "start" in cls.__dict__:
            fn = cls.__dict__["start"]
            if not getattr(fn, "_cuda_start_wrapped", False):
                cls.start = _wrap_start(fn)

    def start(self):
        """Register CUDA stream with all cuda_buffers for auto-sync.

        Sets ``_cuda_start_done`` so a custom start() wrapped by
        ``_wrap_start`` can tell whether the super() chain already reached
        here and avoid invoking it a second time.
        """
        self._cuda_start_done = True
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
