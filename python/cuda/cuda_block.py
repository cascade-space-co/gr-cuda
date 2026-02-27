#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""
Base classes for CuPy-based GPU blocks.

``cuda_block``
    A mixin that works with any GNU Radio block type.  Automatically
    wraps ``work()`` / ``general_work()`` so that:

    1. CUDA buffer synchronization is handled (wait_for_inputs / mark_outputs_ready).
    2. All input/output arrays are converted to CuPy (zero-copy).
    3. GPU operations run on the block's own CUDA stream.

    The user writes a normal ``work()`` — the only difference is that
    ``input_items`` and ``output_items`` contain CuPy arrays.

``gpu_work``
    Decorator that does the same wrapping explicitly.  Apply it to
    ``work()`` / ``general_work()`` when you want the sync + CuPy
    conversion to be visible at the call site, or when you need to
    opt out of the automatic ``__init_subclass__`` wrapping.

``sync_block``, ``decim_block``, ``interp_block``, ``basic_block``
    Shorthand bases for the corresponding ``gr.*`` block types.

Examples::

    # ---- Option A: implicit wrapping (auto via __init_subclass__) ----
    class my_multiply(cuda.sync_block):
        def __init__(self, k):
            self.k = k
            cuda.sync_block.__init__(self, "my_multiply",
                [np.complex64],      # 1 input  — auto-wrapped to CUDA sig
                [np.complex64])      # 1 output

        def work(self, input_items, output_items):
            output_items[0][:] = input_items[0] * self.k
            return len(output_items[0])

    # ---- Option B: explicit decorator ----
    class my_multiply(cuda.sync_block):
        def __init__(self, k):
            ...

        @cuda.gpu_work
        def work(self, input_items, output_items):
            output_items[0][:] = input_items[0] * self.k
            return len(output_items[0])

    # ---- any block type (mixin) ----
    class my_decim(cuda.cuda_block, gr.decim_block):
        def __init__(self, decim):
            cuda.cuda_block.__init__(self, "my_decim",
                [np.complex64], [np.complex64], decim)

        def work(self, input_items, output_items):
            ...
            return noutput_items
"""

import functools

try:
    import cupy as cp
except ImportError:
    cp = None

from gnuradio import gr
from gnuradio import cuda


# ======================================================================
# Standalone decorator — explicit alternative to __init_subclass__
# ======================================================================

def gpu_work(fn):
    """Wrap ``work()`` / ``general_work()`` with CUDA sync and CuPy conversion.

    Use this decorator when you want the wrapping to be visible at the
    call site rather than relying on the implicit ``__init_subclass__``
    mechanism in ``cuda_block``::

        class my_block(cuda.sync_block):
            @cuda.gpu_work
            def work(self, input_items, output_items):
                ...
    """
    @functools.wraps(fn)
    def wrapped(self, input_items, output_items):
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)
        with self.stream:
            cp_in = [cuda.as_cupy(x) for x in input_items]
            cp_out = [cuda.as_cupy(x) for x in output_items]
            result = fn(self, cp_in, cp_out)
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        return result
    wrapped._cuda_gpu_work = True
    return wrapped


# ======================================================================
# Mixin — works with any GR block type
# ======================================================================

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

    def __init__(self, name, in_sig, out_sig, *args, **kwargs):
        in_sig = self._auto_sig(in_sig)
        out_sig = self._auto_sig(out_sig)
        super().__init__(name, in_sig, out_sig, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method_name in ('work', 'general_work'):
            if method_name in cls.__dict__:
                original = cls.__dict__[method_name]
                if not getattr(original, '_cuda_gpu_work', False):
                    setattr(cls, method_name, gpu_work(original))

    @property
    def stream(self):
        """Non-blocking CuPy CUDA stream, created on first access."""
        try:
            return self._cuda_stream
        except AttributeError:
            if cp is None:
                raise ImportError("CuPy is required for CUDA blocks")
            self._cuda_stream = cp.cuda.Stream(non_blocking=True)
            return self._cuda_stream

    @staticmethod
    def _auto_sig(sig):
        """Wrap a plain dtype list into a CUDA io_signature."""
        if isinstance(sig, (list, tuple)):
            n = len(sig)
            return cuda.io_signature_make(n, n, sig)
        return sig


# ======================================================================
# Convenience bases for common block types
# ======================================================================

class sync_block(cuda_block, gr.sync_block):
    """Convenience base for CuPy sync blocks.

    Equivalent to ``class MyBlock(cuda.cuda_block, gr.sync_block)``.
    """
    pass


class decim_block(cuda_block, gr.decim_block):
    """Convenience base for CuPy decimation blocks.

    Equivalent to ``class MyBlock(cuda.cuda_block, gr.decim_block)``.
    """
    pass


class interp_block(cuda_block, gr.interp_block):
    """Convenience base for CuPy interpolation blocks.

    Equivalent to ``class MyBlock(cuda.cuda_block, gr.interp_block)``.
    """
    pass


class basic_block(cuda_block, gr.basic_block):
    """Convenience base for CuPy general blocks.

    Equivalent to ``class MyBlock(cuda.cuda_block, gr.basic_block)``.
    """
    pass
