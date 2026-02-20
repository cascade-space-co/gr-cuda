#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from typing import Optional, Sequence

import numpy as np
import cupy as cp
from cupy.cuda import cufft as cp_cufft
from gnuradio import gr, cuda

class fft_cupy(gr.sync_block):
    """Performs FFT/IFFT on the GPU using CuPy."""

    def __init__(self,
                 fft_size: int,
                 forward: bool = True,
                 window: Optional[Sequence[float]] = None,
                 shift: bool = False,
                 real_input: bool = False):
        """
        Parameters
        ----------
        fft_size : int
            Size of the FFT.
        forward : bool
            True for forward FFT, False for inverse.
        window : sequence of float, optional
            Window coefficients (applied as real-valued multiply).
        shift : bool
            Apply fftshift (forward) or ifftshift (inverse) around the transform.
        real_input : bool
            True for float32 input (float-to-complex FFT).
        """
        self.fft_size = fft_size
        self.forward = forward
        self.shift = shift
        self.real_input = bool(real_input)
        
        # IO Signature
        # Input is complex or float, output is always complex
        if self.real_input:
            in_dtype = [(np.float32, fft_size)]
        else:
            in_dtype = [(np.complex64, fft_size)]
        out_dtype = [(np.complex64, fft_size)]
        sig_in = cuda.io_signature_make(1, 1, in_dtype)
        sig_out = cuda.io_signature_make(1, 1, out_dtype)

        gr.sync_block.__init__(self, "fft_cupy", sig_in, sig_out)
        
        self.stream = cp.cuda.Stream(non_blocking=True)
        
        self.d_window = None
        if window:
            with self.stream:
                self.d_window = cp.asarray(window, dtype=np.float32)

        # Cached cuFFT plans (keyed by batch size) for C2C transforms.
        # Executes directly into the output buffer -- no temp allocation.
        self._plans = {}
        self._direction = (cp_cufft.CUFFT_FORWARD if forward
                           else cp_cufft.CUFFT_INVERSE)

    def _get_plan(self, batch_size: int) -> cp_cufft.Plan1d:
        """Get or create a cuFFT plan for the given batch size."""
        plan = self._plans.get(batch_size)
        if plan is None:
            plan = cp_cufft.Plan1d(self.fft_size, cp_cufft.CUFFT_C2C,
                                   batch_size)
            self._plans[batch_size] = plan
        return plan

    def work(self, input_items, output_items) -> int:
        n = len(input_items[0])
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)

        with self.stream:
            d_in = cuda.as_cupy(input_items[0])
            d_out = cuda.as_cupy(output_items[0])

            curr_in = self._apply_window(d_in)

            if not self.forward and self.shift:
                curr_in = cp.fft.ifftshift(curr_in, axes=(-1,))

            if self.real_input:
                if self.forward:
                    res = cp.fft.fft(curr_in, axis=-1)
                else:
                    res = cp.fft.ifft(curr_in, axis=-1)
                    res *= self.fft_size
                d_out[:] = res
            else:
                # C2C: direct cuFFT execution into output buffer.
                # cuFFT INVERSE is already unnormalized (matches GR).
                plan = self._get_plan(n)
                plan.fft(curr_in, d_out, self._direction)

            if self.forward and self.shift:
                d_out[:] = cp.fft.fftshift(d_out, axes=(-1,))

        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)

        return n

    def _apply_window(self, d_in: cp.ndarray) -> cp.ndarray:
        if self.d_window is None:
            return d_in
        return d_in * self.d_window

