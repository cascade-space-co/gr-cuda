#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from typing import Optional, Sequence

import numpy as np
import cupy as cp
from cupy.cuda import cufft as cp_cufft
from gnuradio import cuda


class fft_cupy(cuda.sync_block):
    """
    Performs FFT/IFFT on the GPU using CuPy.
    """
    def __init__(self,
                 fft_size: int,
                 forward: bool = True,
                 window: Optional[Sequence[float]] = None,
                 shift: bool = False,
                 real_input: bool = False):
        self.fft_size = fft_size
        self.forward = forward
        self.shift = shift
        self.real_input = bool(real_input)

        if self.real_input:
            in_dtype = (np.float32, fft_size)
        else:
            in_dtype = (np.complex64, fft_size)
        out_dtype = (np.complex64, fft_size)

        cuda.sync_block.__init__(self, "fft_cupy",
            [in_dtype],
            [out_dtype])

        self.d_window = None
        if window:
            with self.stream:
                self.d_window = cp.asarray(window, dtype=np.float32)

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

    def work(self, input_items, output_items):
        d_in = input_items[0]
        d_out = output_items[0]

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
            plan = self._get_plan(len(d_in))
            plan.fft(curr_in, d_out, self._direction)

        if self.forward and self.shift:
            d_out[:] = cp.fft.fftshift(d_out, axes=(-1,))

        return len(output_items[0])

    def _apply_window(self, d_in: cp.ndarray) -> cp.ndarray:
        if self.d_window is None:
            return d_in
        return d_in * self.d_window
