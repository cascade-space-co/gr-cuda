#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from gnuradio import gr, cuda

class fft_cupy(gr.sync_block):
    """
    fft_cupy
    
    Performs FFT/IFFT on the GPU using CuPy.
    """
    def __init__(self,
                 fft_size,
                 forward=True,
                 window=None,
                 shift=False,
                 nthreads=1,
                 real_input=False):
        """
        Args:
            fft_size: Size of the FFT
            forward: True for forward FFT, False for inverse
            window: Optional window function
            shift: Apply fftshift
            nthreads: Ignored (for compatibility)
            real_input: True for float32 input (float-to-complex FFT)
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
        
        if cp is None:
            raise ImportError("CuPy is required for fft_cupy")
            
        self.stream = cp.cuda.Stream(non_blocking=True)
        
        self.d_window = None
        if window:
            with self.stream:
                window_dtype = np.float32 if self.real_input else np.complex64
                self.d_window = cp.asarray(window, dtype=window_dtype)

    def work(self, input_items, output_items):
        n = len(input_items[0])
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)
        
        if n > 0:
            with self.stream:
                d_in = cuda.as_cupy(input_items[0])
                d_out = cuda.as_cupy(output_items[0])

                curr_in = self._apply_window(d_in)

                if self.forward:
                    # cp.fft.fft computes along last axis by default (axis=-1)
                    res = cp.fft.fft(curr_in, axis=-1)
                    if self.shift:
                        res = cp.fft.fftshift(res, axes=(-1,))
                else:
                    if self.shift:
                        curr_in = cp.fft.ifftshift(curr_in, axes=(-1,))
                    res = cp.fft.ifft(curr_in, axis=-1)
                    # Match GNU Radio FFT unnormalized inverse behavior.
                    res *= self.fft_size
                    
                # Copy result to output buffer
                d_out[:] = res
            
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        
        return n

    def _apply_window(self, d_in):
        if self.d_window is None:
            return d_in
        # Broadcast window multiplication without touching the input buffer.
        return d_in * self.d_window

