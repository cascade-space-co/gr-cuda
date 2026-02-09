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

class multiply_const_py(gr.sync_block):
    """
    multiply_const_py
    
    Multiplies input stream by a constant value (k) on the GPU using CuPy.
    """
    def __init__(self, k, dtype=np.complex64, vlen=1):
        """
        Args:
            k: Constant to multiply by
            dtype: Data type (numpy dtype)
            vlen: Vector length
        """
        self.k = k
        self.dtype = np.dtype(dtype)
        self.vlen = vlen
        
        # Calculate IO signature
        if vlen > 1:
            io_dtype = [(self.dtype, vlen)]
        else:
            io_dtype = [self.dtype]

        # Create CUDA-aware IO signature
        # We need to pass a list of types if we want specific ports, 
        # or a single type for all ports. Here we have 1 in, 1 out.
        sig = cuda.io_signature_make(1, 1, io_dtype)

        gr.sync_block.__init__(self, "multiply_const_py", sig, sig)
        
        if cp is None:
            raise ImportError("CuPy is required for multiply_const_py")
            
        # Create a CUDA stream for this block to allow overlap
        self.stream = cp.cuda.Stream(non_blocking=True)

    def work(self, input_items, output_items):
        # Synchronization: Wait for inputs to be ready on the GPU
        # self.gateway is the underlying C++ block object needed by the helper
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)
        
        n = len(input_items[0])
        
        if n > 0:
            with self.stream:
                # Zero-copy wrap the input and output buffers
                # input_items[0] is a numpy array pointing to GPU memory
                d_in = cuda.as_cupy(input_items[0])
                d_out = cuda.as_cupy(output_items[0])
                
                # Perform the operation
                # We use out=d_out to ensure we write directly to the output buffer
                cp.multiply(d_in, self.k, out=d_out)
            
        # Synchronization: Mark outputs as ready so downstream blocks know
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        
        return n

    def set_k(self, k):
        self.k = k

