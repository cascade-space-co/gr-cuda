#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import cupy as cp
from gnuradio import gr, cuda

class add_cupy(gr.sync_block):
    """
    add_cupy
    
    Adds N input streams together on the GPU using CuPy.
    """
    def __init__(self, num_inputs=2, dtype=np.complex64, vlen=1):
        """
        Args:
            num_inputs: Number of input streams to add
            dtype: Data type (numpy dtype)
            vlen: Vector length
        """
        self.num_inputs = num_inputs
        self.dtype = np.dtype(dtype)
        self.vlen = vlen
        
        # IO Signature
        if vlen > 1:
            io_dtype = [(self.dtype, vlen)]
        else:
            io_dtype = [self.dtype]

        # num_inputs IN, 1 OUT
        input_sig = cuda.io_signature_make(num_inputs, num_inputs, io_dtype)
        output_sig = cuda.io_signature_make(1, 1, io_dtype)
        
        gr.sync_block.__init__(self, "add_cupy", input_sig, output_sig)
        
        self.stream = cp.cuda.Stream(non_blocking=True)

    def work(self, input_items, output_items):
        # Synchronization
        cuda.wait_for_work(self.gateway, self.stream.ptr)
        
        n_out = len(output_items[0])
        
        if n_out > 0:
            with self.stream:
                # Get output buffer
                d_out = cuda.as_cupy(output_items[0])
                
                # Wrap all input buffers as CuPy arrays
                d_inputs = [cuda.as_cupy(input_items[i]) for i in range(self.num_inputs)]
                
                # Sum all inputs - CuPy will fuse this into an efficient kernel
                d_out[:] = sum(d_inputs)
                    
        # Synchronization
        cuda.mark_work_done(self.gateway, self.stream.ptr)
        
        return n_out

