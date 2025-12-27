import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from gnuradio import gr, cuda

class add_py(gr.sync_block):
    """
    add_py
    
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
        
        gr.sync_block.__init__(self, "add_py", input_sig, output_sig)
        
        if cp is None:
            raise ImportError("CuPy is required for add_py")
            
        self.stream = cp.cuda.Stream(non_blocking=True)

    def work(self, input_items, output_items):
        # Synchronization
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)
        
        n_out = len(output_items[0])
        
        if n_out > 0:
            with self.stream:
                # Get output buffer
                d_out = cuda.as_cupy(output_items[0])
                
                # Get first input and copy to output (or just wrap it)
                # We can do out = in0 + in1 + ...
                
                # Optimization: copy first input to output, then add others in place
                d_in0 = cuda.as_cupy(input_items[0])
                cp.copyto(d_out, d_in0)
                
                # Add remaining inputs
                for i in range(1, self.num_inputs):
                    d_in = cuda.as_cupy(input_items[i])
                    # In-place add: out += in
                    d_out += d_in
                    
        # Synchronization
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        
        return n_out

