import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda
try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from .multiply_const_py import multiply_const_py
except ImportError:
    from multiply_const_py import multiply_const_py

class add_block_py(gr.sync_block):
    """
    A custom GPU block that adds two inputs together.
    Used to test synchronization when multiple streams merge (fan-in).
    """
    def __init__(self, dtype=np.complex64):
        if cp is None:
            raise ImportError("CuPy is required for add_block_py")
            
        self.dtype = np.dtype(dtype)
        
        # Create CUDA-aware IO signatures
        # Input: 2 ports
        sig_in = cuda.io_signature_make(2, 2, self.dtype)
        # Output: 1 port
        sig_out = cuda.io_signature_make(1, 1, self.dtype)

        gr.sync_block.__init__(self, "add_block_py", sig_in, sig_out)
        
        # Create a CUDA stream for this block
        self.stream = cp.cuda.Stream(non_blocking=True)

    def work(self, input_items, output_items):
        # Synchronization: Wait for ALL inputs to be ready on the GPU
        # This should handle waiting on multiple upstream streams if they differ
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)
        
        n = len(input_items[0])
        
        if n > 0:
            with self.stream:
                # Zero-copy wrap the input and output buffers
                d_in0 = cuda.as_cupy(input_items[0])
                d_in1 = cuda.as_cupy(input_items[1])
                d_out = cuda.as_cupy(output_items[0])
                
                # Perform the operation: out = in0 + in1
                cp.add(d_in0, d_in1, out=d_out)
            
        # Synchronization: Mark outputs as ready
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        
        return n

class qa_fanin_sync(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_fanin(self):
        """
        Test fan-in from multiple parallel GPU blocks to a single consumer.
        This verifies that the consumer correctly waits for multiple upstream streams.
        """
        if cp is None:
            print("Skipping test_001_fanin because CuPy is missing")
            return

        # Parameters
        N = 10000
        
        # Generate random complex data
        src_data = np.random.randn(N) + 1j * np.random.randn(N)
        src_data = src_data.astype(np.complex64)
        
        src = blocks.vector_source_c(src_data, False)
        
        # Move to GPU
        to_dev = cuda.copy(np.dtype(np.complex64).itemsize)
        
        # Branch 1: Multiply by 2.0
        # This will run on its own stream
        mult1 = multiply_const_py(2.0, dtype=np.complex64)
        
        # Branch 2: Multiply by 3.0
        # This will run on its own stream (different from mult1)
        mult2 = multiply_const_py(3.0, dtype=np.complex64)
        
        # Merge: Add branch 1 and branch 2
        # This runs on yet another stream and must wait for both mult1 and mult2
        add_blk = add_block_py(dtype=np.complex64)
        
        # Move from GPU
        from_dev = cuda.copy(np.dtype(np.complex64).itemsize)
        snk = blocks.vector_sink_c()
        
        # Connect
        # src -> to_dev -> mult1 -> add_blk -> from_dev -> snk
        #               -> mult2 -> (port 1 of add_blk)
        
        self.tb.connect(src, to_dev)
        
        self.tb.connect(to_dev, mult1)
        self.tb.connect(to_dev, mult2)
        
        self.tb.connect(mult1, (add_blk, 0))
        self.tb.connect(mult2, (add_blk, 1))
        
        self.tb.connect(add_blk, from_dev)
        self.tb.connect(from_dev, snk)
        
        # Run
        self.tb.run()
        
        # Verify
        result = np.array(snk.data(), dtype=np.complex64)
        expected = (src_data * 2.0) + (src_data * 3.0) # == src_data * 5.0
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    gr_unittest.run(qa_fanin_sync)

