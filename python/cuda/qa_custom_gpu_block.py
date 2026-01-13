import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda
import cupy as cp

class my_gpu_block(gr.sync_block):
    def __init__(self):
        # Using the helper function to create CUDA IO signatures
        sig_in = cuda.io_signature_make(1, 1, np.complex64)
        sig_out = cuda.io_signature_make(1, 1, np.complex64)
        
        gr.sync_block.__init__(self, "my_gpu_block", 
            in_sig=sig_in,
            out_sig=sig_out)
        self.stream = cp.cuda.Stream(non_blocking=True)

    def work(self, input_items, output_items):
        # Must pass self.gateway (the underlying C++ block) to helpers
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)
        with self.stream:
            d_in = cuda.as_cupy(input_items[0])
            d_out = cuda.as_cupy(output_items[0])
            cp.multiply(d_in, 2.0, out=d_out)
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        return len(input_items[0])

class qa_custom_gpu_block(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001(self):
        src_data = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
        src = blocks.vector_source_c(src_data, False)
        
        dut = my_gpu_block()
        
        snk = blocks.vector_sink_c()
        
        # Connect: src -> dut -> snk (Implicit copies)
        self.tb.connect(src, dut, snk)
        self.tb.run()
        
        result = snk.data()
        expected = src_data * 2.0
        
        self.assertComplexTuplesAlmostEqual(result, expected, 5)

if __name__ == '__main__':
    gr_unittest.run(qa_custom_gpu_block)
