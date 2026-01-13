
import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda
try:
    from .multiply_const_py import multiply_const_py
except ImportError:
    from multiply_const_py import multiply_const_py

class qa_multiply_const_py(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_complex_multiply(self):
        # Parameters
        N = 1000
        k = 2.0 + 1.0j
        
        # Generate data
        src_data = np.random.randn(N) + 1j * np.random.randn(N)
        src_data = src_data.astype(np.complex64)
        
        src = blocks.vector_source_c(src_data, False)
        
        # DUT
        dut = multiply_const_py(k, dtype=np.complex64)
        
        snk = blocks.vector_sink_c()
        
        # Connect: src -> dut -> snk (Implicit copies)
        self.tb.connect(src, dut, snk)
        self.tb.run()
        
        result = snk.data()
        expected = src_data * k
        
        self.assertComplexTuplesAlmostEqual(result, expected, 5)

    def test_002_float_multiply(self):
        # Parameters
        N = 1000
        k = 2.5
        
        # Generate data
        src_data = np.random.randn(N).astype(np.float32)
        
        src = blocks.vector_source_f(src_data, False)
        
        # DUT
        dut = multiply_const_py(k, dtype=np.float32)
        
        snk = blocks.vector_sink_f()
        
        # Connect: src -> dut -> snk (Implicit copies)
        self.tb.connect(src, dut, snk)
        self.tb.run()
        
        result = snk.data()
        expected = src_data * k
        
        self.assertFloatTuplesAlmostEqual(result, expected, 5)

if __name__ == '__main__':
    gr_unittest.run(qa_multiply_const_py)
