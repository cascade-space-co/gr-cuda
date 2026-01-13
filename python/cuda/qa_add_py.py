import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda
try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from .add_py import add_py
except ImportError:
    from add_py import add_py

class qa_add_py(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_add_2_streams(self):
        if cp is None:
            return

        N = 1000
        src_data1 = np.random.randn(N).astype(np.float32)
        src_data2 = np.random.randn(N).astype(np.float32)
        
        src1 = blocks.vector_source_f(src_data1, False)
        src2 = blocks.vector_source_f(src_data2, False)
        
        dut = add_py(num_inputs=2, dtype=np.float32)
        
        snk = blocks.vector_sink_f()
        
        # Connect: src -> dut -> snk (Implicit copies)
        self.tb.connect(src1, (dut, 0))
        self.tb.connect(src2, (dut, 1))
        self.tb.connect(dut, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.float32)
        expected = src_data1 + src_data2
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_002_add_complex(self):
        if cp is None:
            return

        N = 1000
        src_data1 = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
        src_data2 = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
        src_data3 = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
        
        src1 = blocks.vector_source_c(src_data1, False)
        src2 = blocks.vector_source_c(src_data2, False)
        src3 = blocks.vector_source_c(src_data3, False)
        
        dut = add_py(num_inputs=3, dtype=np.complex64)
        
        snk = blocks.vector_sink_c()
        
        # Connect: src -> dut -> snk (Implicit copies)
        self.tb.connect(src1, (dut, 0))
        self.tb.connect(src2, (dut, 1))
        self.tb.connect(src3, (dut, 2))
        
        self.tb.connect(dut, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.complex64)
        expected = src_data1 + src_data2 + src_data3
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    gr_unittest.run(qa_add_py)
