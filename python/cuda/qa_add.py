import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda

class qa_add(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_add_2_streams(self):
        N = 1000
        src_data1 = np.random.randn(N).astype(np.float32)
        src_data2 = np.random.randn(N).astype(np.float32)
        
        src1 = blocks.vector_source_f(src_data1, False)
        src2 = blocks.vector_source_f(src_data2, False)
        
        # Use new C++ block
        dut = cuda.add_ff(num_inputs=2)
        
        snk = blocks.vector_sink_f()
        
        # Connect src -> dut directly (auto copy host->device)
        self.tb.connect(src1, (dut, 0))
        self.tb.connect(src2, (dut, 1))
        
        # Connect dut -> snk directly (auto copy device->host)
        self.tb.connect(dut, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.float32)
        expected = src_data1 + src_data2
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_002_add_complex(self):
        N = 1000
        src_data1 = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
        src_data2 = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
        src_data3 = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
        
        src1 = blocks.vector_source_c(src_data1, False)
        src2 = blocks.vector_source_c(src_data2, False)
        src3 = blocks.vector_source_c(src_data3, False)
        
        # Use new C++ block
        dut = cuda.add_cc(num_inputs=3)
        
        snk = blocks.vector_sink_c()
        
        # Connect src -> dut directly
        self.tb.connect(src1, (dut, 0))
        self.tb.connect(src2, (dut, 1))
        self.tb.connect(src3, (dut, 2))
        
        # Connect dut -> snk directly
        self.tb.connect(dut, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.complex64)
        expected = src_data1 + src_data2 + src_data3
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_003_add_int(self):
        N = 1000
        # Use integers. Keep range small enough to avoid overflow when adding
        src_data1 = np.random.randint(-100, 100, N).astype(np.int32)
        src_data2 = np.random.randint(-100, 100, N).astype(np.int32)
        
        src1 = blocks.vector_source_i(src_data1, False)
        src2 = blocks.vector_source_i(src_data2, False)
        
        # Use new C++ block (add_ii for int32)
        dut = cuda.add_ii(num_inputs=2)
        
        snk = blocks.vector_sink_i()
        
        # Connect src -> dut directly
        self.tb.connect(src1, (dut, 0))
        self.tb.connect(src2, (dut, 1))
        
        # Connect dut -> snk directly
        self.tb.connect(dut, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.int32)
        expected = src_data1 + src_data2
        
        # Exact match for integers
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    gr_unittest.run(qa_add)
