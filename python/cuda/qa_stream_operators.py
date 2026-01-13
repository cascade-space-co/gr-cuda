import numpy as np
from gnuradio import gr, gr_unittest, blocks
from gnuradio import cuda

class qa_conversions(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_stream_vector_chain(self):
        """
        Test: stream -> vector -> stream -> vector -> stream (all on GPU)
        Using a loop to create a longer chain.
        """
        N = 1000000
        vlen = 16
        chain_length = 10 # Number of S2V -> V2S pairs
        
        # Ensure N is multiple of vlen
        N = (N // vlen) * vlen
        
        src_data = np.random.randn(N).astype(np.float32)
        src = blocks.vector_source_f(src_data, False)
        
        # Chain construction
        # [S2V] -> [V2S] -> [S2V] -> [V2S] ...
        
        blocks_chain = []
        for _ in range(chain_length):
            s2v = cuda.stream_to_vector(4, vlen)
            v2s = cuda.vector_to_stream(4, vlen)
            blocks_chain.append(s2v)
            blocks_chain.append(v2s)
            
        snk = blocks.vector_sink_f()
        
        # Connect: src -> chain[0] -> ... -> chain[end] -> snk
        # Implicit copies should handle CPU <-> GPU transitions
        
        if blocks_chain:
            self.tb.connect(src, blocks_chain[0])
            for i in range(len(blocks_chain) - 1):
                self.tb.connect(blocks_chain[i], blocks_chain[i+1])
            self.tb.connect(blocks_chain[-1], snk)
        else:
            self.tb.connect(src, snk)
            
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data)

    def test_002_interleave_deinterleave_chain(self):
        """
        Test: streams -> vector -> streams -> vector -> streams ...
        Chain of Interleave/Deinterleave operations.
        """
        N = 1000000
        num_streams = 2
        chain_length = 5 # Number of (Interleave -> Deinterleave) stages
        
        # Two input streams
        src_data1 = np.random.randint(0, 100, N).astype(np.int32)
        src_data2 = np.random.randint(0, 100, N).astype(np.int32)
        
        src1 = blocks.vector_source_i(src_data1, False)
        src2 = blocks.vector_source_i(src_data2, False)
        
        # Construct chain
        # Stage i: [Interleave] -> [Deinterleave]
        
        interleaves = []
        deinterleaves = []
        
        for _ in range(chain_length):
            intl = cuda.streams_to_vector(4, num_streams)
            deintl = cuda.vector_to_streams(4, num_streams)
            interleaves.append(intl)
            deinterleaves.append(deintl)
            
        snk1 = blocks.vector_sink_i()
        snk2 = blocks.vector_sink_i()
        
        # Initial connection: src -> first Interleave (Implicit Copy)
        self.tb.connect(src1, (interleaves[0], 0))
        self.tb.connect(src2, (interleaves[0], 1))
        
        for i in range(chain_length):
            # Interleave -> Deinterleave (Single vector connection)
            self.tb.connect(interleaves[i], deinterleaves[i])
            
            # If not the last stage, connect Deinterleave output -> Next Interleave input
            if i < chain_length - 1:
                # Connect all N streams
                for k in range(num_streams):
                    self.tb.connect((deinterleaves[i], k), (interleaves[i+1], k))
                    
        # Final output: Last Deinterleave -> Sink (Implicit Copy)
        self.tb.connect((deinterleaves[-1], 0), snk1)
        self.tb.connect((deinterleaves[-1], 1), snk2)
        
        self.tb.run()
        
        res1 = np.array(snk1.data(), dtype=np.int32)
        res2 = np.array(snk2.data(), dtype=np.int32)
        
        np.testing.assert_array_equal(res1, src_data1)
        np.testing.assert_array_equal(res2, src_data2)

    def test_003_mixed_chain(self):
        """
        Test: Stream -> Stream2Vector(vlen=4) -> Vector2Streams(4 streams) -> Streams2Vector(4 streams) -> Vector2Stream(vlen=4) -> Stream
        Essentially reshaping and shuffling.
        """
        N = 1000000
        # Total items must be divisible by 4
        N = (N // 4) * 4
        
        src_data = np.arange(N, dtype=np.float32)
        src = blocks.vector_source_f(src_data, False)
        
        # 1. Stream -> Vector (vlen=4)
        s2v = cuda.stream_to_vector(4, 4)
        
        # 2. Vector -> Streams (4 streams) (Deinterleave)
        v2ss = cuda.vector_to_streams(4, 4)
        
        # 3. Streams -> Vector (4 streams) (Interleave)
        ss2v = cuda.streams_to_vector(4, 4)
        
        # 4. Vector -> Stream
        v2s = cuda.vector_to_stream(4, 4)
        
        snk = blocks.vector_sink_f()
        
        # Implicit copies
        self.tb.connect(src, s2v)
        self.tb.connect(s2v, v2ss)
        
        # Connect all 4 parallel paths
        for i in range(4):
            self.tb.connect((v2ss, i), (ss2v, i))
            
        self.tb.connect(ss2v, v2s)
        self.tb.connect(v2s, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data)

    def test_004_vector_interleave(self):
        """
        Test interleaving vectors instead of scalars.
        vlen=4, num_streams=2.
        Input is streams of vectors (vlen=4).
        Output is stream of larger vectors (vlen=8).
        """
        N = 1000000 # items per stream
        vlen = 4
        num_streams = 2
        # Ensure N is multiple of vlen
        N = (N // vlen) * vlen
        
        src_data1 = np.random.randint(0, 100, N).astype(np.int32)
        src_data2 = np.random.randint(0, 100, N).astype(np.int32)
        
        # Sources produce vectors of length 4
        src1 = blocks.vector_source_i(src_data1, False, vlen)
        src2 = blocks.vector_source_i(src_data2, False, vlen)
        
        # Interleave blocks of 4 integers
        # itemsize passed to C++ is 4 * 4 = 16 bytes
        itemsize_bytes = 4 * vlen
        
        intl = cuda.streams_to_vector(itemsize_bytes, num_streams)
        deintl = cuda.vector_to_streams(itemsize_bytes, num_streams)
        
        snk1 = blocks.vector_sink_i(vlen)
        snk2 = blocks.vector_sink_i(vlen)
        
        # src1 (vec4) -> intl port 0
        # src2 (vec4) -> intl port 1
        self.tb.connect(src1, (intl, 0))
        self.tb.connect(src2, (intl, 1))
        
        self.tb.connect(intl, deintl)
        
        self.tb.connect((deintl, 0), snk1)
        self.tb.connect((deintl, 1), snk2)
        
        self.tb.run()
        
        res1 = np.array(snk1.data(), dtype=np.int32)
        res2 = np.array(snk2.data(), dtype=np.int32)
        
        np.testing.assert_array_equal(res1, src_data1)
        np.testing.assert_array_equal(res2, src_data2)

    def test_005_stream_vector_blocking(self):
        """
        Test Stream to Vector with input vector length > 1.
        (Block vectors into larger vectors).
        Input: Stream of Vec2.
        S2V (num_items=3): Consumes 3 input vectors -> Produces 1 output vector (Vec6).
        Output: Stream of Vec6.
        """
        N = 1200000 # Total items (Large enough to trigger multiple buffers)
        input_vlen = 2
        blocking_factor = 3 # num_items
        
        # Total output items = N
        # Output vector length = 2 * 3 = 6
        
        src_data = np.arange(N, dtype=np.float32)
        
        # Source produces Vec2
        src = blocks.vector_source_f(src_data, False, input_vlen)
        
        # S2V: Input Vec2, Block 3 of them -> Output Vec6
        # itemsize passed to C++ = sizeof(float) * input_vlen = 8 bytes
        # nitems passed to C++ = blocking_factor = 3
        s2v = cuda.stream_to_vector(4 * input_vlen, blocking_factor)
        
        # V2S: Input Vec6 -> Output Vec2
        # itemsize passed to C++ = sizeof(float) * input_vlen = 8 bytes
        # nitems passed to C++ = blocking_factor = 3
        v2s = cuda.vector_to_stream(4 * input_vlen, blocking_factor)
        
        snk = blocks.vector_sink_f(input_vlen)
        
        self.tb.connect(src, s2v)
        self.tb.connect(s2v, v2s)
        self.tb.connect(v2s, snk)
        
        self.tb.run()
        
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data)

if __name__ == '__main__':
    gr_unittest.run(qa_conversions)
