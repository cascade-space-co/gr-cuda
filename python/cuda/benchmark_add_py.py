#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import time
import numpy as np
from gnuradio import gr, blocks, cuda
try:
    import cupy as cp
except ImportError:
    print("CuPy not found, cannot benchmark")
    exit(1)

try:
    from add_py import add_py
except ImportError:
    try:
        from gnuradio.cuda.add_py import add_py
    except ImportError:
        # Fallback for local run
        import sys, os
        sys.path.append(os.path.dirname(__file__))
        from add_py import add_py

def benchmark(num_inputs=8, n_samples=10000000, vlen=1):
    print(f"Benchmarking add_py with {num_inputs} inputs, vlen={vlen}, {n_samples} samples...")
    
    tb = gr.top_block()
    
    # Generate random data (reuse same array to save RAM)
    # 10M complex64 = 80MB
    data = (np.random.randn(n_samples*vlen) + 1j * np.random.randn(n_samples*vlen)).astype(np.complex64)
    
    # Sources
    sources = []
    copies_to_dev = []
    
    # Create streams
    for i in range(num_inputs):
        # vector_source_c
        src = blocks.vector_source_c(data, False, vlen)
        sources.append(src)
        
        # cuda.copy (CPU -> GPU)
        # Allocates Device Memory for output, uses Host Memory for input
        to_dev = cuda.copy(np.dtype(np.complex64).itemsize * vlen)
        copies_to_dev.append(to_dev)
        
        tb.connect(src, to_dev)

    # DUT
    dut = add_py(num_inputs=num_inputs, dtype=np.complex64, vlen=vlen)
    
    # Sink path
    from_dev = cuda.copy(np.dtype(np.complex64).itemsize * vlen)
    snk = blocks.null_sink(np.dtype(np.complex64).itemsize * vlen)
    
    # Connect Inputs
    for i in range(num_inputs):
        tb.connect(copies_to_dev[i], (dut, i))
        
    # Connect Output
    tb.connect(dut, from_dev, snk)
    
    # Run
    start_time = time.time()
    tb.run()
    end_time = time.time()
    
    duration = end_time - start_time
    throughput = n_samples / duration
    
    print(f"Done in {duration:.4f}s")
    print(f"Throughput: {throughput/1e6:.2f} Msps (Vectors/sec)")
    print(f"Sample Rate: {(throughput*vlen)/1e6:.2f} Msps")
    
    # Total data throughput (in + out)
    # 8 streams in + 1 stream out = 9 streams * 8 bytes * rate
    data_rate_gbps = (throughput * vlen * 8 * (num_inputs + 1)) / 1e9
    print(f"Total Memory Bandwidth (est): {data_rate_gbps:.2f} GB/s")

if __name__ == '__main__':
    # Run with 8 inputs, 10M samples total (adjusted for vlen)
    # Try vlen=1024
    # Total samples = 10000 * 1024 = ~10M
    benchmark(num_inputs=8, n_samples=10000, vlen=1024)

