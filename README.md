# gr-cuda

GPU-accelerated signal processing blocks for [GNU Radio](https://www.gnuradio.org/) 3.10+. Data moves to/from the GPU automatically; blocks receive device pointers directly in `work()`, no manual `cudaMemcpy` needed.

<img src="docs/img/gr-cuda_example.png" alt="Example flowgraph">

## Key features

- **Zero-copy GPU pipeline**: `cuda_buffer` handles host-device transfers and CUDA-event synchronisation between blocks. GPU kernels run back-to-back without round-tripping through the CPU.
- **Write GPU blocks in Python**: inherit from `cuda.sync_block`, get CuPy arrays in `work()`, done. Ten lines of Python for a new GPU block (see [example below](#pythoncupy-block)). Performance is within 5% of hand-written CUDA C++ (see [benchmarks](#performance)).
- **C++ GPU blocks**: `cuda_buffer` takes care of memory transfers, buffer management, and keeping the GPU fed; you just write the kernel (see [C++ block](#c-block)).
- **Virtual memory management**: `cuda_buffer` uses CUDA VMM to implement circular buffers on the device, mirroring the same trick GNU Radio uses on the CPU side.
- **Batteries included**: gr-cuda ships with building-block primitives to get you started (probe rate, GPU null source/sink, stream/vector operators, throttle, and more). All blocks include GRC definitions for drag-and-drop use.

## Quick start

### Prerequisites

- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 10.2
- GNU Radio >= 3.10 (>3.10.12 recommended; older versions have a [fan-out deadlock bug](https://github.com/gnuradio/gnuradio/pull/8029))
- CMake >= 3.18, Ninja (recommended)
- [CuPy](https://cupy.dev/) (for Python GPU blocks)

### Build and install

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -Wno-dev
ninja && ninja install && ninja test
```

For conda environments, add `-DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"`.

> The build auto-detects your GPU architecture. To override: `-DCMAKE_CUDA_ARCHITECTURES=86`.

## Writing GPU blocks

### Python/CuPy block

Simply inherit from `cuda.sync_block` instead of `gr.sync_block`. Input/output items are CuPy arrays on the GPU -- use any `cp.*` operation:

```python
import cupy as cp
import numpy as np
from gnuradio import cuda

class my_cupy_block(cuda.sync_block):   # cuda.sync_block instead of gr.sync_block
    def __init__(self):
        cuda.sync_block.__init__(self, "my_cupy_block",
            [np.complex64], [np.complex64])

    def work(self, input_items, output_items):
        # input_items / output_items are CuPy arrays on the GPU
        cp.multiply(input_items[0], 2.0, out=output_items[0])
        return len(output_items[0])
```

`cuda.decim_block`, `cuda.interp_block`, and `cuda.basic_block` are also available. See [`multiply_const_cupy.py`](python/cuda/multiply_const_cupy.py) for a complete example.

### C++ block

Inherit from `cuda_block` to get a managed CUDA stream and launch your own kernels. Synchronisation between blocks is currently explicit (`wait_for_work`/`mark_work_done`) but will be automated in a future release.

```cpp
int my_block_impl::work(int noutput_items, ...)
{
    gr::cuda::wait_for_work(detail(), d_stream);

    auto in  = static_cast<const float*>(input_items[0]);
    auto out = static_cast<float*>(output_items[0]);
    my_kernel<<<grid, block, 0, d_stream>>>(in, out, noutput_items);

    gr::cuda::mark_work_done(detail(), d_stream);
    return noutput_items;
}
```

See [`cuda_block.h`](include/gnuradio/cuda/cuda_block.h) and [`multiply_const_impl.cc`](lib/multiply_const_impl.cc) for a full example.

> Want to use gr-cuda blocks in your own out-of-tree module? See **[docs/OOT_INTEGRATION.md](docs/OOT_INTEGRATION.md)**.

## Performance

Benchmarked on an **NVIDIA DGX Spark (GB10)** and an **NVIDIA RTX PRO 6000 Blackwell** (PCIe 5.0 x16).

| Benchmark | GB10 | RTX PRO 6000 |
|-----------|-----:|-------------:|
| H2D transfer | 58.3 GB/s | 54.1 GB/s |
| D2H transfer | 59.1 GB/s | 56.4 GB/s |
| Full round-trip | 29.2 GB/s | 38.7 GB/s |
| cuFFT C64→C64 (C++) | 13.9 Gsps | 92.8 Gsps |
| CuPy FFT C64→C64 (Python) | 13.5 Gsps | 88.3 Gsps |
| FFTW C64→C64 (CPU baseline) | 0.45 Gsps | 1.01 Gsps |

CuPy blocks track within 5% of C++ CUDA blocks -- write your GPU blocks in Python with no meaningful performance penalty.

See **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** for more details and **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** for known limitations.

## Acknowledgements

This OOT is adapted from [gr-cuda_buffer](https://github.com/BlackLynx-Inc/gr-cuda_buffer) by Black Lynx, Inc. and relies on the custom buffer feature developed by David Sorber ([Custom Buffers wiki](https://wiki.gnuradio.org/index.php/CustomBuffers)). Further developed by Josh Morman and heavily modified and extended by Wael Farah & Brett Gottula at Cascade Space.

## License

GPL-3.0-or-later. See individual file headers for copyright details.
