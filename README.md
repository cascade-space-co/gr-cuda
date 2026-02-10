# gr-cuda

GPU-accelerated signal processing blocks for [GNU Radio](https://www.gnuradio.org/) 3.10+, built on the custom buffer API introduced by David Sorber. Data moves to/from the GPU automatically via `cuda_buffer`, so blocks receive device pointers directly in their `work()` function -- no manual `cudaMemcpy` needed.

<img title="Simple flowgraph with CUDA block" alt="Simple CUDA flowgraph" src="docs/img/flowgraph_copy.png">

## Features

- **Zero-copy GPU pipeline** -- `cuda_buffer` handles H2D/D2H transfers and event-based synchronisation between blocks, so GPU kernels run back-to-back without stalling. Internally it uses a double-buffered scheme so the CPU can fill one half while the GPU processes the other.
- **C++ blocks** with hand-written CUDA kernels (cuFFT, custom element-wise ops).
- **Python/CuPy blocks** that run entirely on the GPU using CuPy arrays -- write new GPU blocks in pure Python. Carefully written CuPy blocks achieve throughput within 1-2% of equivalent C++ CUDA blocks (see [Performance](#performance)).
- **GRC support** -- all blocks ship with GNU Radio Companion block definitions.

## Included blocks

### C++ (CUDA kernel)

| Block | Description |
|-------|-------------|
| `cuda.add` | Adds N input streams element-wise |
| `cuda.multiply_const` | Multiplies input stream by a scalar constant |
| `cuda.fft` | Forward/inverse FFT via cuFFT, with optional windowing and shift |
| `cuda.fft_shift` | Applies `fftshift` on complex vectors |
| `cuda.copy` | Copies input to output (passthrough mode available for benchmarking) |
| `cuda.stream_to_vector` | Packs a stream into fixed-size vectors |
| `cuda.vector_to_stream` | Unpacks vectors back into a stream |
| `cuda.streams_to_vector` | Interleaves N streams into one vector stream |
| `cuda.vector_to_streams` | De-interleaves one vector stream into N streams |
| `cuda.throttle` | Rate-limits the GPU pipeline to a target sample rate |
| `cuda.null_source` | Generates zeros directly on the GPU (no H2D transfer) |
| `cuda.null_sink` | Consumes data on the GPU (no D2H transfer) |
| `cuda.probe_rate` | Measures throughput and reports via message port |
| `cuda.load` | Synthetic load generator for benchmarking GPU pipelines |

### Python (CuPy)

| Block | Description |
|-------|-------------|
| `cuda.add_py` | Adds N input streams on the GPU using CuPy |
| `cuda.multiply_const_py` | Multiplies by a constant on the GPU using CuPy |
| `cuda.fft_cupy` | Forward/inverse FFT on the GPU using CuPy |

## Prerequisites

- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- GNU Radio >= 3.10
- CMake >= 3.18, Ninja (recommended)
- [CuPy](https://cupy.dev/) (for Python GPU blocks)
- A C++17 compiler

## Building

```bash
mkdir build && cd build

cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev

ninja
ninja install
ninja test
```

This installs to the default system prefix (usually `/usr/local`). To change the install location:

```bash
cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/path/to/prefix \
    -DCMAKE_PREFIX_PATH=/path/to/prefix \
    -Wno-dev
```

If you're using a **conda environment**, point both paths at `$CONDA_PREFIX`:

```bash
cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -Wno-dev
```

> **Note:** The build automatically detects your GPU architecture. To target a specific architecture, add `-DCMAKE_CUDA_ARCHITECTURES=86` (or whichever compute capability you need).

## Writing a new GPU block

### C++ block

1. **Inherit from `cuda_block`** (alongside your GR block base) to get a managed non-blocking CUDA stream (`d_stream`) and kernel launch helpers:

```cpp
#include <gnuradio/cuda/cuda_block.h>
#include <gnuradio/cuda/cuda_block_helper.h>

class my_block_impl : public my_block, public gr::cuda_block
{
    // d_stream, d_min_grid_size, d_block_size are provided
};
```

2. **Request `cuda_buffer`** in the io_signature so buffers live on the GPU:

```cpp
my_block_impl::my_block_impl(...)
    : gr::sync_block("my_block",
          io_signature::make(1, 1, sizeof(float), cuda_buffer::type),
          io_signature::make(1, 1, sizeof(float), cuda_buffer::type))
{
}
```

3. **Synchronise with events** in `work()`:

```cpp
int my_block_impl::work(int noutput_items, ...)
{
    // Wait for upstream GPU data to be ready
    gr::cuda::wait_for_inputs(this->detail(), d_stream);

    // Launch your kernel
    auto in  = reinterpret_cast<const float*>(input_items[0]);
    auto out = reinterpret_cast<float*>(output_items[0]);
    my_kernel<<<grid, block, 0, d_stream>>>(in, out, noutput_items);

    // Signal downstream that outputs are ready
    gr::cuda::mark_outputs_ready(this->detail(), d_stream);
    return noutput_items;
}
```

See `cuda_block.h` for the full documentation and `multiply_const_impl.cc` for a complete example.

### Python/CuPy block

Python GPU blocks use CuPy arrays and the same synchronisation helpers:

```python
import numpy as np
import cupy as cp
from gnuradio import gr, cuda

class my_block_cupy(gr.sync_block):
    def __init__(self):
        # Define input and output signatures (can differ if needed)
        in_sig = cuda.io_signature_make(1, 1, [np.complex64])
        out_sig = cuda.io_signature_make(1, 1, [np.complex64])

        # Create block as normal, but with CUDA buffers
        gr.sync_block.__init__(self, "my_block_cupy", in_sig, out_sig)

        # Create a non-blocking CUDA stream for async GPU operations
        self.stream = cp.cuda.Stream(non_blocking=True)

    def work(self, input_items, output_items):
        # Wait for upstream GPU data to be ready
        cuda.wait_for_inputs(self.gateway, self.stream.ptr)

        # Do GPU work with CuPy on self.stream
        with self.stream:
            # Get CuPy input/output pointers
            d_in = cuda.as_cupy(input_items[0])
            d_out = cuda.as_cupy(output_items[0])
            
            # Do some GPU work
            d_out[:] = d_in * 2.0

        # Signal downstream that outputs are ready
        cuda.mark_outputs_ready(self.gateway, self.stream.ptr)
        return len(output_items[0])
```

See `multiply_const_py.py` for a complete example.

## Using gr-cuda in your own OOT

### Root `CMakeLists.txt`

Enable CUDA in your project:

```cmake
project(gr-myoot CXX C CUDA)
```

### `lib/CMakeLists.txt`

```cmake
# Find gr-cuda
find_package(gnuradio-cuda REQUIRED)

# Compile your CUDA kernels
add_library(gnuradio-myoot-cu STATIC my_kernel.cu)
set_target_properties(gnuradio-myoot-cu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Link your OOT to gr-cuda and your kernels
target_link_libraries(gnuradio-myoot PUBLIC gnuradio::gnuradio-runtime gnuradio-cuda)
target_link_libraries(gnuradio-myoot PRIVATE gnuradio-myoot-cu)
```

## Limitations

- **No mixed fan-out.** A block's output cannot fan out to both GPU (`cuda_buffer`) and CPU (default) downstream blocks simultaneously. All consumers of a given output port must use the same buffer type.
- **Large buffers at CPU/GPU boundaries.** H2D and D2H transfers need large batches (2^18--2^20 items) to saturate PCIe bandwidth. Use `set_output_multiple()` on boundary blocks.

## Performance

Benchmarked on an **NVIDIA DGX Spark (GB10)** and an **NVIDIA RTX PRO 6000 Blackwell**.

| Benchmark | GB10 peak | RTX PRO 6000 peak |
|-----------|-----------|-------------------|
| H2D transfer | 58.9 GB/s (99.8% of HW) | 56.5 GB/s (99.4% of HW) |
| D2H transfer | 50.7 GB/s | 55.4 GB/s |
| cuFFT (C++ block) | 9.2 Gsps | 64.7 Gsps |
| CuPy FFT (Python block) | 9.1 Gsps | 63.6 Gsps |
| FFTW (CPU baseline) | 0.46 Gsps | 1.02 Gsps |

CUDA buffers at CPU/GPU boundaries (H2D, D2H) should be large -- on the order of 2^18 to 2^20 items -- to amortise PCIe latency and achieve peak bandwidth. Use `set_output_multiple()` on boundary blocks to control this. See **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** for full tables, methodology, and analysis.

## Acknowledgements

This OOT is adapted from [gr-cuda_buffer](https://github.com/BlackLynx-Inc/gr-cuda_buffer) by Black Lynx, Inc. and relies on the custom buffer feature developed by David Sorber ([Custom Buffers wiki](https://wiki.gnuradio.org/index.php/CustomBuffers)). Further developed by Josh Morman and heavily modified and extended by Wael Farah at Cascade Space.

## License

GPL-3.0-or-later. See individual file headers for copyright details.