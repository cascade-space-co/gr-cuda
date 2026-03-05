# gr-cuda

GPU-accelerated signal processing blocks for [GNU Radio](https://www.gnuradio.org/) 3.10+, built on the custom buffer API introduced by David Sorber. Data moves to/from the GPU automatically via `cuda_buffer`, so blocks receive device pointers directly in their `work()` function -- no manual `cudaMemcpy` needed.

<img title="Simple flowgraph with CUDA block" alt="Simple CUDA flowgraph" src="docs/img/flowgraph_copy.png">

## Features

- **Zero-copy GPU pipeline** -- `cuda_buffer` handles H2D/D2H transfers and event-based synchronisation between blocks, so GPU kernels run back-to-back without stalling. Internally it uses a double-buffered scheme so the CPU can fill one half while the GPU processes the other.
- **C++ blocks** with hand-written CUDA kernels (cuFFT, custom element-wise ops).
- **Python/CuPy blocks** that look exactly like regular GNU Radio Python blocks. The only differences: inherit from `cuda.sync_block` (or `cuda.decim_block`, etc.) instead of `gr.sync_block`, and `input_items`/`output_items` are CuPy arrays pointing directly into the underlying CUDA buffers. No manual memory management, no synchronisation code. CuPy blocks achieve throughput within 1-2% of equivalent C++ CUDA blocks (see [Performance](#performance)).
- **GRC support** -- all blocks ship with GNU Radio Companion block definitions.

## Included blocks

### C++ (CUDA kernel)

| Block | Description |
|-------|-------------|
| `cuda.add` | Adds N input streams element-wise |
| `cuda.multiply_const` | Multiplies input stream by a scalar constant |
| `cuda.fft` | Forward/inverse FFT via cuFFT, with optional windowing and shift |
| `cuda.copy` | Copies input to output (no-op mode available for benchmarking) |
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
| `cuda.add_cupy` | Adds N input streams on the GPU using CuPy |
| `cuda.multiply_const_cupy` | Multiplies by a constant on the GPU using CuPy |
| `cuda.fft_cupy` | Forward/inverse FFT on the GPU using CuPy |

## Prerequisites

- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- GNU Radio >= 3.10 (>3.10.12 recommended; older versions have a [fan-out deadlock bug](https://github.com/gnuradio/gnuradio/pull/8029))
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

> **Note:** Ongoing work will move the synchronisation logic into `cuda_buffer` itself (via an upstream GNU Radio change), making the C++ API as simple as the Python one -- just inherit and write your kernel, no manual `wait_for_work`/`mark_work_done` calls.

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
    gr::cuda::wait_for_work(detail(), d_stream);

    // Launch your kernel
    auto in  = static_cast<const float*>(input_items[0]);
    auto out = static_cast<float*>(output_items[0]);
    my_kernel<<<grid, block, 0, d_stream>>>(in, out, noutput_items);

    // Signal downstream that outputs are ready
    gr::cuda::mark_work_done(detail(), d_stream);
    return noutput_items;
}
```

See [`cuda_block.h`](include/gnuradio/cuda/cuda_block.h) for the full documentation and [`multiply_const_impl.cc`](lib/multiply_const_impl.cc) for a complete example.

### Python/CuPy block

Inherit from `cuda.sync_block` instead of `gr.sync_block`. CUDA buffer allocation, synchronisation, and CuPy array conversion are handled automatically -- `input_items` and `output_items` are CuPy arrays:

```python
import numpy as np
from gnuradio import cuda

class my_block_cupy(cuda.sync_block):    # cuda.sync_block instead of gr.sync_block
    def __init__(self):
        cuda.sync_block.__init__(self, "my_block_cupy",
            [np.complex64],
            [np.complex64])

    def work(self, input_items, output_items):
        # input_items / output_items are CuPy arrays
        # use any cp.* operation
        cp.multiply(input_items[0], 2.0, out=output_items[0])
        return len(output_items[0])
```

`cuda.decim_block`, `cuda.interp_block`, and `cuda.basic_block` are also available for other block types. See [`multiply_const_cupy.py`](python/cuda/multiply_const_cupy.py) for a complete example.

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

| Benchmark | GB10 | RTX PRO 6000 |
|-----------|-----:|-------------:|
| H2D transfer | 58.3 GB/s | 54.1 GB/s |
| D2H transfer | 59.1 GB/s | 56.4 GB/s |
| Full round-trip | 29.2 GB/s | 38.7 GB/s |
| cuFFT (C++) | 13.9 Gsps | 92.8 Gsps |
| CuPy FFT (Python) | 13.5 Gsps | 88.3 Gsps |
| FFTW (CPU baseline) | 0.45 Gsps | 1.01 Gsps |

See **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** for full tables, methodology, and analysis.

## Acknowledgements

This OOT is adapted from [gr-cuda_buffer](https://github.com/BlackLynx-Inc/gr-cuda_buffer) by Black Lynx, Inc. and relies on the custom buffer feature developed by David Sorber ([Custom Buffers wiki](https://wiki.gnuradio.org/index.php/CustomBuffers)). Further developed by Josh Morman and heavily modified and extended by Wael Farah at Cascade Space.

## License

GPL-3.0-or-later. See individual file headers for copyright details.
