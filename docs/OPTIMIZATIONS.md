# Performance Optimizations

Optimal buffer sizes and batch counts depend on your GPU, PCIe
generation, item size, and pipeline depth. The guidance below provides
a good starting point, but profiling your specific system (e.g. with
`nsys`) is the best way to find the right values.

## 1. Start with defaults

`cuda_buffer` enforces a 32 MB floor on every GPU-side buffer allocation.
The GNU Radio scheduler caps each `work()` call at `bufsize / 2`, so
out of the box each block processes ~16 MB per call. This is often
sufficient for moderate-throughput pipelines; try it first.

The floor is configurable in `~/.gnuradio/config.conf`:

```ini
[cuda_buffer]
min_buffer_bytes = 16777216   # 16 MB (default: 33554432 = 32 MB)
```

Note: `set_max_output_buffer()` caps item counts *before* the buffer
is allocated, but `cuda_buffer` still applies the floor afterwards.
To truly shrink below the floor, lower `min_buffer_bytes` in the config.

## 2. Increase `set_min_output_buffer()`

If throughput is not sufficient, `set_min_output_buffer()` is the
single most impactful knob. Call it on each block to increase the
output buffer size. All values are in **items** (not bytes). For
vector-length ports, one "item" is one full vector, so divide by
`vlen`. Target at least 2--8 MB worth of data per batch (e.g. 2^18
complex64 samples = 2 MB), multiplied by `nbuff` to leave room for
double buffering.

```python
buff_size_samples = 2**18          # items per batch
nbuff = 16                         # buffer holds nbuff batches

# Scalar ports
block.set_min_output_buffer(nbuff * buff_size_samples)

# Vector-length ports (e.g. after stream_to_vector)
block.set_min_output_buffer(nbuff * buff_size_samples // vlen)
```

This can also be set per-block in GRC via the **Minoutbuf** property
in the block's Advanced tab.

### Secondary: `set_output_multiple()` and `set_max_noutput_items()`

For further tuning, these two knobs control the batch size the
scheduler passes to `work()`:

- `set_output_multiple(buff_size_samples)`: constrains `noutput_items`
  to always be a multiple of the given value (the scheduler will not
  call `work()` with fewer items, and will round down to the nearest
  multiple). This particularly helps at H2D and D2H edges.
- `set_max_noutput_items(buff_size_samples)`: caps the maximum items
  per `work()` call. Setting this equal to `output_multiple` forces
  fixed-size batches, which can be advantageous for blocks that
  cache plans or use temporary allocations internally.

```python
# Scalar ports
block.set_output_multiple(buff_size_samples)
block.set_max_noutput_items(buff_size_samples)

# Vector-length ports
block.set_output_multiple(buff_size_samples // vlen)
block.set_max_noutput_items(buff_size_samples // vlen)
```

## 3. Time your blocks

gr-cuda ships with `cuda.null_source`, `cuda.null_sink`, and
`cuda.probe_rate` blocks for benchmarking. Isolate the block under
test between a null source and null sink, attach a probe rate, and
measure sustained throughput. This lets you identify whether the
bottleneck is a specific block, buffer sizing, or transfers.

## 4. CuPy temporary allocations

CuPy is convenient for writing GPU blocks in Python, but intermediate
expressions allocate temporary device arrays behind the scenes. For
example:

```python
# complex_to_mag_squared: 2 hidden temporaries
cp.add(x.real**2, x.imag**2, out=output)
#      ^^^^^^^^   ^^^^^^^^
#      temp #1    temp #2   (each a cudaMalloc on first call)

# nlog10: 1 hidden temporary
cp.log10(cp.maximum(x, tiny), out=output)
#        ^^^^^^^^^^^^^^^^^^
#        temp #1
```

CuPy's memory pool caches these after the first `work()` call, so
subsequent iterations reuse them without hitting `cudaMalloc`. However,
if `noutput_items` varies between calls (which it can), new sizes trigger
fresh allocations. To eliminate runtime `cudaMalloc` entirely, replace
hot-path expressions with fused `cp.ElementwiseKernel` calls.
