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

## 2. Increase `set_min_output_buffer()`

If throughput is not sufficient, call `set_min_output_buffer()` on each
block to at least **4 × 32 MB = 128 MB** worth of items
(`128 * 1024 * 1024 / itemsize`, where `itemsize` already includes
`vlen`, e.g. `sizeof(gr_complex) * vlen`). This inflates all buffers
on that block's output.

```python
buff_size_samples = 2**18          # items per batch
nbuff = 16                         # number of batches per buffer

# Scalar ports
block.set_min_output_buffer(nbuff * buff_size_samples)

# Vector-length ports (e.g. after stream_to_vector)
block.set_min_output_buffer(nbuff * buff_size_samples // vlen)
```

This can also be set per-block in GRC via the **Minoutbuf**
property in the block's Advanced tab.

## 3. Set `set_output_multiple()`

`set_output_multiple()` tells the scheduler the minimum number of items
to accumulate before calling `work()`. Set it to at least **16 MB** worth
of items (`16 * 1024 * 1024 / itemsize`). This should be a few times
smaller than `set_min_output_buffer()` -- the buffer needs room for double
buffering while the scheduler enforces the batch minimum.

## 4. Time your blocks

gr-cuda ships with `cuda.null_source`, `cuda.null_sink`, and
`cuda.probe_rate` blocks for benchmarking. Isolate the block under
test between a null source and null sink, attach a probe rate, and
measure sustained throughput. This lets you identify whether the
bottleneck is a specific block, buffer sizing, or transfers.

## 5. CuPy temporary allocations

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
