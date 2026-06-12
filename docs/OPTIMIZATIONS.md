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

The floor is configurable via the GNU Radio user config
(`gnuradio-config-info --userprefsdir` shows the directory):

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

## 4. GPU synchronization: busy-wait vs sleep

`cuda_buffer` uses CUDA events to synchronize CPU and GPU work. By
default, `cudaEventSynchronize` busy-waits (spin-polls): the thread
loops on the CPU, giving the lowest possible latency but consuming a
full CPU core per waiting thread. This is a good default for
throughput-critical pipelines where CPU cores are plentiful relative
to GPU blocks.

For flowgraphs with many GPU blocks, or on systems where CPU usage
matters more than shaving microseconds of latency, disable busy-wait.
This puts the thread to sleep and yields the CPU core while waiting:

```ini
[cuda_buffer]
gpu_busy_wait = false
```

> **Note:** The performance impact is system-dependent. Sleeping
> introduces OS thread wake-up latency each time an event completes,
> which can measurably reduce transfer bandwidth on some systems (e.g.
> PCIe 5.0 → RTX PRO 6000). Profile both modes on your hardware with
> the transfer benchmark (`benchmark_gr_cuda_transfer.py`) before
> choosing.

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

## 6. IBV sink/source tuning (`ibv_sink` / `ibv_source`)

The InfiniBand-verbs blocks drive a NIC at line rate using GPUDirect 
RDMA. Their tuning knobs are **not** constructor arguments; they are 
read from the GNU Radio user config (`gnuradio-config-info --userprefsdir`) 
at block construction, with the compile-time `DEFAULT_*` constants as
fallbacks. The defaults target 100 GbE on a ConnectX-class NIC; profile
before changing them.

```ini
[ibv_sink]
num_wr        = 4096        ; send pipeline depth (work requests in flight)
signal_batch  = 512         ; sends per completion signal
cq_size       = 8192        ; completion queue depth (default: 2 * num_wr)
cq_poll_batch = 64          ; completions drained per ibv_poll_cq() call
gpu_buf_bytes = 67108864    ; GPU landing-buffer size in bytes (64 MiB)

[ibv_source]
num_wr        = 4096        ; receive pipeline depth (recv WRs posted)
cq_size       = 8192        ; completion queue depth (default: 2 * num_wr)
cq_poll_batch = 512         ; completions drained per ibv_poll_cq() call
gpu_buf_bytes = 67108864    ; GPU landing-buffer size in bytes (64 MiB)
```

What each knob does:

- **`num_wr`** — depth of the send (sink) or receive (source) pipeline:
  the number of work requests that can be outstanding (posted to the NIC
  but not yet completed) at once. Deeper pipelines hide PCIe/NIC latency
  and absorb bursts (on the source, this is the main lever against RX
  drops), at the cost of more registered memory and CQ entries.

- **`signal_batch`** (sink only) — only every `signal_batch`-th send is
  posted with `IBV_SEND_SIGNALED`, so the NIC generates one completion
  per batch instead of one per frame. This amortizes completion-queue
  processing, the dominant CPU cost at line rate. Must divide `num_wr`
  evenly. It also sets `output_multiple(signal_batch)`, so the scheduler
  always hands the block at least a full batch of frames. The source has
  no equivalent: receives complete one-per-frame and are drained in bulk
  via `cq_poll_batch`.

- **`cq_size`** — depth of the completion queue. Must be large enough to
  hold the maximum number of outstanding completions (`num_wr /
  signal_batch` on the sink; up to `num_wr` on the source). The `2 *
  num_wr` default leaves comfortable headroom.

- **`cq_poll_batch`** — how many completions a single `ibv_poll_cq()`
  call harvests (the size of the work-completion scratch array). Larger
  values mean fewer poll calls when many completions are ready. The
  source defaults higher (512 vs the sink's 64) because RX completes
  one-per-frame and benefits from draining many at once.

- **`gpu_buf_bytes`** — size of the GPU landing buffer (registered as an
  IB memory region) that frames are assembled into (sink) or received
  into (source) and DMA'd to/from the NIC. It is divided into fixed
  slots of one full frame each, so it must hold at least `num_wr` slots
  at the frame size; larger buffers allow a deeper pipeline and bigger
  frames.
