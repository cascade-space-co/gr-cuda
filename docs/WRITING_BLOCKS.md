# Writing GPU blocks

The [README](../README.md#writing-gpu-blocks) has minimal Python and C++
examples. This page lists the rules every gr-cuda block must follow, in both
languages, and explains each one.

## The rules

1. **All device work runs on the block's stream**: in `work()` /
   `general_work()`, and also in the constructor, `start()`, setters, getters,
   and message handlers.
2. **In `general_work()`, enqueue all GPU work before calling `produce()` /
   `consume()` / `consume_each()`.**

Breaking either rule does not raise an error. The block races with its
neighbours and produces wrong data intermittently.

## How synchronization works

Every block gets its own non-blocking CUDA stream: `d_stream` (`cudaStream_t`)
in C++, `self.stream` (`cupy.cuda.Stream`) in Python. `cuda_buffer` orders
blocks by recording CUDA events on the producer's stream and making the
consumer's stream wait on them. Only work submitted on the block's stream is
covered by those events. Work on any other stream, including the default
stream, is not covered by `cuda_buffer`'s event ordering.

## Rule 1: use the block's stream

### Inside `work()` / `general_work()`

- **C++**: pass `d_stream` to every kernel launch
  (`<<<grid, block, 0, d_stream>>>`) and every CUDA / library API call
  (`cudaMemcpyAsync(..., d_stream)`, `cufftSetStream(plan, d_stream)`,
  `cublasSetStream(handle, d_stream)`, …).
- **Python**: `cuda.sync_block` and friends wrap `work()` / `general_work()` in
  `with self.stream:`, so any `cp.*` call inside picks up the stream
  automatically.

### Outside `work()`

The constructor, `start()`, setters, getters, and message handlers do **not**
run inside `with self.stream:` (`start()` is wrapped in Python, but only to
guarantee stream registration). In Python, put device work there inside
`with self.stream:` yourself; in C++, pass `d_stream` as usual. Otherwise the
work lands on the default stream and is not ordered with `work()`: for example,
`work()` may read the data a setter uploads before the upload completes.

```python
class my_lookup(cuda.sync_block):
    def __init__(self, table):
        cuda.sync_block.__init__(self, "my_lookup", [np.int32], [np.float32])
        with self.stream:
            self._table = cp.asarray(table, dtype=np.float32)

    def set_table(self, table):  # e.g. a GRC callback
        with self.stream:
            self._table = cp.asarray(table, dtype=np.float32)

    def work(self, input_items, output_items):  # already on self.stream
        cp.take(self._table, input_items[0], out=output_items[0])
        return len(output_items[0])
```

`with self.stream:` orders GPU work only; it does not make setters thread-safe
with respect to `work()`. That is the same as for any GNU Radio block.

### Reading GPU results on the host

GPU work is asynchronous, so a getter that returns a value computed on the
device must wait for the work that produces it, or it returns stale data.

- **Python**: inside `with self.stream:`, `.get()`, `.item()`, `float()` and
  `int()` copy on `self.stream` and block until the copy finishes, so they see
  all work enqueued on the stream before them. Outside it, the copy runs on the
  default stream, which does not wait for the block's non-blocking stream.
- **C++**: `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, d_stream)`, then
  `cudaStreamSynchronize(d_stream)` before using the host value.

```python
def table_sum(self):  # a getter on my_lookup above
    with self.stream:
        return float(self._table.sum())  # waits for self.stream
```

### Other GPU libraries in Python

CuPy is the only library exercised by the QA suite. Mixing in other GPU
libraries (PyTorch, Numba CUDA, PyCUDA, …) is untested and requires you to route
their work onto `self.stream` yourself. For libraries with thread-local "current
stream" state (PyTorch, cuDNN handles, cuFFT plans), the cleanest pattern is to
bind them once in `start()`. GR's Thread Per Block (TPB) scheduler runs
`start()` and every subsequent `work()` call on the same worker thread, so the
binding persists:

```python
class my_torch_block(cuda.sync_block):
    def __init__(self):
        cuda.sync_block.__init__(self, "my_torch_block",
                                 [np.complex64], [np.complex64])

    def start(self):
        torch.cuda.set_stream(torch.cuda.ExternalStream(self.stream.ptr))
```

For libraries that take a stream argument per call (Numba CUDA, PyCUDA), pass
`self.stream` (or `self.stream.ptr`) explicitly to each call instead.

## Rule 2: enqueue, then publish

GPU work is asynchronous: `produce()` / `consume_each()` record the buffer's
device-ready / read-done events at the moment they are called. **All GPU work
that reads an input or writes an output must be enqueued on the block's stream
before you `consume()` / `produce()` that port.** Publishing first and computing
afterwards records the events too early and lets a neighbour read before the
write (or overwrite while reading) with no diagnostic. The rule is **identical
in C++ and Python**: write your block as if it applied to both.

- **C++**: call `produce()` / `consume_each()` last, after every
  `<<<..., d_stream>>>` launch and `cudaMemcpyAsync(..., d_stream)` for that
  data. See `lib/qa_general_work_cpp.cc` for the canonical shape.
- **Python**: as a backstop, `cuda.basic_block` defers `produce()` /
  `consume()` / `consume_each()` until after `general_work()` returns, so a
  mis-ordered call hardens into correct behaviour instead of silently
  corrupting data. **This is a safety net, not a feature**: do not rely on it;
  write to the rule above so the block ports cleanly to C++. Two consequences
  of the deferral:
    - The side effects of `produce()` / `consume()` (e.g. `nitems_written()`,
      `nitems_read()`, tags) are **not observable within the same
      `general_work()` call**; they take effect when it returns. Track running
      offsets in local variables (the idiomatic GR style) rather than
      re-querying the framework mid-call.
    - Deferral only covers the ordering of work on `self.stream`. Work
      submitted on any other stream still races (see rule 1).
