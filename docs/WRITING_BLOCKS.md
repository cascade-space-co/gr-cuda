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
3. **Python only: in your `__init__`, use `self.stream` only after the
   `cuda.sync_block.__init__(...)` call** (or `cuda.basic_block.__init__`,
   etc.). Plain Python attributes can be set before it; device state cannot.

Breaking a rule does not raise an error. The block races with its neighbours
and produces wrong data intermittently.

## How synchronization works

Every block gets its own non-blocking CUDA stream: `d_stream` (`cudaStream_t`)
in C++, `self.stream` (`cupy.cuda.Stream`) in Python. `cuda_buffer` orders
blocks by recording CUDA events on the producer's stream and making the
consumer's stream wait on them. Only work submitted on the block's stream is
covered by those events. Work on any other stream, including the default
stream, is not ordered with anything.

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

The constructor, `start()`, setters, getters, and message handlers are **not**
wrapped. In Python, put device work there inside `with self.stream:` yourself;
in C++, pass `d_stream` as usual. Otherwise the work lands on the default stream
and is not ordered with `work()`: for example, `work()` may read the data a
setter uploads before the upload completes.

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

### Other GPU libraries in Python

CuPy is the only library exercised by the QA suite. Mixing in other GPU
libraries (PyTorch, Numba CUDA, PyCUDA, …) is untested and requires you to route
their work onto `self.stream` yourself. For libraries with thread-local "current
stream" state (PyTorch, cuDNN handles, cuFFT plans), the cleanest pattern is to
bind them once in `start()`. GR's TPB scheduler runs `start()` and every
subsequent `work()` call on the same worker thread, so the binding persists:

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

## Rule 3: use `self.stream` only after `cuda.sync_block.__init__(...)`

In Python, `self.stream` is created on first access, which needs the GNU Radio
gateway that `cuda.sync_block.__init__(...)` (or the `basic_block` /
`decim_block` / `interp_block` equivalent) sets up. Accessing it before that
call recurses in the gateway's `__getattr__` until Python raises
`RecursionError`:

```python
def __init__(self, taps):
    self._ntaps = len(taps)                     # fine: plain Python
    cuda.sync_block.__init__(self, "my_fir", [np.float32], [np.float32])
    with self.stream:                           # fine: after the call above
        self._taps = cp.asarray(taps, dtype=np.float32)
```
