# Limitations

- **No mixed fan-out.** A block's output cannot fan out to both GPU (`cuda_buffer`) and CPU (default) downstream blocks simultaneously. All consumers of a given output port must use the same buffer type. This applies to both GPU and CPU source blocks. To work around this, use the `cuda.tee` block to split a single output into two separate ports, each with its own buffer. Alternatively, break the fan-out with a copy block so each path gets its own buffer: `cuda.copy` before CPU downstream blocks, or `blocks.copy()` before GPU downstream blocks. This is a GNU Radio core limitation (`buffer::set_transfer_type()` only supports a single transfer type per buffer); fixing it requires per-reader transfer types in upstream GR.
- **Large buffers at CPU/GPU boundaries.** While not a limitation per se, H2D and D2H transfers need large batches to saturate PCIe bandwidth. `cuda_buffer` enforces a 32 MB floor on GPU-side buffers to provide a reasonable default. See [docs/OPTIMIZATIONS.md](OPTIMIZATIONS.md) for tuning guidance.
- **Stream contract.** Every block gets its own CUDA stream. For `cuda_buffer`'s inter-block auto-sync to work, **all GPU work in `work()` / `general_work()` must run on that stream**. Work submitted on any other stream lands outside the buffer's sync graph and races with neighbouring blocks.
    - **C++**: no equivalent of CuPy's context manager exists, so pass `d_stream` to every kernel launch (`<<<grid, block, 0, d_stream>>>`) and every CUDA / library API call (`cudaMemcpyAsync(..., d_stream)`, `cufftSetStream(plan, d_stream)`, `cublasSetStream(handle, d_stream)`, …).
    - **Python**: `cuda.sync_block` wraps `work()` / `general_work()` in `with self.stream:`, so any `cp.*` call inside picks up the stream automatically. CuPy is the only library exercised by the QA suite. Mixing in other GPU libraries (PyTorch, Numba CUDA, PyCUDA, …) is untested and requires you to route their work onto `self.stream` yourself. For libraries with thread-local "current stream" state (PyTorch, cuDNN handles, cuFFT plans), the cleanest pattern is to bind them once in `start()` — GR's TPB scheduler runs `start()` and every subsequent `work()` call on the same worker thread, so the binding persists:

        ```python
        class my_torch_block(cuda.sync_block):
            def __init__(self):
                cuda.sync_block.__init__(self, "my_torch_block",
                                         [np.complex64], [np.complex64])

            def start(self):
                torch.cuda.set_stream(torch.cuda.ExternalStream(self.stream.ptr))
        ```

        For libraries that take a stream argument per call (Numba CUDA, PyCUDA), pass `self.stream` (or `self.stream.ptr`) explicitly to each call instead.
