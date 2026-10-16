# Limitations

- **Single GPU only.** All blocks and buffers use the current CUDA device (device 0 by default). Use `CUDA_VISIBLE_DEVICES` to choose which GPU the flowgraph runs on.
- **Linux only.** The host-side circular buffers rely on Linux-specific system calls.
- **No mixed fan-out.** A block's output cannot fan out to both GPU (`cuda_buffer`) and CPU (default) downstream blocks simultaneously. All consumers of a given output port must use the same buffer type. This applies to both GPU and CPU source blocks. To work around this, use the `cuda.tee` block to split a single output into two separate ports, each with its own buffer. Alternatively, break the fan-out with a copy block so each path gets its own buffer: `cuda.copy` before CPU downstream blocks, or `blocks.copy()` before GPU downstream blocks. This is a GNU Radio core limitation (`buffer::set_transfer_type()` only supports a single transfer type per buffer); fixing it requires per-reader transfer types in upstream GR.

The rules for writing blocks (stream usage, produce/consume ordering) are in [WRITING_BLOCKS.md](WRITING_BLOCKS.md).
