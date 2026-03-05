# Limitations

- **No mixed fan-out.** A block's output cannot fan out to both GPU (`cuda_buffer`) and CPU (default) downstream blocks simultaneously. All consumers of a given output port must use the same buffer type.
- **Large buffers at CPU/GPU boundaries.** H2D and D2H transfers need large batches (2^18--2^20 items) to saturate PCIe bandwidth. Use `set_output_multiple()` on boundary blocks.
