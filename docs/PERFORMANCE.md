# Performance

All numbers are single-chain, `gr_complex` (complex64, 8 bytes/item).
Run the benchmarks on your own system with `benchmark_gr_cuda_transfer.py` and `benchmark_gr_cuda_fft.py` (installed to `bin/`).

## Transfer throughput

The GPU block in each chain is `multiply_const_cc(1)`, so a kernel reads and writes every sample. Buffer sizes are the ring size per edge; each `work()` call is a quarter of the ring. All numbers at default settings (`gpu_busy_wait = true`, 10s duration, 1s warmup), with `--buff-exp-min 20 --buff-exp-max 32` (rings from 1 MiB to 4 GiB).

### NVIDIA RTX PRO 6000 Blackwell — AMD EPYC 9335, PCIe 5.0 x16

| Direction | Peak GB/s | Peak Gsps |
|-----------|----------:|----------:|
| H2D (CPU → GPU)       | 56.8 | 7.10 |
| D2H (GPU → CPU)       | 54.0 | 6.75 |
| Full round-trip        | 40.5 | 5.06 |

### NVIDIA DGX Spark (GB10) — ARM Cortex-X925, unified memory

| Direction | Peak GB/s | Peak Gsps |
|-----------|----------:|----------:|
| H2D (CPU → GPU)       | 116.0 | 14.50 |
| D2H (GPU → CPU)       | 108.8 | 13.60 |
| Full round-trip        | 82.3 | 10.28 |

### Notes

- **RTX PRO 6000**: all directions are within about 10% of their peak from a 16 MiB buffer.
- **GB10**: H2D and D2H edges are zero-copy on integrated GPUs, so no host/device copy takes place. These numbers measure the kernel working on shared memory and aren't comparable with the PCIe ones. All directions are within about 10% of their peak from a 16 MiB buffer.
- **GB10 small buffers**: 4–8 MiB buffers measured higher than the table (up to 24.3 Gsps H2D at 8 MiB). That's more memory traffic than DRAM can deliver, so these buffers are likely served from on-chip cache, which flowgraphs with more edges won't fit in. The table lists peaks from 16 MiB up.
- **`gpu_busy_wait = false`** gave lower results at small and mid-sized buffers, e.g. 6.74 Gsps H2D at 32 MiB on the GB10 and 3.41 Gsps at 4 MiB on the RTX PRO 6000. Each wait then includes waking a sleeping thread, a cost paid on every `work()` call; smaller buffers mean more calls per second, so it weighs more.

## FFT throughput

GPU benchmarks run entirely on-device (GPU null_source → FFT → GPU null_sink) to isolate compute from PCIe overhead.
Batch size is auto-scaled to ~64 MiB per `work()` call. All numbers at default settings (10s duration, 1s warmup).

### NVIDIA RTX PRO 6000 Blackwell — AMD EPYC 9335

| FFT size | cuFFT FFTs/s | cuFFT Gsps | CuPy Gsps | FFTW Gsps | GPU vs CPU |
|----------|------------:|----------:|----------:|----------:|----------:|
| 2^8  | 358,330,654 | 91.7 | 86.9 | 0.87 | 105x |
| 2^9  | 180,538,541 | 92.4 | 87.8 | 0.97 | 95x |
| 2^10 | 90,249,551 | 92.4 | 87.9 | 0.94 | 98x |
| 2^11 | 45,188,051 | 92.6 | 87.9 | 1.01 | 92x |
| 2^12 | 22,630,148 | 92.7 | 88.2 | 0.99 | 94x |
| 2^13 | 11,329,841 | 92.8 | 88.3 | 0.80 | 116x |
| 2^14 | 5,630,885 | 92.3 | 88.1 | 0.73 | 126x |
| 2^15 | 1,911,377 | 62.6 | 60.4 | 0.63 | 99x |
| 2^16 | 960,388 | 62.9 | 60.7 | 0.54 | 117x |
| 2^17 | 482,089 | 63.2 | 60.6 | 0.46 | 137x |
| 2^18 | 238,529 | 62.5 | 60.3 | 0.40 | 156x |
| 2^19 | 119,652 | 62.7 | 60.5 | 0.21 | 299x |
| 2^20 | 58,151 | 61.0 | 58.6 | 0.21 | 290x |

### NVIDIA DGX Spark (GB10) — ARM Cortex-X925

| FFT size | cuFFT FFTs/s | cuFFT Gsps | CuPy Gsps | FFTW Gsps | GPU vs CPU |
|----------|------------:|----------:|----------:|----------:|----------:|
| 2^8  | 53,407,212 | 13.7 | 13.3 | 0.44 | 31x |
| 2^9  | 26,698,588 | 13.7 | 13.4 | 0.45 | 30x |
| 2^10 | 13,521,338 | 13.9 | 13.6 | 0.41 | 34x |
| 2^11 | 6,749,619 | 13.8 | 13.5 | 0.38 | 36x |
| 2^12 | 3,381,133 | 13.9 | 13.5 | 0.35 | 40x |
| 2^13 | 1,693,456 | 13.9 | 13.5 | 0.32 | 43x |
| 2^14 | 834,128 | 13.7 | 13.4 | 0.27 | 51x |
| 2^15 | 187,644 | 6.15 | 6.16 | 0.21 | 29x |
| 2^16 | 95,064 | 6.23 | 6.15 | 0.16 | 39x |
| 2^17 | 48,569 | 6.37 | 6.28 | 0.15 | 42x |
| 2^18 | 23,561 | 6.18 | 6.15 | 0.15 | 41x |
| 2^19 | 11,931 | 6.26 | 6.00 | 0.14 | 45x |
| 2^20 | 5,995 | 6.29 | 5.86 | 0.13 | 48x |

### Notes

- **cuFFT shared-memory cliff at 2^14 → 2^15.** FFTs up to 16,384 points fit in GPU shared memory and execute in a single kernel pass. At 32K points the working set exceeds shared memory, forcing `cuFFT` to use multiple passes through global memory. This is intrinsic to `cuFFT`: GB10 drops from ~14 to ~6 Gsps, RTX PRO 6000 from ~93 to ~63 Gsps.
- **CuPy ≈ C++ cuFFT.** The CuPy block uses a cached `cufft.Plan1d` and writes directly into the output buffer. It tracks within 5% of the C++ block on both platforms — write your GPU blocks in Python with no performance penalty.
- **FFTW (CPU)** column uses GNU Radio's built-in `fft_vcc` block (single-threaded FFTW). Speedup over CPU ranges from 29–48x on the GB10 to 92–299x on the RTX PRO 6000, increasing with FFT size.

> Run `benchmark_gr_cuda_fft.py --plot fft_bench.png` to generate a comparison plot, or `--csv results.csv` to save raw data.
