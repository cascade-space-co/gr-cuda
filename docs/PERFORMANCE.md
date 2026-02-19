# Performance

## PCIe transfer throughput

Measured with `benchmark_gr_cuda_transfer.py` (installed to `bin/`).
All numbers are single-chain, item size 8 bytes (`gr_complex`).
Raw hardware baseline measured with [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) (Copy Engine mode).

### NVIDIA DGX Spark (GB10) -- ARM Cortex-X925, PCIe 5.0 x16

| Direction | Peak GB/s | Peak Gsps | nvbandwidth CE | Efficiency |
|-----------|-----------|-----------|----------------|------------|
| H2D (CPU -> GPU) | 58.9 | 7.36 | 59.0 GB/s | 99.8% |
| D2H (GPU -> CPU) | 50.7 | 6.34 | 59.0 GB/s | 85.9% |
| Full round-trip | 29.2 | 3.65 | 58.9 GB/s | 49.6% |
| Full round-trip (4 chains) | 28.7 | 3.59 | 58.9 GB/s | 48.7% |

### NVIDIA RTX PRO 6000 Blackwell -- AMD EPYC 9335, PCIe 5.0 x16

| Direction | Peak GB/s | Peak Gsps | nvbandwidth CE | Efficiency |
|-----------|-----------|-----------|----------------|------------|
| H2D (CPU -> GPU) | 56.5 | 7.06 | 56.8 GB/s | 99.4% |
| D2H (GPU -> CPU) | 55.4 | 6.92 | 56.5 GB/s | 98.0% |
| Full round-trip | 30.1 | 3.76 | 42.1 GB/s | 71.5% |
| Full round-trip (4 chains) | 40.0 | 5.00 | 42.1 GB/s | 95.0% |

### Transfer notes

- **H2D** achieves near-theoretical PCIe bandwidth on both systems (99%+).
- **D2H asymmetry** varies by platform. The GB10 shows the typical PCIe read penalty (~86%), while the RTX PRO 6000 is nearly symmetric (~98%). This depends on chipset, IOMMU, and PCIe topology.
- **Full round-trip** throughput is limited by the serialisation of H2D and D2H transfers within each batch.
- **Parallel chains** help on higher-latency platforms. The RTX PRO 6000 (832 ns PCIe latency, discrete add-in card) jumps from 71% to **95%** of bidirectional bandwidth with 4 chains. The DGX Spark's GB10 (304 ns, on-module GPU with short PCIe path to the Grace CPU) already saturates with a single chain, so extra chains don't help.
- **Buffer size sweet spot** varies by platform: the GB10 is stable across 2^19--2^22, while the RTX PRO 6000 performs best at 2^18--2^20 and drops at larger sizes.

> **Tip:** Run `benchmark_gr_cuda_transfer.py` on your system to find the optimal buffer size and chain count. Start with `set_output_multiple()` of 2^18 to 2^20.

## FFT throughput

Measured with `benchmark_gr_cuda_fft.py`. All benchmarks use `gr_complex` (complex64) vectors.
GPU tests run entirely on-device (GPU null_source -> FFT -> GPU null_sink) to isolate compute from PCIe overhead.
The `output_multiple` (batch size per `work()` call) is auto-scaled to ~64 MiB per call so that all engines are compared fairly.
Raw cuFFT ceiling measured with a standalone cuFFT C2C benchmark using CUDA event timing (plan creation excluded).

### NVIDIA DGX Spark (GB10) -- ARM Cortex-X925

| FFT size | Raw cuFFT (FFTs/s) | cuFFT block (FFTs/s) | Efficiency | cuFFT Gsps | CuPy Gsps | FFTW Gsps |
|----------|-------------------:|---------------------:|-----------:|-----------:|----------:|----------:|
| 2^8  | 53,430,438 | 34,982,713 | 65% | 8.96 | 8.78 | 0.46 |
| 2^9  | 26,880,218 | 17,728,759 | 66% | 9.08 | 8.93 | 0.45 |
| 2^10 | 13,542,667 | 8,946,223  | 66% | 9.16 | 8.95 | 0.41 |
| 2^11 | 6,741,661  | 4,478,111  | 66% | 9.17 | 8.94 | 0.38 |
| 2^12 | 3,378,803  | 2,247,229  | 67% | 9.20 | 8.97 | 0.35 |
| 2^13 | 1,697,477  | 1,126,043  | 66% | 9.22 | 9.06 | 0.32 |
| 2^14 | 836,652    | 549,314    | 66% | 9.00 | 9.03 | 0.27 |
| 2^15 | 193,767    | 153,902    | 79% | 5.04 | 5.17 | 0.22 |
| 2^16 | 96,149     | 76,900     | 80% | 5.04 | 5.12 | 0.16 |
| 2^17 | 49,505     | 39,481     | 80% | 5.17 | 4.98 | 0.15 |
| 2^18 | 24,131     | 19,430     | 81% | 5.09 | 4.89 | 0.15 |
| 2^19 | 12,216     | 9,595      | 79% | 5.03 | 5.24 | 0.14 |
| 2^20 | 6,078      | 4,638      | 76% | 4.86 | 4.28 | 0.13 |

### NVIDIA RTX PRO 6000 Blackwell -- AMD EPYC 9335

| FFT size | Raw cuFFT (FFTs/s) | cuFFT block (FFTs/s) | Efficiency | cuFFT Gsps | CuPy Gsps | FFTW Gsps |
|----------|-------------------:|---------------------:|-----------:|-----------:|----------:|----------:|
| 2^8  | 1,329,676,760 | 242,021,517 | 18% | 61.96 | 62.77 | 0.87 |
| 2^9  | 615,705,056   | 121,677,748 | 20% | 62.30 | 62.44 | 0.80 |
| 2^10 | 307,783,874   | 60,955,980  | 20% | 62.42 | 62.36 | 0.95 |
| 2^11 | 143,644,458   | 30,652,955  | 21% | 62.78 | 62.32 | 1.02 |
| 2^12 | 71,429,368    | 15,430,543  | 22% | 63.20 | 63.43 | 1.00 |
| 2^13 | 32,242,187    | 7,671,378   | 24% | 62.84 | 63.63 | 0.83 |
| 2^14 | 11,345,905    | 3,950,762   | 35% | 64.73 | 63.44 | 0.67 |
| 2^15 | 1,921,537     | 1,116,150   | 58% | 36.57 | 36.51 | 0.64 |
| 2^16 | 961,394       | 557,627     | 58% | 36.54 | 36.39 | 0.55 |
| 2^17 | 481,432       | 281,092     | 58% | 36.84 | 36.81 | 0.47 |
| 2^18 | 239,889       | 140,441     | 59% | 36.82 | 36.69 | 0.40 |
| 2^19 | 120,175       | 70,502      | 59% | 36.96 | 36.91 | 0.21 |
| 2^20 | 57,756        | 35,125      | 61% | 36.83 | 36.72 | 0.21 |

### FFT notes

- **cuFFT shared-memory cliff at 2^14 -> 2^15.** FFTs up to 16,384 points fit in GPU shared memory and execute in a single kernel pass. At 32K points the working set exceeds shared memory, forcing cuFFT to use multiple passes through global memory. This is intrinsic to cuFFT: GB10 drops from ~9 to ~5 Gsps, RTX PRO 6000 from ~65 to ~37 Gsps.
- **gr-cuda cuFFT efficiency** depends on GPU and FFT size. On the GB10, efficiency is a steady 65-67% for small FFTs and 76-81% for large FFTs. On the RTX PRO 6000 (62 Gsps at 2^8), the raw hardware ceiling is much higher (340 Gsps), so the efficiency percentage is lower (18-35%) even though absolute throughput is excellent[^note]. The gap comes from `cuda_buffer`'s double-buffered event synchronisation: `mark_device_ready()` calls `cudaEventSynchronize` to prevent event re-recording races, which blocks the CPU until the previous GPU kernel completes. This prevents pipelining of consecutive `work()` calls and bounds throughput by kernel latency rather than kernel throughput. For large FFTs (2^15+) where each kernel takes longer, this overhead is proportionally smaller and efficiency rises to 58-81%.
- **CuPy FFT** uses a cached `cupy.cuda.cufft.Plan1d` and writes directly into the output buffer, bypassing `cp.fft.fft()`'s per-call allocation and copy. This makes it essentially **identical to the C++ cuFFT block**: within 2% on the GB10 (8.8-9.1 Gsps) and within 1% on the RTX PRO 6000 (62-63 Gsps). Write your GPU blocks in Python with no performance penalty.
- **FFTW (CPU)** peaks at 0.46 Gsps (ARM Cortex-X925) and 1.02 Gsps (AMD EPYC 9335). cuFFT on the GB10 provides a **20-37x speedup** over single-threaded FFTW; the RTX PRO 6000 achieves **60-175x**.

> **Tip:** Run `benchmark_gr_cuda_fft.py --plot fft_bench.png` on your system to generate a comparison plot. Use `--csv results.csv` to save raw data.

[^note]: If 64 Gsps of FFTs isn't enough for your application, I'm prety sure you probably have bigger problems than GNURadio.
