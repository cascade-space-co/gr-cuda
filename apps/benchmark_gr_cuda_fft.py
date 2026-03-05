#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Benchmark FFT throughput: cuFFT (GPU), CuPy FFT (GPU), and FFTW (CPU).

Measures FFTs/sec and Gsps for each engine across a range of FFT sizes.
GPU benchmarks run entirely on-device (GPU null_source -> FFT -> GPU null_sink)
to isolate compute performance from PCIe transfer overhead.

The output_multiple parameter controls how many FFT vectors the scheduler
batches into each work() call.  By default it is auto-scaled so that each
call processes ~64 MiB of data, keeping the comparison fair across engines.

Usage:
    python benchmark_gr_cuda_fft.py                        # all engines
    python benchmark_gr_cuda_fft.py --mode cufft           # cuFFT only
    python benchmark_gr_cuda_fft.py --mode cpu             # CPU only
    python benchmark_gr_cuda_fft.py --fft-exp-min 8 --fft-exp-max 20
    python benchmark_gr_cuda_fft.py --output-multiple 128  # override for all
    python benchmark_gr_cuda_fft.py --csv results.csv      # save CSV
    python benchmark_gr_cuda_fft.py --plot fft_bench.png   # save plot
"""

import argparse
import csv
import os
import time

from gnuradio import blocks, cuda, gr
from gnuradio import fft as gr_fft

# Each work() call processes ~WORK_BATCH_BYTES of data, which also determines
# the cuFFT batch size (vectors per call).  Adjusting this value (inherently
# the batch size of the cuFFT plan) might or might not produce better
# performance.  The buffer is BUFFER_MULTIPLE times larger to give the
# producer headroom over the consumer.
WORK_BATCH_BYTES = 64 * 1024 * 1024  # 64 MiB per work() call
BUFFER_MULTIPLE = 16  # buffer = 16 work() calls = ~1 GiB


def _auto_output_multiple(fft_size):
    """Compute output_multiple so total batch ~ WORK_BATCH_BYTES."""
    bytes_per_vector = fft_size * 8  # complex64
    return max(1, WORK_BATCH_BYTES // bytes_per_vector)


def make_chain(mode, fft_size, output_multiple):
    """Build one FFT benchmark chain.

    Returns (blocks_list, connections, meter_block).
    meter_block.nitems_read(0) gives the number of vectors processed.
    """
    vlen_bytes = gr.sizeof_gr_complex * fft_size

    if mode == "cufft":
        # GPU -> GPU: null_source produces zero vectors on device,
        # cuFFT processes them, null_sink consumes on device.
        # memset=False avoids saturating GPU memory bandwidth so the
        # benchmark measures FFT throughput, not memset contention.
        src = cuda.null_source(vlen_bytes, memset=False)
        fft_blk = cuda.fft(fft_size, forward=True)
        sink = cuda.null_sink(vlen_bytes)
        blk_list = [src, fft_blk, sink]
        conns = [(src, 0, fft_blk, 0), (fft_blk, 0, sink, 0)]
        meter = sink

    elif mode == "cupy":
        # GPU -> GPU: same chain but using the CuPy FFT block.
        src = cuda.null_source(vlen_bytes, memset=False)
        fft_blk = cuda.fft_cupy(fft_size, forward=True)
        sink = cuda.null_sink(vlen_bytes)
        blk_list = [src, fft_blk, sink]
        conns = [(src, 0, fft_blk, 0), (fft_blk, 0, sink, 0)]
        meter = sink

    elif mode == "cpu":
        # CPU -> CPU: null_source produces zero vectors,
        # GR's FFTW-based fft_vcc processes them, null_sink consumes.
        src = blocks.null_source(vlen_bytes)
        fft_blk = gr_fft.fft_vcc(fft_size, forward=True, window=[])
        sink = blocks.null_sink(vlen_bytes)
        blk_list = [src, fft_blk, sink]
        conns = [(src, 0, fft_blk, 0), (fft_blk, 0, sink, 0)]
        meter = sink

    else:
        raise ValueError(f"Unknown mode: {mode}")

    for b in blk_list:
        b.set_min_output_buffer(BUFFER_MULTIPLE * output_multiple)
        b.set_output_multiple(output_multiple)
        b.set_max_noutput_items(output_multiple)

    return blk_list, conns, meter


def run_benchmark(mode, fft_size, output_multiple, duration, warmup=1.0):
    """Run a single FFT benchmark and return (total_vectors, elapsed_seconds)."""
    tb = gr.top_block()
    all_blocks = []

    blk_list, conns, meter = make_chain(mode, fft_size, output_multiple)
    all_blocks.extend(blk_list)
    for s, sp, d, dp in conns:
        tb.connect((s, sp), (d, dp))

    tb.start()
    time.sleep(warmup)

    n0 = meter.nitems_read(0)
    t0 = time.perf_counter()

    time.sleep(duration)

    t1 = time.perf_counter()
    n1 = meter.nitems_read(0)

    tb.stop()
    tb.wait()

    elapsed = t1 - t0
    total_vectors = n1 - n0

    if mode == "cupy":
        import cupy

        # Free all CuPy memory blocks to avoid memory leaks.
        cupy.get_default_memory_pool().free_all_blocks()

    return total_vectors, elapsed


MODE_LABELS = {
    "cufft": "cuFFT  (GPU, C++)",
    "cupy": "CuPy FFT  (GPU, Python)",
    "cpu": "FFTW  (CPU)",
}

ALL_MODES = ["cufft", "cupy", "cpu"]


def main():
    gr.logging().set_default_level(gr.log_levels.warn)

    parser = argparse.ArgumentParser(
        description="Benchmark FFT throughput: cuFFT, CuPy, and CPU (FFTW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Measurement duration per config in seconds (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Warmup before measuring in seconds (default: 1)",
    )
    parser.add_argument(
        "--mode",
        choices=ALL_MODES + ["all"],
        default="all",
        help="FFT engine to test (default: all)",
    )
    parser.add_argument(
        "--fft-exp-min",
        type=int,
        default=8,
        help="Min FFT size as power of 2 (default: 8 = 256 points)",
    )
    parser.add_argument(
        "--fft-exp-max",
        type=int,
        default=20,
        help="Max FFT size as power of 2 (default: 20 = 1M points)",
    )
    parser.add_argument(
        "--output-multiple",
        type=int,
        default=None,
        help="output_multiple for scheduler (vectors per work call). "
        "Default: auto-scaled to ~64 MiB/call for all engines",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="FILE",
        help="Write results to a CSV file",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="FILE",
        help="Save a benchmark plot (PNG) to the given path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the plot and CSV comment (default: GPU device name)",
    )
    args = parser.parse_args()

    modes = ALL_MODES if args.mode == "all" else [args.mode]
    fft_exps = list(range(args.fft_exp_min, args.fft_exp_max + 1))

    # Resolve title (default: GPU name from CUDA runtime)
    title = args.title
    if title is None:
        try:
            import pycuda.driver as drv

            title = drv.Device(0).name()
        except Exception:
            import socket

            title = socket.gethostname()

    print("FFT benchmark")
    print(f"  Device:          {title}")
    print(
        f"  Duration:        {args.duration}s per measurement ({args.warmup}s warmup)"
    )
    if args.output_multiple is not None:
        print(f"  Output multiple: {args.output_multiple} vectors per work() call")
    else:
        print(
            f"  Output multiple: auto (~{WORK_BATCH_BYTES // (1024 * 1024)} MiB/call)"
        )
    print(f"  FFT sizes:       2^{args.fft_exp_min} .. 2^{args.fft_exp_max} points")
    print()

    # Collect all results for CSV and plotting:
    #   results[mode] = [(fft_exp, fft_size, om, total, ffts_per_sec, gsps), ...]
    results = {}

    for mode in modes:
        results[mode] = []
        print(f"--- {MODE_LABELS[mode]} ---")
        print(
            f"  {'fft_size':<12s}  {'out_mult':>8s}  {'total_ffts':<18s}"
            f"  {'FFTs/s':<12s}  {'Gsps':<8s}"
        )
        print(f"  {'─' * 12}  {'─' * 8}  {'─' * 18}  {'─' * 12}  {'─' * 8}")

        for exp in fft_exps:
            fft_size = 2**exp
            if args.output_multiple is not None:
                om = args.output_multiple
            else:
                om = _auto_output_multiple(fft_size)

            total_vectors, elapsed = run_benchmark(
                mode, fft_size, om, args.duration, args.warmup
            )
            ffts_per_sec = total_vectors / elapsed
            gsps = (total_vectors * fft_size) / elapsed / 1e9
            print(
                f"  2^{exp:<9d} {om:>8d}  {total_vectors:>18,d}"
                f"  {ffts_per_sec:>12,.0f}  {gsps:>8.2f}"
            )

            results[mode].append((exp, fft_size, om, total_vectors, ffts_per_sec, gsps))
        print()

    # ---- CSV output ----
    if args.csv:
        write_header = not os.path.isfile(args.csv)
        with open(args.csv, "a", newline="") as f:
            if write_header:
                f.write(f"# {title}\n")
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "engine",
                        "fft_exp",
                        "fft_size",
                        "output_multiple",
                        "total_ffts",
                        "ffts_per_sec",
                        "gsps",
                    ]
                )
            for mode in modes:
                for exp, fft_size, om, total, fps, gsps in results[mode]:
                    writer.writerow(
                        [mode, exp, fft_size, om, total, f"{fps:.2f}", f"{gsps:.4f}"]
                    )
        print(f"Results written to {args.csv}")

    # ---- Plot ----
    if args.plot:
        _generate_plot(results, fft_exps, title, args.plot)


def _generate_plot(results, fft_exps, title, plot_path):
    """Generate a 2-subplot benchmark plot and save as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot generation.")
        return

    fig, (ax_gsps, ax_ffts) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    style_map = {
        "cufft": {
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
            "label": "cuFFT (GPU, C++)",
        },
        "cupy": {
            "color": "#ff7f0e",
            "linestyle": "-",
            "marker": "^",
            "label": "CuPy FFT (GPU, Python)",
        },
        "cpu": {
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "D",
            "label": "FFTW (CPU)",
        },
    }

    for mode, rows in results.items():
        if not rows:
            continue
        exps = [r[0] for r in rows]
        gsps = [r[5] for r in rows]
        ffts = [r[4] for r in rows]
        s = style_map.get(
            mode, {"color": "black", "linestyle": "-", "marker": "x", "label": mode}
        )
        ax_gsps.plot(
            exps,
            gsps,
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"],
            label=s["label"],
            markersize=5,
        )
        ax_ffts.plot(
            exps,
            ffts,
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"],
            label=s["label"],
            markersize=5,
        )

    # Top subplot: Gsps
    ax_gsps.set_ylabel("Throughput (Gsps)")
    ax_gsps.set_title(f"FFT Benchmark — {title}")
    ax_gsps.legend(fontsize=9)
    ax_gsps.grid(True, alpha=0.3)
    ax_gsps.set_xlim(fft_exps[0], fft_exps[-1])

    # Bottom subplot: FFTs/s (log scale)
    ax_ffts.set_ylabel("FFTs/s")
    ax_ffts.set_xlabel("FFT size (log₂ N)")
    ax_ffts.set_yscale("log")
    ax_ffts.legend(fontsize=9)
    ax_ffts.grid(True, alpha=0.3, which="both")
    ax_ffts.set_xlim(fft_exps[0], fft_exps[-1])

    # X-axis ticks as 2^N labels
    ax_ffts.set_xticks(fft_exps)
    ax_ffts.set_xticklabels([f"2^{e}" for e in fft_exps], fontsize=8, rotation=45)

    fig.tight_layout()

    # Create output directory if needed
    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
