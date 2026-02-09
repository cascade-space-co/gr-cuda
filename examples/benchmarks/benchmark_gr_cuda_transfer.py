#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Benchmark CUDA buffer transfer throughput.

Measures Host-to-Device (H2D), Device-to-Host (D2H), and full round-trip
transfer rates through the cuda_buffer pipeline for various buffer sizes
and numbers of parallel chains.

Usage:
    python benchmark_transfer.py                          # all modes, 1 chain
    python benchmark_transfer.py --chains 3               # 3 parallel chains
    python benchmark_transfer.py --mode h2d --chains 1    # H2D only
    python benchmark_transfer.py --buff-exp-min 18 --buff-exp-max 22
"""

import argparse
import time
from gnuradio import gr, blocks, cuda


def make_chain(mode, buff_len):
    """Build one transfer chain.

    Returns (blocks_list, connections, meter_block).
    meter_block is the block whose nitems_read(0) gives the item count.
    """
    if mode == "h2d":
        # CPU -> GPU: null_source writes into cuda_buffer, copy forces the DMA,
        # null_sink consumes on device.  Without the copy block the DMA event
        # synchronisation is not enforced and the counter races ahead.
        src = blocks.null_source(gr.sizeof_gr_complex)
        copy = cuda.copy(gr.sizeof_gr_complex, True)
        sink = cuda.null_sink(gr.sizeof_gr_complex)
        blk_list = [src, copy, sink]
        conns = [(src, 0, copy, 0), (copy, 0, sink, 0)]
        meter = sink

    elif mode == "d2h":
        # GPU -> CPU: null_source produces on device, copy forces the DMA,
        # CPU null_sink reads via cuda_buffer D2H path.
        src = cuda.null_source(gr.sizeof_gr_complex)
        copy = cuda.copy(gr.sizeof_gr_complex, True)
        sink = blocks.null_sink(gr.sizeof_gr_complex)
        blk_list = [src, copy, sink]
        conns = [(src, 0, copy, 0), (copy, 0, sink, 0)]
        meter = sink

    elif mode == "full":
        # Full round-trip: CPU -> H2D -> GPU copy -> D2H -> CPU
        src = blocks.null_source(gr.sizeof_gr_complex)
        copy = cuda.copy(gr.sizeof_gr_complex, True)
        sink = blocks.null_sink(gr.sizeof_gr_complex)
        blk_list = [src, copy, sink]
        conns = [(src, 0, copy, 0), (copy, 0, sink, 0)]
        meter = sink

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Large output_multiple forces the scheduler to issue big work() calls,
    # which translates to large DMA transfers -- critical for PCIe throughput.
    for b in blk_list:
        b.set_min_output_buffer(buff_len)
        b.set_output_multiple(buff_len)

    return blk_list, conns, meter


def run_benchmark(mode, num_chains, buff_len, duration, warmup=1.0):
    """Run a single benchmark configuration.

    Returns (total_items, elapsed_seconds).
    """
    tb = gr.top_block()
    meters = []
    all_blocks = []  # prevent garbage collection

    for _ in range(num_chains):
        blk_list, conns, meter = make_chain(mode, buff_len)
        all_blocks.extend(blk_list)
        meters.append(meter)
        for s, sp, d, dp in conns:
            tb.connect((s, sp), (d, dp))

    # Start and let the pipeline reach steady state
    tb.start()
    time.sleep(warmup)

    # Snapshot counters and measure over the requested duration
    n0 = [m.nitems_read(0) for m in meters]
    t0 = time.perf_counter()

    time.sleep(duration)

    t1 = time.perf_counter()
    n1 = [m.nitems_read(0) for m in meters]

    tb.stop()
    tb.wait()

    elapsed = t1 - t0
    total_items = sum(after - before for before, after in zip(n0, n1))
    return total_items, elapsed


def format_bytes(n):
    """Format byte count as a human-readable string."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.0f} MiB"
    if n >= 1024:
        return f"{n / 1024:.0f} KiB"
    return f"{n} B"


MODE_LABELS = {
    "h2d":  "CPU -> GPU  (Host-to-Device)",
    "d2h":  "GPU -> CPU  (Device-to-Host)",
    "full": "CPU -> GPU -> CPU  (full round-trip)",
}


def main():
    # Suppress GNU Radio info-level messages (e.g. set_min_output_buffer)
    gr.logging().set_default_level(gr.log_levels.warn)

    parser = argparse.ArgumentParser(
        description="Benchmark CUDA buffer transfer throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chains", type=int, default=1,
        help="Number of parallel chains (default: 1)")
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Measurement duration per config in seconds (default: 10)")
    parser.add_argument(
        "--warmup", type=float, default=1.0,
        help="Warmup before measuring in seconds (default: 1)")
    parser.add_argument(
        "--mode", choices=["h2d", "d2h", "full", "all"], default="all",
        help="Transfer direction to test (default: all)")
    parser.add_argument(
        "--buff-exp-min", type=int, default=16,
        help="Min buffer size as power of 2 (default: 16 = 64K items)")
    parser.add_argument(
        "--buff-exp-max", type=int, default=22,
        help="Max buffer size as power of 2 (default: 22 = 4M items)")
    args = parser.parse_args()

    modes = ["h2d", "d2h", "full"] if args.mode == "all" else [args.mode]
    buff_exps = range(args.buff_exp_min, args.buff_exp_max + 1)
    item_bytes = gr.sizeof_gr_complex

    print("CUDA buffer transfer benchmark")
    print(f"  Chains:       {args.chains}")
    print(f"  Duration:     {args.duration}s per measurement ({args.warmup}s warmup)")
    print(f"  Item size:    {item_bytes} bytes (gr_complex)")
    print(f"  Buffer sizes: 2^{args.buff_exp_min} .. 2^{args.buff_exp_max} items")
    print()

    for mode in modes:
        print(f"--- {MODE_LABELS[mode]} ---")
        print(f"  {'buf_items':<12s}  {'buf_bytes':>10s}"
              f"  {'total_items':<18s}  {'Gsps':<8s}  {'GB/s':<8s}")
        print(f"  {'─' * 12}  {'─' * 10}  {'─' * 18}  {'─' * 8}  {'─' * 8}")

        for exp in buff_exps:
            buff_len = 2 ** exp
            buf_bytes_str = format_bytes(buff_len * item_bytes)
            total_items, elapsed = run_benchmark(
                mode, args.chains, buff_len, args.duration, args.warmup)
            gsps = total_items / elapsed / 1e9
            gbps = gsps * item_bytes
            print(f"  2^{exp:<9d} {buf_bytes_str:>10s}"
                  f"  {total_items:>18,d}  {gsps:>8.2f}  {gbps:>8.2f}")
        print()

    print("Note: NVIDIA profiling tools (nsys, ncu) may report higher raw DMA")
    print("transfer rates. This benchmark measures end-to-end throughput including")
    print("GNU Radio scheduler overhead and PCIe round-trip latency.")


if __name__ == "__main__":
    main()
