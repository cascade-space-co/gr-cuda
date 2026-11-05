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

Buffer sizes are given in bytes, as powers of 2, and are the size of each
edge's ring.  Every work() call is a quarter of the ring (RING_CALLS), so
producer and consumer don't take turns on the same two halves.

The GPU block in the middle of each chain is a multiply by 1, so a kernel
reads and writes every sample, as a real GPU block would.  On integrated GPUs
with zero-copy edges there is no host/device copy, and this kernel is the real
cost.  On discrete GPUs it is far from VRAM bandwidth and overlaps the copies.
``--passthru nop`` uses a block that never touches its data instead, which
measures only the copies on discrete GPUs and nothing meaningful on zero-copy
edges.

Usage:
    benchmark_gr_cuda_transfer.py                          # all modes, 32 MiB-1 GiB
    benchmark_gr_cuda_transfer.py --chains 3               # 3 parallel chains
    benchmark_gr_cuda_transfer.py --mode h2d --chains 1    # H2D only
    benchmark_gr_cuda_transfer.py --buff-exp-min 21 --buff-exp-max 25  # 2-32 MiB
    benchmark_gr_cuda_transfer.py --passthru nop           # force the nop block
"""

import argparse
import os
import time

from gnuradio import blocks, cuda, gr

# Let set_min_output_buffer() alone size the rings, instead of the user's
# [cuda_buffer] min_buffer_bytes floor.  cuda_buffer reads it once, at its
# first allocation, and the environment overrides config files.
os.environ["GR_CONF_CUDA_BUFFER_MIN_BUFFER_BYTES"] = "1"

# work() calls each ring holds.
RING_CALLS = 4

# GNU Radio counts buffer items in an int.
MAX_RING_ITEMS = 2**31 - 1


def make_passthru(passthru):
    """GPU block in the middle of each chain."""
    if passthru == "kernel":
        return cuda.multiply_const_cc(1.0)
    return cuda.nop(gr.sizeof_gr_complex)


def make_chain(mode, ring_items, passthru_kind):
    """Build one transfer chain.

    Returns (blocks_list, connections, meter_block).
    meter_block is the block whose nitems_read(0) gives the item count.
    """
    if mode == "h2d":
        # CPU -> GPU: null_source writes into cuda_buffer, the GPU block reads
        # it, null_sink consumes on device.  Without a GPU block in between the
        # DMA event synchronisation is not enforced and the counter races ahead.
        src = cuda.host_null_source(gr.sizeof_gr_complex, memset=False)
        passthru = make_passthru(passthru_kind)
        sink = cuda.null_sink(gr.sizeof_gr_complex)
        blk_list = [src, passthru, sink]
        conns = [(src, 0, passthru, 0), (passthru, 0, sink, 0)]
        meter = sink

    elif mode == "d2h":
        # GPU -> CPU: null_source produces on device, the GPU block writes the
        # D2H buffer, CPU null_sink reads via cuda_buffer D2H path.
        src = cuda.null_source(gr.sizeof_gr_complex, memset=False)
        passthru = make_passthru(passthru_kind)
        sink = blocks.null_sink(gr.sizeof_gr_complex)
        blk_list = [src, passthru, sink]
        conns = [(src, 0, passthru, 0), (passthru, 0, sink, 0)]
        meter = sink

    elif mode == "full":
        # Full round-trip: CPU -> H2D -> GPU block -> D2H -> CPU
        src = cuda.host_null_source(gr.sizeof_gr_complex, memset=False)
        passthru = make_passthru(passthru_kind)
        sink = blocks.null_sink(gr.sizeof_gr_complex)
        blk_list = [src, passthru, sink]
        conns = [(src, 0, passthru, 0), (passthru, 0, sink, 0)]
        meter = sink

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Fixed-size work() calls: large enough to amortise the scheduler's
    # per-call overhead and, on PCIe, to make large DMA transfers.
    call_items = ring_items // RING_CALLS
    for b in blk_list:
        b.set_min_output_buffer(ring_items)
        b.set_output_multiple(call_items)
        b.set_max_noutput_items(call_items)

    return blk_list, conns, meter


def run_benchmark(mode, num_chains, ring_items, duration, passthru, warmup=1.0):
    """Run a single benchmark configuration.

    Returns (total_items, elapsed_seconds).
    """
    tb = gr.top_block()
    meters = []
    all_blocks = []  # prevent garbage collection

    for _ in range(num_chains):
        blk_list, conns, meter = make_chain(mode, ring_items, passthru)
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
    total_items = sum(after - before for before, after in zip(n0, n1, strict=True))
    return total_items, elapsed


def format_bytes(n):
    """Format byte count as a human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.0f} GiB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.0f} MiB"
    if n >= 1024:
        return f"{n / 1024:.0f} KiB"
    return f"{n} B"


PASSTHRU_LABELS = {
    "kernel": "multiply_const_cc(1), a kernel touches every sample",
    "nop": "nop, data is never touched",
}


def zero_copy_status():
    """cuda_buffer's zero-copy decision for the current device."""
    try:
        active = cuda.zero_copy_enabled()
    except Exception as e:
        return f"unknown ({e})"
    if active:
        return "yes (integrated GPU, H2D/D2H edges share host memory)"
    return "no (H2D/D2H edges copy between host and device)"


MODE_LABELS = {
    "h2d": "CPU -> GPU  (Host-to-Device)",
    "d2h": "GPU -> CPU  (Device-to-Host)",
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
        "--chains", type=int, default=1, help="Number of parallel chains (default: 1)"
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
        choices=["h2d", "d2h", "full", "all"],
        default="all",
        help="Transfer direction to test (default: all)",
    )
    parser.add_argument(
        "--buff-exp-min",
        type=int,
        default=25,
        help="Min ring size in bytes, as a power of 2 (default: 25 = 32 MiB, "
        "cuda_buffer's default ring floor)",
    )
    parser.add_argument(
        "--buff-exp-max",
        type=int,
        default=30,
        help="Max ring size in bytes, as a power of 2 (default: 30 = 1 GiB)",
    )
    parser.add_argument(
        "--passthru",
        choices=["kernel", "nop"],
        default="kernel",
        help="GPU block between source and sink: 'kernel' multiplies every "
        "sample by 1, 'nop' never touches its data (default: kernel)",
    )
    args = parser.parse_args()

    item_bytes = gr.sizeof_gr_complex
    if 2**args.buff_exp_min < RING_CALLS * item_bytes:
        parser.error(
            f"--buff-exp-min: the ring must hold at least {RING_CALLS} items"
        )
    if 2**args.buff_exp_max // item_bytes > MAX_RING_ITEMS:
        parser.error(
            f"--buff-exp-max: the ring must hold at most {MAX_RING_ITEMS:,} items"
        )

    passthru = args.passthru

    modes = ["h2d", "d2h", "full"] if args.mode == "all" else [args.mode]
    buff_exps = range(args.buff_exp_min, args.buff_exp_max + 1)

    print("CUDA buffer transfer benchmark")
    print(f"  Chains:       {args.chains}")
    print(f"  Duration:     {args.duration}s per measurement ({args.warmup}s warmup)")
    print(f"  Item size:    {item_bytes} bytes (gr_complex)")
    print(
        f"  Ring sizes:   {format_bytes(2**args.buff_exp_min)} .. "
        f"{format_bytes(2**args.buff_exp_max)}, {RING_CALLS} work() calls each"
    )
    print(f"  GPU block:    {PASSTHRU_LABELS[passthru]}")
    print(f"  Zero-copy:    {zero_copy_status()}")
    print()

    for mode in modes:
        print(f"--- {MODE_LABELS[mode]} ---")
        print(
            f"  {'ring_bytes':<12s}  {'ring':>10s}  {'call':>10s}"
            f"  {'total_items':<18s}  {'Gsps':<8s}  {'GB/s':<8s}"
        )
        print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 18}  {'─' * 8}  {'─' * 8}")

        for exp in buff_exps:
            ring_bytes = 2**exp
            ring_items = ring_bytes // item_bytes
            call_bytes = (ring_items // RING_CALLS) * item_bytes
            total_items, elapsed = run_benchmark(
                mode, args.chains, ring_items, args.duration, passthru, args.warmup
            )
            gsps = total_items / elapsed / 1e9
            gbps = gsps * item_bytes
            print(
                f"  2^{exp:<9d} {format_bytes(ring_bytes):>10s}"
                f"  {format_bytes(call_bytes):>10s}"
                f"  {total_items:>18,d}  {gsps:>8.2f}  {gbps:>8.2f}"
            )
        print()

    print("Note: NVIDIA profiling tools (nsys, ncu) may report higher raw DMA")
    print("transfer rates. This benchmark measures end-to-end throughput including")
    print("GNU Radio scheduler overhead and host/device transfer latency.")


if __name__ == "__main__":
    main()
