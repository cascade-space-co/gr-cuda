#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
IBV link diagnostic -- sequence-counter loopback / two-machine test.

NOT a unit test.  This is a hands-on diagnostic for verifying that
the IBV path (NIC + GPUDirect RDMA) can carry traffic without drops,
duplicates, or reordering.  Run it after standing up a new host or
after physical re-cabling, before chasing bugs in higher-level
flowgraphs that build on `cuda.ibv_source` / `cuda.ibv_sink`.

Pipeline:
  TX: null_source -> seq_stamp -> ibv_sink     (+ probe_rate off stamp)
  RX: ibv_source  -> seq_strip -> seq_checker  (+ probe_rate off source)

  seq_stamp    prepends a little-endian uint64 counter (8-byte header)
               to each packet, growing it by 8 bytes (CUDA block, GPU).
  seq_strip    recovers that 8-byte header back to CPU as uint64 values
               (CUDA block, GPU->CPU); the payload is dropped.
  seq_checker  pure-Python sink that checks the counter sequence and
               reports drops / duplicates / reordering.

Healthy output is `drops=0  dups=0  reorder=0` over the whole run.

You'll need two pieces of host-specific info to invoke this script:
  - IBV device names: list them with `ibv_devices` (or `ibv_devinfo`
    for richer per-device output).
  - Destination MAC address: on the receiver host, read the NIC's
    MAC from `ip addr show <netdev>` (the `link/ether` line).  Pass
    it as `--tx-dst-mac` on the sender.

Usage (two machines, or single-NIC loopback):
  Loopback on one host (uses NIC-internal loopback):
    sudo python3 apps/ibv_link_check.py --mode both \\
        --tx-ibv-dev mlx5_0 --rx-ibv-dev mlx5_0 \\
        --tx-dst-mac aa:bb:cc:dd:ee:ff --duration 10

  Two-machine test:
    machine A:  sudo python3 apps/ibv_link_check.py --mode tx ...
    machine B:  sudo python3 apps/ibv_link_check.py --mode rx ...

This tool is only built/installed when gr-cuda was configured with
IBV blocks enabled (`-DENABLE_IBV=ON`, on by default if libibverbs
is detected at configure time).
"""

import argparse
import sys
import threading
import time

import cupy as cp
import numpy as np
from gnuradio import cuda, gr

DEFAULT_PAYLOAD_SIZE = 8000
SEQ_HDR_BYTES = 8  # seq_stamp prepends an 8-byte sequence number to each packet

# seq_strip's sequence-number output items are only 8 bytes, so without a hint
# the scheduler can hand the GPU checker many tiny batches (lots of small
# strided copies + small CuPy reductions).  Request a large output buffer on
# that edge so the checker processes big chunks instead.  set_min_output_buffer
# takes a count of items; cuda_buffer still enforces its own 32 MB floor.
RX_SEQ_BUF_ITEMS = (64 * 1024 * 1024) // SEQ_HDR_BYTES  # ~64 MB of uint64s


# ---------------------------------------------------------------------------
#  GPU sequence checker (CuPy, receives uint64 cuda_buffer from seq_strip)
# ---------------------------------------------------------------------------


class seq_checker(cuda.sync_block):
    """Check an incrementing uint64 stream for drops, duplicates, reorder.

    GPU sink: seq_strip hands the recovered sequence numbers over a
    cuda_buffer, so each batch arrives as a CuPy array already resident on
    the device.  The per-batch diff/reductions run on the GPU and only a
    handful of scalars are pulled back to the host per call (rather than the
    whole batch), which keeps the checker off the PCIe-bandwidth critical
    path at 100 GbE rates.  The periodic stdout printing is handled by the
    main thread (see `printer_thread` in main()).
    """

    def __init__(self):
        cuda.sync_block.__init__(self, "seq_checker", [np.uint64], None)
        self._expected = None
        self._total = 0
        self._drops = 0
        self._duplicates = 0
        self._reorders = 0
        self._max_seen = -1

    def work(self, input_items, output_items):
        counters = input_items[0]  # CuPy uint64 array (device memory)
        n = int(counters.size)
        if n == 0:
            return 0

        # Reductions/diffs stay on the GPU; only the few scalars below are
        # copied back to the host.
        c_first = int(counters[0])
        c_last = int(counters[-1])
        c_max = int(counters.max())
        self._total += n

        if self._expected is None:
            self._expected = c_first

        # Gap from the previous batch boundary to the first item of
        # this batch.
        if c_first > self._expected:
            self._drops += c_first - self._expected
        elif c_first < self._expected:
            # Out-of-order across batch boundary; rare on a healthy
            # link, count conservatively.
            if c_first <= self._max_seen:
                self._duplicates += 1
            else:
                self._reorders += 1

        # Within-batch deltas: delta == 1 is normal, > 1 is a drop,
        # <= 0 is a dup or reorder.  cp.diff returns int64 deltas.
        if n > 1:
            deltas = cp.diff(counters.astype(cp.int64))
            # Each gap of size k contributes (k-1) missed packets.
            self._drops += int(cp.maximum(deltas - 1, 0).sum())
            # Approximate split between dups (delta == 0) and
            # reorders (delta < 0).
            self._duplicates += int((deltas == 0).sum())
            self._reorders += int((deltas < 0).sum())

        self._expected = c_last + 1
        if c_max > self._max_seen:
            self._max_seen = c_max

        return n

    def stats(self):
        """Return (total, drops, duplicates, reorders, max_seen)."""
        return (
            self._total,
            self._drops,
            self._duplicates,
            self._reorders,
            self._max_seen,
        )

    def print_summary(self):
        total_expected = self._max_seen + 1 if self._max_seen >= 0 else 0
        loss = (self._drops / total_expected * 100) if total_expected > 0 else 0
        print(f"\n{'=' * 60}")
        print("  Sequence Checker Summary")
        print(f"    Received:    {self._total}")
        print(f"    Drops:       {self._drops}  ({loss:.2f}%)")
        print(f"    Counter max: {self._max_seen}")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
#  Flowgraphs
# ---------------------------------------------------------------------------


class ibv_seq_tx(gr.top_block):
    def __init__(self, ibv_dev, dst_ip, dst_port, payload_size, dst_mac, mcast):
        super().__init__("ibv_seq_tx")

        self.payload_size = payload_size
        # seq_stamp appends an 8-byte sequence header, so the on-wire frame
        # is 8 bytes larger than the data payload.
        frame_size = payload_size + SEQ_HDR_BYTES
        self.src = cuda.null_source(payload_size, memset=False)
        self.stamp = cuda.seq_stamp(payload_size)
        self.snk = cuda.ibv_sink(
            ibv_dev,
            dst_ip,
            dst_port,
            frame_size,
            dst_mac,
            mcast,
        )
        # probe_rate is a passive tap in the GPU pipeline; we read
        # .rate() (items/sec) from the main thread and convert to Gbps.
        # alpha=1.0 disables the EMA filter (default 0.0001 has a
        # ~5000 s time constant which never converges in our short
        # diagnostic runs); update_rate_ms=200 keeps the readout
        # responsive.  It taps the post-stamp stream, so frame_size items.
        self.probe = cuda.probe_rate(frame_size, update_rate_ms=200.0, alpha=1.0)

        self.connect(self.src, self.stamp, self.snk)
        self.connect(self.stamp, self.probe)


class ibv_seq_rx(gr.top_block):
    def __init__(self, ibv_dev, port, payload_size, mcast):
        super().__init__("ibv_seq_rx")

        self.payload_size = payload_size
        # On-wire frame carries the data payload plus the 8-byte sequence
        # header; seq_strip takes the matching data payload size.
        frame_size = payload_size + SEQ_HDR_BYTES
        self.src = cuda.ibv_source(ibv_dev, port, frame_size, mcast)
        self.strip = cuda.seq_strip(payload_size)
        # Give the checker large batches off seq_strip's 8-byte uint64 output.
        self.strip.set_min_output_buffer(0, RX_SEQ_BUF_ITEMS)
        self.checker = seq_checker()
        self.probe = cuda.probe_rate(frame_size, update_rate_ms=200.0, alpha=1.0)

        self.connect(self.src, self.strip, self.checker)
        self.connect(self.src, self.probe)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def _rate_to_gbps(items_per_sec, item_bytes):
    return items_per_sec * item_bytes * 8.0 / 1e9


def _format_status_lines(tx_fg, rx_fg, frame_size):
    """Return a list of status lines for whichever flowgraph(s) are
    running -- one line for TX, one for RX (so the printer can show
    them stacked when --mode=both rather than wrapping a long line).

    `frame_size` is the on-wire packet size (data payload + 8-byte
    sequence header), which is what the probes count."""
    lines = []
    if tx_fg is not None:
        tx_pps = tx_fg.probe.rate()
        tx_gbps = _rate_to_gbps(tx_pps, frame_size)
        lines.append(f"TX: {tx_gbps:7.3f} Gbps ({tx_pps / 1e6:5.2f} Mpkts/s)")
    if rx_fg is not None:
        rx_pps = rx_fg.probe.rate()
        rx_gbps = _rate_to_gbps(rx_pps, frame_size)
        total, drops, _dups, _reorders, max_seen = rx_fg.checker.stats()
        total_expected = max_seen + 1 if max_seen >= 0 else 0
        loss = (drops / total_expected * 100) if total_expected > 0 else 0
        lines.append(
            f"RX: {rx_gbps:7.3f} Gbps ({rx_pps / 1e6:5.2f} Mpkts/s)  "
            f"rx={total}  drops={drops} ({loss:.2f}%)"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="IBV link diagnostic (sequence-counter loopback / two-machine test)"
    )
    parser.add_argument("--mode", choices=["rx", "tx", "both"], default="both")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--payload-size",
        type=int,
        default=DEFAULT_PAYLOAD_SIZE,
        help="data payload size in bytes; the on-wire frame is 8 bytes larger "
        "(seq_stamp prepends an 8-byte sequence header)",
    )
    parser.add_argument(
        "--tx-udp-port", type=int, default=5000, help="UDP destination port for TX"
    )
    parser.add_argument(
        "--rx-udp-port",
        type=int,
        default=5000,
        help="UDP destination port for RX flow-steering",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=2.0,
        help="Seconds between in-place status updates",
    )

    parser.add_argument("--tx-ibv-dev", default="rocep119s0f1")
    parser.add_argument("--tx-dst-ip", default="172.16.1.21")
    parser.add_argument("--tx-dst-mac", default="b8:e9:24:b9:87:6c")

    parser.add_argument("--rx-ibv-dev", default="rocep119s0f0")

    parser.add_argument("--mcast", default="")

    args = parser.parse_args()

    if args.payload_size < 1:
        parser.error("--payload-size must be >= 1")

    flowgraphs = []
    rx_fg = None
    tx_fg = None

    if args.mode in ("rx", "both"):
        rx_fg = ibv_seq_rx(
            args.rx_ibv_dev,
            args.rx_udp_port,
            args.payload_size,
            args.mcast,
        )
        flowgraphs.append(("RX", rx_fg))

    if args.mode in ("tx", "both"):
        tx_fg = ibv_seq_tx(
            args.tx_ibv_dev,
            args.tx_dst_ip,
            args.tx_udp_port,
            args.payload_size,
            args.tx_dst_mac,
            args.mcast,
        )
        flowgraphs.append(("TX", tx_fg))

    for name, fg in flowgraphs:
        print(f"Starting {name} flowgraph...")
        fg.start()
        if name == "RX" and args.mode == "both":
            time.sleep(1)

    print(f"Running for {args.duration}s (Ctrl-C to stop early).")

    # Periodic status printer -- rewrites a fixed block of N lines in
    # place (one line per direction) so --mode=both doesn't wrap.
    stop_print = threading.Event()

    def printer_thread():
        printed = 0  # how many lines we wrote on the previous tick
        while not stop_print.is_set():
            lines = _format_status_lines(
                tx_fg, rx_fg, args.payload_size + SEQ_HDR_BYTES
            )
            # Move cursor up to the first line we previously wrote so
            # we can overwrite it; if this is the first tick, no-op.
            if printed > 0:
                sys.stdout.write(f"\033[{printed}A")
            for ln in lines:
                sys.stdout.write("\r\033[K" + ln + "\n")
            sys.stdout.flush()
            printed = len(lines)
            stop_print.wait(args.report_interval)

    pt = threading.Thread(target=printer_thread, daemon=True)
    pt.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        sys.stdout.write("\nInterrupted.\n")

    stop_print.set()
    pt.join(timeout=1.0)

    for name, fg in flowgraphs:
        print(f"Stopping {name}...")
        fg.stop()
        fg.wait()

    if rx_fg is not None:
        rx_fg.checker.print_summary()

    print("Done.")


if __name__ == "__main__":
    main()
