#!/usr/bin/env python3
"""
IBV sequence-counter test.

TX: null_source -> seq_stamp -> ibv_sink   (+ probe_rate off stamp)
RX: ibv_source  -> seq_strip -> seq_checker   (+ probe_rate off source)

seq_stamp    writes a little-endian uint64 counter into the first
             8 bytes of each packet (C++ CUDA block, operates on GPU).
seq_strip    extracts those 8 bytes back to CPU as uint64 values
             (C++ block, GPU->CPU).
seq_checker  pure-Python sink that checks the counter sequence and
             reports drops / duplicates / reordering.

Usage (two machines, or loopback):
  machine A:  sudo python3 apps/test_ibv_seq.py --mode tx ...
  machine B:  sudo python3 apps/test_ibv_seq.py --mode rx ...
"""

import argparse
import time

import numpy as np
from gnuradio import blocks, cuda, gr

DEFAULT_PAYLOAD_SIZE = 8000


# ---------------------------------------------------------------------------
#  Python-side sequence checker (pure CPU, receives uint64 from seq_check)
# ---------------------------------------------------------------------------


class seq_checker(gr.sync_block):
    """Check an incrementing uint64 stream for drops, duplicates, reorder."""

    def __init__(self, report_interval=2.0):
        gr.sync_block.__init__(
            self,
            name="seq_checker",
            in_sig=[np.uint64],
            out_sig=None,
        )
        self._expected = None
        self._total = 0
        self._drops = 0
        self._duplicates = 0
        self._reorders = 0
        self._max_seen = -1
        self._interval = report_interval
        self._last_report = time.monotonic()

    def work(self, input_items, output_items):
        counters = input_items[0]

        for c_np in counters:
            c = int(c_np)
            self._total += 1

            if self._expected is None:
                self._expected = c

            if c == self._expected:
                self._expected = c + 1
            elif c > self._expected:
                self._drops += c - self._expected
                self._expected = c + 1
            else:
                if c <= self._max_seen:
                    self._duplicates += 1
                else:
                    self._reorders += 1
                self._expected = max(self._expected, c + 1)

            if c > self._max_seen:
                self._max_seen = c

        now = time.monotonic()
        if now - self._last_report >= self._interval:
            self._print_stats()
            self._last_report = now

        return len(counters)

    def _print_stats(self):
        total_expected = self._max_seen + 1 if self._max_seen >= 0 else 0
        loss = (self._drops / total_expected * 100) if total_expected > 0 else 0
        print(
            f"  SEQ: rx={self._total}  drops={self._drops} ({loss:.2f}%)  "
            f"dups={self._duplicates}  reorder={self._reorders}  "
            f"last={self._max_seen}"
        )

    def print_summary(self):
        total_expected = self._max_seen + 1 if self._max_seen >= 0 else 0
        loss = (self._drops / total_expected * 100) if total_expected > 0 else 0
        print(f"\n{'=' * 60}")
        print("  Sequence Checker Summary")
        print(f"    Received:    {self._total}")
        print(f"    Drops:       {self._drops}  ({loss:.2f}%)")
        print(f"    Duplicates:  {self._duplicates}")
        print(f"    Reorders:    {self._reorders}")
        print(f"    Counter max: {self._max_seen}")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
#  Flowgraphs
# ---------------------------------------------------------------------------


class ibv_seq_tx(gr.top_block):
    def __init__(self, ibv_dev, dst_ip, dst_port, payload_size, dst_mac, mcast):
        super().__init__("ibv_seq_tx")

        self.src = cuda.null_source(payload_size, memset=False)
        self.stamp = cuda.seq_stamp(payload_size)
        self.snk = cuda.ibv_sink(
            ibv_dev,
            dst_ip,
            dst_port,
            payload_size,
            dst_mac,
            mcast,
        )
        self.probe = cuda.probe_rate(payload_size, 1000.0, 0.15, "TX")
        self.dbg = blocks.message_debug(False)

        self.connect(self.src, self.stamp, self.snk)
        self.connect(self.stamp, self.probe)
        self.msg_connect(self.probe, "rate", self.dbg, "print")


class ibv_seq_rx(gr.top_block):
    def __init__(self, ibv_dev, port, payload_size, mcast):
        super().__init__("ibv_seq_rx")

        self.src = cuda.ibv_source(ibv_dev, port, payload_size, mcast)
        self.strip = cuda.seq_strip(payload_size)
        self.checker = seq_checker()
        self.probe = cuda.probe_rate(payload_size, 1000.0, 0.15, "RX")
        self.dbg = blocks.message_debug(False)

        self.connect(self.src, self.strip, self.checker)
        self.connect(self.src, self.probe)
        # self.msg_connect(self.probe, "rate", self.dbg, "print")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="IBV sequence-counter test")
    parser.add_argument("--mode", choices=["rx", "tx", "both"], default="both")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--payload-size", type=int, default=DEFAULT_PAYLOAD_SIZE)
    parser.add_argument("--udp-port", type=int, default=5000)

    parser.add_argument("--tx-ibv-dev", default="rocep119s0f1")
    parser.add_argument("--tx-dst-ip", default="172.16.1.21")
    parser.add_argument("--tx-dst-mac", default="b8:e9:24:b9:87:6c")

    parser.add_argument("--rx-ibv-dev", default="rocep119s0f0")

    parser.add_argument("--mcast", default="")

    args = parser.parse_args()
    flowgraphs = []
    checker = None

    if args.mode in ("rx", "both"):
        rx = ibv_seq_rx(
            args.rx_ibv_dev,
            args.udp_port,
            args.payload_size,
            args.mcast,
        )
        checker = rx.checker
        flowgraphs.append(("RX", rx))

    if args.mode in ("tx", "both"):
        tx = ibv_seq_tx(
            args.tx_ibv_dev,
            args.tx_dst_ip,
            args.udp_port,
            args.payload_size,
            args.tx_dst_mac,
            args.mcast,
        )
        flowgraphs.append(("TX", tx))

    for name, fg in flowgraphs:
        print(f"Starting {name} flowgraph...")
        fg.start()
        if name == "RX" and args.mode == "both":
            time.sleep(1)

    print(f"Running for {args.duration}s (Ctrl-C to stop early)...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    for name, fg in flowgraphs:
        print(f"Stopping {name}...")
        fg.stop()
        fg.wait()

    if checker is not None:
        checker.print_summary()

    print("Done.")


if __name__ == "__main__":
    main()
