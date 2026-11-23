#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Every edge type carries data across many laps of its ring.

A cuda_buffer is a double-mapped ring of N items: the 2N address range maps
the same memory twice, so a block's range can start before N and run past it
into the alias of the first half.  Host-side copies into and out of the
device ring are split at N; device-side reads, writes and kernels rely on
the alias (as do all accesses on integrated GPUs, where H2D and D2H edges
hand blocks pointers into the host ring).

Each test pushes several rings' worth of an index ramp through one kind of
edge or reader and checks every sample.  GPU blocks run with prime call
sizes, so their ranges drift across N instead of lining up with it.
"""

import os
import threading

import cupy as cp
import numpy as np
from gnuradio import blocks, gr, gr_unittest

try:
    from gnuradio import cuda
except ImportError:
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio import cuda

ITEM = np.dtype(np.complex64)
RING_ITEMS = (
    gr.prefs().get_long("cuda_buffer", "min_buffer_bytes", 32 << 20) // ITEM.itemsize
)

# Primes, so neither the pattern nor the GPU blocks' calls line up with the
# ring.  Calls stay below half a ring, where the scheduler would cap them.
PATTERN_LEN = 1_000_003
CALL_ITEMS = 100_003
OTHER_CALL_ITEMS = 65_537
TOTAL_ITEMS = 4 * RING_ITEMS + PATTERN_LEN
TIMEOUT_S = 60

_pattern = None


def ramp(start, n):
    """Items start..start+n-1 of the pattern: k - 1j*k for k = index mod PATTERN_LEN."""
    k = ((start + np.arange(n)) % PATTERN_LEN).astype(np.float32)
    return (k - 1j * k).astype(ITEM)


def pattern_source():
    global _pattern
    if _pattern is None:
        _pattern = ramp(0, PATTERN_LEN).tolist()
    return blocks.vector_source_c(_pattern, repeat=True)


def check(errors, start, got, scale=1):
    """Record the first mismatch of ``got`` against items start.. of scale * ramp."""
    bad = np.flatnonzero(got != scale * ramp(start, len(got)))
    if bad.size and len(errors) < 5:
        errors.append((start + int(bad[0]), complex(got[bad[0]]), bad.size))


class checker(gr.sync_block):
    """CPU sink that compares every sample with ``scale * ramp``.

    With ``hist > 1`` it also checks the history items each call sees, which
    the scheduler hands over as a range starting before the read pointer.
    Mismatches are recorded rather than raised: an exception would stop only
    this block's thread and could leave the flowgraph hanging.
    """

    def __init__(self, scale=1, hist=1, delay=0):
        gr.sync_block.__init__(self, "checker", in_sig=[ITEM], out_sig=[])
        self.set_history(hist)
        if delay:
            self.declare_sample_delay(delay)
        self.scale = scale
        self.count = 0
        self.errors = []

    def work(self, input_items, output_items):
        h = self.history() - 1
        inp = input_items[0]
        if self.count >= h:
            check(self.errors, self.count - h, inp, self.scale)
        else:
            check(self.errors, self.count, inp[h:], self.scale)
        self.count += len(inp) - h
        return len(inp) - h


class demanding_checker(gr.basic_block):
    """CPU sink that asks for more items than it consumes.

    Each call needs ``demand`` items available but consumes only ``consume``,
    so its read windows straddle N on a different schedule from the writer's.
    """

    def __init__(self, demand=7919, consume=4999):
        gr.basic_block.__init__(self, "demanding_checker", in_sig=[ITEM], out_sig=[])
        self.demand = demand
        self.max_consume = consume
        self.count = 0
        self.errors = []

    def forecast(self, noutput_items, ninputs):
        return [self.demand] * ninputs

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        check(self.errors, self.count, inp)
        n = min(len(inp), self.max_consume)
        self.consume(0, n)
        self.count += n
        return 0


class gpu_passthrough_gw(cuda.basic_block):
    """Python general_work GPU block that copies with cupy, capped per call."""

    def __init__(self, max_items):
        cuda.basic_block.__init__(self, "gpu_passthrough_gw", [ITEM], [ITEM])
        self.max_items = max_items

    def forecast(self, noutput_items, ninputs):
        return [noutput_items] * ninputs

    def general_work(self, input_items, output_items):
        n = min(len(input_items[0]), len(output_items[0]), self.max_items)
        cp.copyto(output_items[0][:n], input_items[0][:n])
        self.consume_each(n)
        return n


def gpu(block, call_items=CALL_ITEMS):
    block.set_max_noutput_items(call_items)
    return block


class qa_buffer_wrap(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()
        self.src = pattern_source()
        self.head = blocks.head(ITEM.itemsize, TOTAL_ITEMS)
        self.tb.connect(self.src, self.head)

    def tearDown(self):
        self.tb = None

    def run_tb(self):
        # A failed CUDA call kills only one block's thread, which can leave
        # tb.run() waiting forever.
        runner = threading.Thread(target=self.tb.run)
        runner.start()
        runner.join(TIMEOUT_S)
        if runner.is_alive():
            self.tb.stop()
            runner.join()
            self.fail(f"flowgraph did not finish within {TIMEOUT_S} s")

    def assertChecked(self, sink, expected_count=TOTAL_ITEMS):
        self.assertEqual(sink.errors, [], "(first bad index, value, bad in call)")
        self.assertEqual(sink.count, expected_count)

    def test_001_copy(self):
        """H2D and D2H: cudaMemcpyAsync on ranges that run past N."""
        sink = checker()
        self.tb.connect(self.head, gpu(cuda.copy(ITEM.itemsize)), sink)
        self.run_tb()
        self.assertChecked(sink)

    def test_002_kernel(self):
        """A kernel reading and writing ranges that run past N."""
        sink = checker(scale=2)
        self.tb.connect(self.head, gpu(cuda.multiply_const_cc(2)), sink)
        self.run_tb()
        self.assertChecked(sink)

    def test_003_d2d_chain(self):
        """D2D edges between GPU blocks with different call sizes."""
        sink = checker(scale=2)
        self.tb.connect(
            self.head,
            gpu(cuda.copy(ITEM.itemsize)),
            gpu(cuda.multiply_const_cc(2), OTHER_CALL_ITEMS),
            gpu(cuda.copy(ITEM.itemsize)),
            sink,
        )
        self.run_tb()
        self.assertChecked(sink)

    def test_004_fanout_gpu_and_cpu(self):
        """A GPU tee feeding a GPU reader and, over D2H, a CPU reader."""
        tee = gpu(cuda.tee(ITEM.itemsize, gpu=True))
        gpu_sink = checker(scale=2)
        cpu_sink = checker()
        self.tb.connect(self.head, gpu(cuda.copy(ITEM.itemsize)), tee)
        self.tb.connect(
            (tee, 0), gpu(cuda.multiply_const_cc(2), OTHER_CALL_ITEMS), gpu_sink
        )
        self.tb.connect((tee, 1), cpu_sink)
        self.run_tb()
        self.assertChecked(gpu_sink)
        self.assertChecked(cpu_sink)

    def test_005_fanin(self):
        """Two GPU readers of one H2D edge, drifting apart, summed by a GPU block."""
        add = gpu(cuda.add_cc(2))
        sink = checker(scale=2)
        self.tb.connect(self.head, gpu(cuda.copy(ITEM.itemsize)), (add, 0))
        self.tb.connect(
            self.head, gpu(cuda.copy(ITEM.itemsize), OTHER_CALL_ITEMS), (add, 1)
        )
        self.tb.connect(add, sink)
        self.run_tb()
        self.assertChecked(sink)

    def test_006_history_readers(self):
        """CPU readers with history on one D2H edge.

        history=2 with a sample delay of 1 is how qtgui.time_sink_c reads;
        it used to deadlock cuda_buffer when it was single-mapped.
        """
        like_time_sink = checker(hist=2, delay=1)
        long_history = checker(hist=257)
        copy = gpu(cuda.copy(ITEM.itemsize))
        self.tb.connect(self.head, copy, like_time_sink)
        self.tb.connect(copy, long_history)
        self.run_tb()
        self.assertChecked(like_time_sink)
        self.assertChecked(long_history)

    def test_007_forecast_reader(self):
        """A CPU reader on a D2H edge that needs more items than it consumes."""
        sink = demanding_checker()
        self.tb.connect(self.head, gpu(cuda.copy(ITEM.itemsize)), sink)
        self.run_tb()
        self.assertEqual(sink.errors, [], "(first bad index, value, bad in call)")
        # It stops once fewer than `demand` items are left.
        self.assertGreater(sink.count, TOTAL_ITEMS - sink.demand)

    def test_008_output_multiple(self):
        """A writer whose output multiple does not divide the ring."""
        mult = cuda.multiply_const_cc(2)
        mult.set_output_multiple(96)
        sink = checker(scale=2)
        self.tb.connect(self.head, gpu(mult), sink)
        self.run_tb()
        # It stops once fewer than one output multiple is left.
        self.assertEqual(sink.errors, [], "(first bad index, value, bad in call)")
        self.assertEqual(sink.count, TOTAL_ITEMS - TOTAL_ITEMS % 96)

    def test_009_python_general_work(self):
        """Python general_work GPU blocks: cupy views of ranges that run past N."""
        # Python blocks must outlive the run; the flowgraph doesn't keep them.
        first = gpu_passthrough_gw(CALL_ITEMS)
        second = gpu_passthrough_gw(OTHER_CALL_ITEMS)
        sink = checker()
        self.tb.connect(self.head, first, second, sink)
        self.run_tb()
        self.assertChecked(sink)


if __name__ == "__main__":
    gr_unittest.run(qa_buffer_wrap)
