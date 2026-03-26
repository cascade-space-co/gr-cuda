#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
QA tests for general_work() GPU blocks with auto-sync.

Reproduces patterns reported by external users:
  - Chained general_work GPU blocks (variable consume/produce)
  - Fan-out from GPU to both GPU and CPU paths
  - Early return (produce 0) from general_work
"""

import cupy as cp
import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest


class _gpu_delay(cuda.basic_block):
    """Integer sample delay using general_work.

    Buffers input and delays output by ``delay`` samples, so consume
    and produce counts differ — exercising the variable-rate path.
    """

    def __init__(self, delay, dtype=np.float32):
        self._delay = delay
        self._buf = cp.zeros(0, dtype=dtype)
        self._dtype = dtype
        cuda.basic_block.__init__(self, "gpu_delay", [dtype], [dtype])

    def forecast(self, noutput_items, ninputs):
        return [noutput_items + self._delay] * ninputs

    def general_work(self, input_items, output_items):
        n_in = len(input_items[0])
        if n_in > 0:
            self._buf = cp.concatenate([self._buf, input_items[0][:n_in]])
            self.consume_each(n_in)

        n_out = len(output_items[0])
        n_avail = len(self._buf) - self._delay
        if n_avail <= 0:
            return 0

        n = min(n_out, n_avail)
        output_items[0][:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


class _gpu_scale_gw(cuda.basic_block):
    """Scale block using general_work (not sync_block).

    Consumes and produces the same number of items, but goes through
    the general_work / consume_each path.
    """

    def __init__(self, k, dtype=np.float32):
        self.k = dtype(k)
        cuda.basic_block.__init__(self, "gpu_scale_gw", [dtype], [dtype])

    def forecast(self, noutput_items, ninputs):
        return [noutput_items] * ninputs

    def general_work(self, input_items, output_items):
        n = min(len(input_items[0]), len(output_items[0]))
        if n == 0:
            return 0
        cp.multiply(input_items[0][:n], self.k, out=output_items[0][:n])
        self.consume_each(n)
        return n


class _gpu_scale_explicit_produce(cuda.basic_block):
    """Scale block that calls produce() and consume_each() explicitly.

    Tests the produce-before-mark_work_done ordering fix: the wrapper
    must call mark_work_done before produce() advances the write pointer.
    Heavy scratch work widens the race window so the test reliably fails
    without the fix.
    """

    def __init__(self, k, dtype=np.float32):
        self.k = dtype(k)
        self._scratch = cp.ones(500_000, dtype=cp.float32)
        cuda.basic_block.__init__(self, "gpu_scale_explicit", [dtype], [dtype])

    def forecast(self, noutput_items, ninputs):
        return [noutput_items] * ninputs

    def general_work(self, input_items, output_items):
        n = min(len(input_items[0]), len(output_items[0]))
        if n == 0:
            return 0
        for _ in range(100):
            cp.sin(self._scratch, out=self._scratch)
        cp.multiply(input_items[0][:n], self.k, out=output_items[0][:n])
        self.consume_each(n)
        self.produce(0, n)
        return -2  # WORK_CALLED_PRODUCE


class _gpu_decimate_explicit_produce(cuda.basic_block):
    """Decimate-by-2 block with explicit produce() and consume_each().

    Averages adjacent input pairs and scales by k.
    consume != produce, exercising the general_work contract fully.
    """

    def __init__(self, k, dtype=np.float32):
        self.k = dtype(k)
        cuda.basic_block.__init__(self, "gpu_decim_explicit", [dtype], [dtype])

    def forecast(self, noutput_items, ninputs):
        return [2 * noutput_items] * ninputs

    def general_work(self, input_items, output_items):
        nin = len(input_items[0])
        nout = len(output_items[0])
        n = min(nin // 2, nout)
        if n == 0:
            return 0
        inp = input_items[0][: 2 * n]
        cp.add(inp[0::2], inp[1::2], out=output_items[0][:n])
        cp.multiply(output_items[0][:n], self.k, out=output_items[0][:n])
        self.consume_each(2 * n)
        self.produce(0, n)
        return -2  # WORK_CALLED_PRODUCE


class qa_general_work_sync(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_single_general_work_block(self):
        """Single general_work GPU block: CPU -> GPU (gw) -> CPU."""
        N = 50_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        scale = _gpu_scale_gw(3.0)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, scale, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, src_data * 3.0, rtol=1e-8)

    def test_fanout_general_work_gpu_and_cpu(self):
        """
        Fan-out: GPU general_work block feeds both a GPU chain and a
        CPU sink via cuda.tee (required to avoid mixed transfer type).
          src -> tee (gpu) -> scale_gw_2 -> snk_gpu
                           -> snk_cpu
        """
        N = 50_000
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        producer = _gpu_scale_gw(1.0)
        tee = cuda.tee(gr.sizeof_float, gpu=True)

        # GPU path: another general_work block
        gpu_consumer = _gpu_scale_gw(2.0)
        snk_gpu = blocks.vector_sink_f()

        # CPU path: through tee
        snk_cpu = blocks.vector_sink_f()

        self.tb.connect(src, producer, tee)
        self.tb.connect((tee, 0), gpu_consumer, snk_gpu)
        self.tb.connect((tee, 1), snk_cpu)

        self.tb.run()

        result_gpu = np.array(snk_gpu.data(), dtype=np.float32)
        result_cpu = np.array(snk_cpu.data(), dtype=np.float32)
        np.testing.assert_allclose(result_gpu, src_data * 2.0, rtol=1e-8)
        np.testing.assert_allclose(result_cpu, src_data, rtol=1e-8)

    def test_chained_delay_blocks(self):
        """
        Two chained delay blocks with variable consume/produce.
        """
        N = 50_000
        delay1 = 7
        delay2 = 13
        src_data = np.arange(N, dtype=np.float32)

        src = blocks.vector_source_f(src_data, False)
        d1 = _gpu_delay(delay1)
        d2 = _gpu_delay(delay2)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, d1, d2, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = src_data[: len(result)]
        np.testing.assert_array_equal(result, expected)

    def test_explicit_produce_chain(self):
        """Two chained blocks with explicit produce() — tests both
        produce and consume ordering across a GPU->GPU boundary.
        Small max_noutput_items forces many sync points to widen
        the race window."""
        N = 50_000
        k = 2.0
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        s1 = _gpu_scale_explicit_produce(k)
        s2 = _gpu_scale_explicit_produce(k)
        snk = blocks.vector_sink_f()

        s1.set_max_noutput_items(1024)
        s2.set_max_noutput_items(1024)

        self.tb.connect(src, s1, s2, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_allclose(result, src_data * k * k, rtol=1e-8)

    def test_explicit_produce_decimate(self):
        """Decimate-by-2 with explicit produce() — consume != produce."""
        N = 49_152
        k = 3.0
        src_data = np.random.randn(N).astype(np.float32)

        src = blocks.vector_source_f(src_data, False)
        d1 = _gpu_decimate_explicit_produce(k)
        d2 = _gpu_decimate_explicit_produce(k)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, d1, d2, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)

        s1 = src_data.reshape(-1, 2)
        stage1 = (s1[:, 0] + s1[:, 1]) * k
        s2 = stage1.reshape(-1, 2)
        expected = (s2[:, 0] + s2[:, 1]) * k

        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_early_return_zero(self):
        """
        Block that returns 0 on some calls (not enough input buffered).
        Verifies no deadlock or data corruption from the early returns.
        """
        N = 10_000
        src_data = np.arange(N, dtype=np.float32)

        src = blocks.vector_source_f(src_data, False)
        # Large delay relative to typical batch size forces many 0-return calls
        d = _gpu_delay(4096)
        snk = blocks.vector_sink_f()

        self.tb.connect(src, d, snk)
        self.tb.run()

        result = np.array(snk.data(), dtype=np.float32)
        expected = src_data[: len(result)]
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    gr_unittest.run(qa_general_work_sync)
