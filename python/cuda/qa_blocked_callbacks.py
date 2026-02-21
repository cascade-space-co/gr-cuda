#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Tests that cuda_buffer's output_blocked_callback fires and compacts
# correctly, and that input_blocked_callback fires (its compaction path
# is unreachable in the base class — see "input_blocked_callback" below).
#
# Background
# ----------
# GNU Radio's single-mapped buffers are linear (not double-mapped).  When the
# write pointer reaches the end, the scheduler calls output_blocked_callback
# to compact (memmove) unread data back to the start.  Similarly, when the
# read pointer is near the end and needs more contiguous items than remain,
# input_blocked_callback compacts from the reader side.
#
# cuda_buffer overrides both callbacks to keep the host and device buffers
# in sync during compaction.  These tests verify that data stays correct
# after thousands of compactions across H2D, D2H, and D2D transfers.
#
# How the callbacks are triggered
# --------------------------------
# output_blocked_callback:
#   Gated by output_blkd_cb_ready(), which requires EITHER space > 0 but
#   less than output_multiple, OR space == 0 with d_has_history set.
#   The scheduler rounds buffer sizes to multiples of output_multiple, so
#   the write pointer always wraps cleanly and the first case never occurs.
#   We use history=2 on the consumer to set d_has_history, enabling the
#   space == 0 path.
#
# input_blocked_callback:
#   The scheduler calls this when (bufsize - read_index) < items_required.
#   Only basic_blocks with an explicit forecast() set items_required high
#   enough to trigger the call; sync_blocks use items_required=1.
#
#   NOTE: the base-class compaction inside input_blocked_callback_logic
#   (buffer_single_mapped.cc) is effectively dead code.  It requires
#   d_write_index < read_index, but output_blocked_callback_logic
#   subtracts the same offset from d_write_index and every reader,
#   preserving the invariant d_write_index >= all d_read_index.  There
#   is no operation in buffer_single_mapped that violates this invariant,
#   so the compaction branch can never execute.  The callback still
#   exercises cuda_buffer's sync_all_gpu_work() and the transfer_type
#   dispatch, just not the memmove/memcpy lambdas.

import numpy as np
import cupy as cp
from gnuradio import gr, gr_unittest, blocks, cuda

SEQ_LEN = 256


def _make_sequence(n):
    """Repeating complex64 ramp: 0, 1, ..., SEQ_LEN-1, 0, 1, ..."""
    pattern = (np.arange(SEQ_LEN, dtype=np.float32)
               + 1j * np.zeros(SEQ_LEN, dtype=np.float32)).astype(np.complex64)
    return pattern, np.tile(pattern, (n // SEQ_LEN) + 1)[:n]


class sequence_sink(gr.sync_block):
    """Validates received data against a repeating expected sequence.

    history=2 sets d_has_history on the upstream cuda_buffer, which is
    required for output_blkd_cb_ready() to pass when the buffer is full.
    The Python per-element validation loop is naturally slower than the
    GPU producer, creating enough backpressure to fill the tiny buffer.
    """
    def __init__(self, expected, hist=2):
        gr.sync_block.__init__(self, "sequence_sink",
                               in_sig=[np.complex64], out_sig=[])
        self._expected = np.asarray(expected, dtype=np.complex64)
        self._len = len(self._expected)
        self._offset = 0
        self._hist = hist
        self.set_history(hist)

    def work(self, input_items, output_items):
        # Skip the first (hist-1) overlapping history samples.
        consume = len(input_items[0]) - (self._hist - 1)
        if consume <= 0:
            return 0
        start = self._hist - 1
        for i in range(consume):
            exp = self._expected[(self._offset + i) % self._len]
            got = input_items[0][start + i]
            if got != exp:
                raise RuntimeError(
                    f"Sequence mismatch at index {self._offset + i}: "
                    f"got {got}, expected {exp}")
        self._offset += consume
        return consume


class demanding_sink(gr.basic_block):
    """Validates data while demanding a minimum number of items via forecast().

    No history is set, so d_has_history stays false on the upstream buffer.
    This means output_blocked_callback short-circuits when min_read > write
    (returns false without compacting).  The scheduler then falls through to
    the input side, where input_blocked_callback fires because fewer items
    remain at the buffer tail than forecast() demands.

    max_consume controls how many items are consumed per call; together with
    the GPU's output_multiple this controls where the read pointer lands
    relative to the buffer boundary.
    """
    def __init__(self, expected, min_demand=48, max_consume=50):
        gr.basic_block.__init__(self, name="demanding_sink",
                                in_sig=[np.complex64], out_sig=[])
        self._expected = np.asarray(expected, dtype=np.complex64)
        self._len = len(self._expected)
        self._offset = 0
        self._min_demand = min_demand
        self._max_consume = max_consume

    def forecast(self, noutput_items, ninputs):
        return [self._min_demand] * ninputs

    def general_work(self, input_items, output_items):
        n = min(len(input_items[0]), self._max_consume)
        if n == 0:
            return 0
        for i in range(n):
            exp = self._expected[(self._offset + i) % self._len]
            if input_items[0][i] != exp:
                raise RuntimeError(
                    f"Sequence mismatch at index {self._offset + i}: "
                    f"got {input_items[0][i]}, expected {exp}")
        self._offset += n
        self.consume(0, n)
        return 0


class demanding_gpu_passthrough(gr.basic_block):
    """GPU basic_block with custom forecast() to trigger input_blocked on
    upstream D2D or H2D buffers.

    Uses cuda_buffer IO signatures so the upstream buffer is D2D (when fed
    by a C++ GPU block) or H2D (when fed by a CPU block).  The large
    forecast demand combined with slow downstream consumption causes the
    read pointer to land near the buffer tail, triggering
    input_blocked_callback on the upstream cuda_buffer.
    """
    def __init__(self, min_demand=48, max_consume=50):
        sig = cuda.io_signature_make(1, 1, [np.complex64])
        gr.basic_block.__init__(self, name="demanding_gpu_passthrough",
                                in_sig=sig, out_sig=sig)
        self._min_demand = min_demand
        self._max_consume = max_consume
        self.stream = cp.cuda.Stream(non_blocking=True)

    def forecast(self, noutput_items, ninputs):
        return [self._min_demand] * ninputs

    def general_work(self, input_items, output_items):
        cuda.wait_for_work(self.gateway, self.stream.ptr)
        n = min(len(input_items[0]), len(output_items[0]), self._max_consume)
        if n > 0:
            with self.stream:
                d_in = cuda.as_cupy(input_items[0][:n])
                d_out = cuda.as_cupy(output_items[0][:n])
                d_out[:] = d_in
        cuda.mark_work_done(self.gateway, self.stream.ptr)
        self.consume(0, n)
        return n


class qa_blocked_callbacks(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_01_output_blocked_d2h(self):
        """output_blocked_callback on a D2H buffer.

        src (CPU) -> multiply_const (GPU) -> sequence_sink (CPU)
        Buffer between GPU and sink is only 128 complex64 items.
        The sink's history=2 enables the callback; the Python validation
        loop is slower than the GPU, so the buffer fills repeatedly.
        """
        pattern, data = _make_sequence(10_000)
        src = blocks.vector_source_c(data, False)
        gpu = cuda.multiply_const_cc(1.0)
        gpu.set_min_output_buffer(128)
        gpu.set_max_output_buffer(128)
        gpu.set_output_multiple(32)
        sink = sequence_sink(pattern, hist=2)
        sink.set_max_noutput_items(32)
        self.tb.connect(src, gpu, sink)
        self.tb.run()

    def test_02_output_blocked_d2d(self):
        """output_blocked_callback on D2D and D2H buffers.

        src (CPU) -> gpu1 -> gpu2 -> sequence_sink (CPU)
        Both inter-GPU (D2D) and GPU-to-CPU (D2H) buffers are 128 items.
        """
        pattern, data = _make_sequence(10_000)
        src = blocks.vector_source_c(data, False)
        gpu1 = cuda.multiply_const_cc(1.0)
        gpu2 = cuda.multiply_const_cc(1.0)
        for g in (gpu1, gpu2):
            g.set_min_output_buffer(128)
            g.set_max_output_buffer(128)
            g.set_output_multiple(32)
        sink = sequence_sink(pattern, hist=2)
        sink.set_max_noutput_items(32)
        self.tb.connect(src, gpu1, gpu2, sink)
        self.tb.run()

    def test_03_input_blocked_d2h(self):
        """input_blocked_callback on a D2H buffer.

        src (CPU) -> multiply_const (GPU) -> demanding_sink (CPU)
        The GPU's output_multiple=64 means it stalls when buffer space < 64.
        The demanding_sink forecasts 48 items but only ~28 remain between
        the read pointer and the buffer end, so input_blocked_callback fires.
        """
        pattern, data = _make_sequence(10_000)
        src = blocks.vector_source_c(data, False)
        gpu = cuda.multiply_const_cc(1.0)
        gpu.set_min_output_buffer(128)
        gpu.set_max_output_buffer(128)
        gpu.set_output_multiple(64)
        sink = demanding_sink(pattern, min_demand=48, max_consume=50)
        self.tb.connect(src, gpu, sink)
        self.tb.run()

    def test_04_output_blocked_h2d(self):
        """output_blocked_callback on an H2D buffer.

        src (CPU, tiny buffer) -> multiply_const (GPU) -> sequence_sink (CPU)
        The H2D buffer between src and GPU is only 128 items.  The slow
        sequence_sink backpressures the GPU, which in turn cannot consume
        from the H2D buffer fast enough.  The CPU source fills the tiny
        H2D buffer repeatedly, triggering output_blocked_callback in H2D
        mode — exercising the d_base memmove + d_cuda_buf mirror path.
        """
        pattern, data = _make_sequence(10_000)
        src = blocks.vector_source_c(data, False)
        src.set_min_output_buffer(128)
        src.set_max_output_buffer(128)
        gpu = cuda.multiply_const_cc(1.0)
        gpu.set_output_multiple(32)
        sink = sequence_sink(pattern, hist=2)
        sink.set_max_noutput_items(32)
        self.tb.connect(src, gpu, sink)
        self.tb.run()

    def test_05_input_blocked_d2d(self):
        """input_blocked_callback on a D2D buffer.

        src (CPU) -> gpu1 (tiny D2D output) -> gpu_pass -> demanding_sink (CPU)
        The D2D buffer between gpu1 and gpu_pass is only 128 items.
        gpu_pass is a GPU basic_block with forecast() demanding 48 items;
        only basic_blocks trigger input_blocked (sync_blocks do not set
        items_required high enough).

        N is kept small (~200) because a scheduler signaling race between
        the reader-side input_blocked spin and the writer-side sleep
        causes a deadlock with larger data sets.  The done signal from
        the finite source breaks the deadlock for small N.
        """
        pattern, data = _make_sequence(200)
        src = blocks.vector_source_c(data, False)
        gpu1 = cuda.multiply_const_cc(1.0)
        gpu1.set_min_output_buffer(128)
        gpu1.set_max_output_buffer(128)
        gpu1.set_output_multiple(32)
        gpu_pass = demanding_gpu_passthrough(min_demand=48, max_consume=50)
        sink = demanding_sink(pattern, min_demand=48, max_consume=50)
        self.tb.connect(src, gpu1, gpu_pass, sink)
        self.tb.run()

    def test_06_input_blocked_h2d(self):
        """input_blocked_callback on an H2D buffer.

        src (CPU, tiny buffer) -> gpu_pass -> demanding_sink (CPU)
        The H2D buffer between src and gpu_pass is only 128 items.
        gpu_pass is a GPU basic_block with forecast() demanding 48 items.
        The demanding_sink backpressures gpu_pass so it consumes slowly,
        leaving the read pointer near the H2D buffer tail.  When fewer
        than 48 contiguous items remain, input_blocked_callback fires
        in H2D mode, exercising sync_all_gpu_work() and the H2D
        transfer_type dispatch.

        The H2D buffer's writer is the CPU source (not a GPU block), so
        the scheduler signaling race that deadlocks test_05 does not
        apply here — the source produces independently and N=10,000
        works fine.
        """
        pattern, data = _make_sequence(10_000)
        src = blocks.vector_source_c(data, False)
        src.set_min_output_buffer(128)
        src.set_max_output_buffer(128)
        # Constrain the source's write granularity so output_blocked can
        # fail (returns false when min_read > write), forcing the scheduler
        # to fall through to input_blocked on the reader side.
        src.set_output_multiple(64)
        gpu_pass = demanding_gpu_passthrough(min_demand=48, max_consume=50)
        sink = demanding_sink(pattern, min_demand=48, max_consume=50)
        self.tb.connect(src, gpu_pass, sink)
        self.tb.run()


if __name__ == "__main__":
    gr_unittest.run(qa_blocked_callbacks)
