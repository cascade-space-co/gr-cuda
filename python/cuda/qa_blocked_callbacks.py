#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Tests that cuda_buffer's output_blocked_callback and input_blocked_callback
# fire and produce correct data after buffer compaction.
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
#   Fires when (bufsize - read_index) < items_required AND write < read.
#   We use a consumer (demanding_sink) that demands 48 items via forecast()
#   and a GPU with output_multiple=64.  The GPU stalls when space < 64,
#   and output_blocked_callback short-circuits because min_read > write
#   (the write pointer wrapped while the read pointer is near the end).
#   The consumer then sees fewer items at the buffer tail than it needs,
#   triggering input_blocked_callback.

import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest

SEQ_LEN = 256


def _make_sequence(n):
    """Repeating complex64 ramp: 0, 1, ..., SEQ_LEN-1, 0, 1, ..."""
    pattern = (
        np.arange(SEQ_LEN, dtype=np.float32) + 1j * np.zeros(SEQ_LEN, dtype=np.float32)
    ).astype(np.complex64)
    return pattern, np.tile(pattern, (n // SEQ_LEN) + 1)[:n]


class sequence_sink(gr.sync_block):
    """Validates received data against a repeating expected sequence.

    history=2 sets d_has_history on the upstream cuda_buffer, which is
    required for output_blkd_cb_ready() to pass when the buffer is full.
    The Python per-element validation loop is naturally slower than the
    GPU producer, creating enough backpressure to fill the tiny buffer.
    """

    def __init__(self, expected, hist=2):
        gr.sync_block.__init__(self, "sequence_sink", in_sig=[np.complex64], out_sig=[])
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
                    f"got {got}, expected {exp}"
                )
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
        gr.basic_block.__init__(
            self, name="demanding_sink", in_sig=[np.complex64], out_sig=[]
        )
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
                    f"got {input_items[0][i]}, expected {exp}"
                )
        self._offset += n
        self.consume(0, n)
        return 0


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
        the read pointer and the buffer end, so input_blocked_callback fires
        and compacts from the reader side.
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


if __name__ == "__main__":
    gr_unittest.run(qa_blocked_callbacks)
