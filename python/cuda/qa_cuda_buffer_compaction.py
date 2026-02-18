#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Regression test for cuda_buffer compaction deadlock.

Root cause
----------
buffer_single_mapped (the base class of cuda_buffer) gates its compaction
callback on the flag ``d_has_history``.  The flag is set inside
``update_reader_block_history(history, delay)``::

    d_has_history = ((history - 1) != delay)

When a downstream reader has ``history = N`` and ``delay = N - 1`` (which
is the canonical "delay-to-compensate-for-history" pattern used by
qtgui.time_sink_c and others), the expression evaluates to False, leaving
``d_has_history = False``.

Both buffer types wrap their write pointer modulo ``d_bufsize`` via
``index_add``.  For double-mapped buffers, reads that span the wrap
boundary see contiguous data (thanks to the virtual-memory double
mapping), so ``space_available()`` never needs to reserve space at the
end — there is no dead zone and ``d_has_history`` is irrelevant.

For single-mapped buffers like cuda_buffer, reads cannot cross the wrap
boundary contiguously, so ``space_available()`` reserves ``history - 1``
items at the end of the buffer.  This creates a "dead zone" where the
writer cannot produce output.

When the write pointer enters this dead zone, ``space_available()``
returns 0.  The scheduler calls ``output_blkd_cb_ready()``, which requires
``d_has_history == true`` to allow the compaction callback to fire when
``space_avail == 0``.  With ``d_has_history == false``, the callback never
fires and the writer is permanently stuck at ``d_bufsize - (history - 1)``.

Reproducer
----------
The minimum trigger conditions are:

    1. A cuda_buffer output (single-mapped)
    2. A downstream reader with history >= 2  (creates a dead zone)
    3. That reader also sets delay = history - 1  (forces d_has_history=false)

The test uses ``cuda.null_source`` feeding a tiny Python sink that sets
``history=2`` and ``declare_sample_delay(1)`` — the same pattern used by
``qtgui.time_sink_c`` — to trigger the deadlock deterministically.

Fix
---
Override ``update_reader_block_history`` in cuda_buffer to unconditionally
force ``d_has_history = true`` after the base-class call.  Single-mapped
buffers ALWAYS need the compaction path.
"""

import time
import numpy as np

from gnuradio import gr, gr_unittest
from gnuradio import cuda


_SETTLE_S = 0.25
_WINDOW_S = 0.50


class _history_sink(gr.sync_block):
    """Minimal CPU sink with history=2 / delay=1.

    This replicates the buffer configuration of qtgui.time_sink_c (which
    sets history(2) and declare_sample_delay(1) for trigger look-ahead)
    without requiring a display.  The resulting (history-1)==delay causes
    buffer_single_mapped to set d_has_history=false.
    """

    def __init__(self):
        gr.sync_block.__init__(
            self, "history_sink", in_sig=[np.complex64], out_sig=[]
        )
        self.set_history(2)
        self.declare_sample_delay(1)

    def work(self, input_items, output_items):
        return len(input_items[0]) - (self.history() - 1)


class qa_cuda_buffer_compaction(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_no_deadlock_with_history_sink(self):
        """
        A CUDA source feeding a CPU sink with history=2 / delay=1 must not
        deadlock.

        The downstream block's history creates a 1-item dead zone at the
        end of the cuda_buffer where space_available() returns 0.  Combined
        with d_has_history=false (from history-1==delay), the compaction
        callback is gated off and the write pointer is stuck forever.
        """
        src = cuda.null_source(gr.sizeof_gr_complex)
        sink = _history_sink()
        self.tb.connect(src, sink)

        self.tb.start()
        time.sleep(_SETTLE_S)

        count_before = src.nitems_written(0)
        time.sleep(_WINDOW_S)
        count_after = src.nitems_written(0)

        self.tb.stop()
        self.tb.wait()

        self.assertGreater(
            count_after,
            count_before,
            f"cuda_buffer deadlock: source froze at {count_before} items "
            f"and wrote 0 items over {_WINDOW_S} s. "
            f"The downstream history_sink (history=2, delay=1) caused "
            f"d_has_history=false, preventing buffer compaction.",
        )


if __name__ == "__main__":
    gr_unittest.run(qa_cuda_buffer_compaction)
