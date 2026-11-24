#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""Test helper: change a running block's settings at an exact input item."""

import time

import numpy as np
from gnuradio import gr


class call_at_item(gr.sync_block):
    """Passthrough that calls ``callback`` once, just before item ``item``.

    It forwards the items before ``item``, waits until ``downstream`` has
    consumed all of them, then calls ``callback`` and forwards the rest.  So
    ``downstream`` sees exactly the items before ``item`` with its old
    settings and the rest with the new ones.  The stream must run past
    ``item``, and ``downstream`` must be able to consume exactly ``item``
    items.  ``timed_out`` is set if it never does.
    """

    def __init__(self, itemsize, item, callback, downstream, timeout_s=10):
        sig = [(np.uint8, itemsize)]
        gr.sync_block.__init__(self, "call_at_item", in_sig=sig, out_sig=sig)
        self.item = item
        self.callback = callback
        self.downstream = downstream
        self.timeout_s = timeout_s
        self.called = False
        self.timed_out = False

    def work(self, input_items, output_items):
        n = len(input_items[0])
        if not self.called:
            remaining = self.item - self.nitems_read(0)
            if remaining > 0:
                n = min(n, remaining)
            else:
                deadline = time.monotonic() + self.timeout_s
                while self.downstream.nitems_read(0) < self.item:
                    if time.monotonic() > deadline:
                        self.timed_out = True
                        break
                    time.sleep(0.0005)
                self.callback()
                self.called = True
        output_items[0][:n] = input_items[0][:n]
        return n
