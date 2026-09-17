#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import operator
import threading

import numpy as np
import pmt
from gnuradio import cuda, gr


def _positive_integer(value, name):
    """Return an integer-like value after validating that it is positive."""
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class repeat_cupy(cuda.basic_block):
    """Repeat each input item a configurable number of times on the GPU.

    This is the CUDA-buffer equivalent of ``blocks.repeat``. Each input item is
    copied unchanged to ``interp`` consecutive output items. Work calls may end
    partway through an item's repetitions; the remaining copies continue in the
    next call. Each input tag is propagated to the first corresponding output
    copy.

    The interpolation factor can be changed between work calls with
    ``set_interpolation()`` or by posting an ``(interpolation . value)`` PMT pair
    to the ``interpolation`` message port.

    Parameters
    ----------
    interp : int
        Number of output copies produced for every input item. Must be positive.
    vlen : int, optional
        Number of scalar values in each stream item. Default is 1.
    dtype : numpy dtype, optional
        Scalar data type. Default is ``numpy.float32``.
    output_multiple : int, optional
        Ask the scheduler for output batches in multiples of this size. Larger
        values can amortize scheduler and host-to-device transfer overhead.
        Default is 1.
    """

    def __init__(
        self,
        interp: int,
        vlen: int = 1,
        dtype=np.float32,
        output_multiple: int = 1,
    ):
        interp = _positive_integer(interp, "interp")
        vlen = _positive_integer(vlen, "vlen")
        output_multiple = _positive_integer(output_multiple, "output_multiple")

        self._interpolation = interp
        self._left_to_copy = 0
        self._interp_lock = threading.Lock()
        self._dtype = np.dtype(dtype)
        if self._dtype.hasobject or self._dtype.itemsize == 0:
            raise ValueError("dtype must have a fixed, nonzero item size")

        io_dtype = (self._dtype, vlen) if vlen > 1 else self._dtype
        cuda.basic_block.__init__(
            self,
            "repeat_cupy",
            [io_dtype],
            [io_dtype],
        )
        self.set_output_multiple(output_multiple)
        self.set_tag_propagation_policy(gr.TPP_DONT)

        self._interpolation_port = pmt.intern("interpolation")
        self.message_port_register_in(self._interpolation_port)
        self.set_msg_handler(
            self._interpolation_port,
            self._handle_interpolation_message,
        )

    def interpolation(self):
        """Return the current interpolation factor."""
        with self._interp_lock:
            return self._interpolation

    def set_interpolation(self, interp):
        """Set the interpolation factor between work calls."""
        interp = _positive_integer(interp, "interp")
        with self._interp_lock:
            self._interpolation = interp

    def _handle_interpolation_message(self, message):
        if not pmt.is_pair(message) or not pmt.eq(
            pmt.car(message),
            self._interpolation_port,
        ):
            self.logger.error("message must be an (interpolation . value) pair")
            return
        try:
            self.set_interpolation(pmt.to_long(pmt.cdr(message)))
        except (ValueError, TypeError):
            self.logger.error("interpolation must be a positive integer")

    def forecast(self, noutput_items, ninputs):
        with self._interp_lock:
            if noutput_items <= self._left_to_copy:
                required = 0
            else:
                remaining = noutput_items - self._left_to_copy
                required = (remaining + self._interpolation - 1) // self._interpolation
        return [required] * ninputs

    def _copy_tags(self, consumed, produced, num_items, interp):
        start_in = self.nitems_read(0) + consumed
        start_out = self.nitems_written(0) + produced
        for tag in self.get_tags_in_range(0, start_in, start_in + num_items):
            offset = interp * (tag.offset - start_in) + start_out
            self.add_item_tag(0, offset, tag.key, tag.value, tag.srcid)

    @staticmethod
    def _repeat_into(output, input_item, count):
        output[:count] = input_item

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        noutput_items = len(out)
        ninput_items = len(in0)
        consumed = 0
        produced = 0

        # Serialize interpolation changes with work. If a previous call stopped
        # partway through one item, its remaining copies retain the old count.
        with self._interp_lock:
            interp = self._interpolation

            if self._left_to_copy:
                count = min(self._left_to_copy, noutput_items)
                self._repeat_into(out, in0[0], count)
                self._left_to_copy -= count
                produced += count
                if self._left_to_copy:
                    return produced
                consumed += 1

            available_outputs = noutput_items - produced
            full_items = min(
                available_outputs // interp,
                ninput_items - consumed,
            )
            if full_items:
                self._copy_tags(consumed, produced, full_items, interp)
                output_shape = (full_items, interp) + out.shape[1:]
                input_shape = (full_items, 1) + in0.shape[1:]
                output_end = produced + full_items * interp
                out[produced:output_end].reshape(output_shape)[:] = in0[
                    consumed : consumed + full_items
                ].reshape(input_shape)
                produced = output_end
                consumed += full_items

            remaining_outputs = noutput_items - produced
            if remaining_outputs and consumed < ninput_items:
                self._copy_tags(consumed, produced, 1, interp)
                self._repeat_into(
                    out[produced:],
                    in0[consumed],
                    remaining_outputs,
                )
                self._left_to_copy = interp - remaining_outputs
                produced += remaining_outputs

        self.consume_each(consumed)
        return produced
