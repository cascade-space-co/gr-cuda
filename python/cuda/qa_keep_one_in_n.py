#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import time

import numpy as np
import pmt
from gnuradio import blocks, cuda, gr, gr_unittest

BLOCKS_BY_DTYPE = {
    np.dtype(np.complex64): (blocks.vector_source_c, blocks.vector_sink_c),
    np.dtype(np.float32): (blocks.vector_source_f, blocks.vector_sink_f),
    np.dtype(np.int32): (blocks.vector_source_i, blocks.vector_sink_i),
    np.dtype(np.int16): (blocks.vector_source_s, blocks.vector_sink_s),
    np.dtype(np.uint8): (blocks.vector_source_b, blocks.vector_sink_b),
}


def make_values(num_items, vlen, dtype):
    values = (np.arange(num_items * vlen) % 251).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += 1j * values[::-1]
    return values


def run_keep(block, data, vlen, dtype, tags=(), max_noutput_items=None):
    source_type, sink_type = BLOCKS_BY_DTYPE[np.dtype(dtype)]
    tb = gr.top_block()
    source = source_type(data, repeat=False, vlen=vlen, tags=tags)
    sink = sink_type(vlen=vlen)
    tb.connect(source, block, sink)
    if max_noutput_items is not None:
        block.set_max_noutput_items(max_noutput_items)
    tb.run()
    return np.asarray(sink.data(), dtype=dtype), sink.tags()


def cpu_block(n, vlen, dtype):
    return blocks.keep_one_in_n(np.dtype(dtype).itemsize * vlen, n)


def gpu_block(n, vlen, dtype):
    return cuda.keep_one_in_n(np.dtype(dtype).itemsize * vlen, n)


class qa_keep_one_in_n(gr_unittest.TestCase):
    def test_001_matches_cpu(self):
        dtype_vlen_pairs = [
            (np.complex64, 1),
            (np.float32, 8),
            (np.int32, 3),
            (np.int16, 4),
            (np.uint8, 1),
        ]
        for n in (1, 3, 17, 1000):
            for dtype, vlen in dtype_vlen_pairs:
                for max_noutput_items in (None, 5):
                    with self.subTest(
                        n=n, dtype=dtype, vlen=vlen, max_noutput_items=max_noutput_items
                    ):
                        values = make_values(2603, vlen, dtype)
                        expected, _ = run_keep(
                            cpu_block(n, vlen, dtype), values, vlen, dtype
                        )
                        result, _ = run_keep(
                            gpu_block(n, vlen, dtype),
                            values,
                            vlen,
                            dtype,
                            max_noutput_items=max_noutput_items,
                        )
                        np.testing.assert_array_equal(result, expected)

    def test_002_tags_match_cpu(self):
        tags = [
            gr.tag_utils.python_to_tag(
                (offset, pmt.intern("marker"), pmt.from_long(offset), pmt.PMT_NIL)
            )
            for offset in (0, 3, 4, 19, 63, 100)
        ]
        values = np.arange(128, dtype=np.float32)
        _, expected = run_keep(
            cpu_block(4, 1, np.float32), values, 1, np.float32, tags=tags
        )
        _, result = run_keep(
            gpu_block(4, 1, np.float32),
            values,
            1,
            np.float32,
            tags=tags,
            max_noutput_items=3,
        )
        self.assertEqual([t.offset for t in result], [t.offset for t in expected])
        self.assertEqual(
            [pmt.to_long(t.value) for t in result],
            [pmt.to_long(t.value) for t in expected],
        )
        self.assertEqual(
            [pmt.symbol_to_string(t.key) for t in result],
            [pmt.symbol_to_string(t.key) for t in expected],
        )
        self.assertEqual(
            [pmt.write_string(t.srcid) for t in result],
            [pmt.write_string(t.srcid) for t in expected],
        )

    def test_003_set_n_matches_cpu(self):
        values = make_values(500, 1, np.float32)
        cpu = cpu_block(3, 1, np.float32)
        cpu.set_n(7)
        gpu = gpu_block(3, 1, np.float32)
        gpu.set_n(7)
        expected, _ = run_keep(cpu, values, 1, np.float32)
        result, _ = run_keep(gpu, values, 1, np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_004_invalid_arguments(self):
        invalid_arguments = [
            {"itemsize": 4, "n": 0},
            {"itemsize": 0, "n": 1},
        ]
        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    cuda.keep_one_in_n(**kwargs)
        with self.assertRaises(TypeError):
            cuda.keep_one_in_n(4, 1.5)
        block = cuda.keep_one_in_n(4, 2)
        with self.assertRaises(ValueError):
            block.set_n(0)

    def test_005_output_multiple(self):
        # As with any GNU Radio block, the end of a finite stream that does not
        # fill a whole output batch may be dropped.
        values = make_values(50_003, 1, np.float32)
        for n in (1, 7):
            expected, _ = run_keep(cpu_block(n, 1, np.float32), values, 1, np.float32)
            for output_multiple in (1, 64, 5000):
                with self.subTest(n=n, output_multiple=output_multiple):
                    block = cuda.keep_one_in_n(4, n)
                    block.set_output_multiple(output_multiple)
                    self.assertEqual(block.output_multiple(), output_multiple)
                    result, _ = run_keep(block, values, 1, np.float32)
                    self.assertGreater(len(result), len(expected) - output_multiple)
                    np.testing.assert_array_equal(result, expected[: len(result)])

    def test_006_set_n_while_running(self):
        # The input is a ramp, so each output value is the index of the input
        # item it came from. Tags carry their own input offset as the value.
        n_old, n_new = 3, 5
        num_items = 400_000
        tags = [
            gr.tag_utils.python_to_tag(
                (offset, pmt.intern("marker"), pmt.from_long(offset), pmt.PMT_NIL)
            )
            for offset in range(0, num_items, 997)
        ]
        block = cuda.keep_one_in_n(gr.sizeof_int, n_old)
        tb = gr.top_block()
        source = blocks.vector_source_i(
            np.arange(num_items, dtype=np.int32), repeat=False, tags=tags
        )
        throttle = blocks.throttle(gr.sizeof_int, 400_000)
        throttle.set_max_noutput_items(4000)
        sink = blocks.vector_sink_i()
        tb.connect(source, throttle, block, sink)
        tb.start()
        time.sleep(0.3)
        block.set_n(n_new)
        tb.wait()

        out = np.asarray(sink.data())
        self.assertEqual(out[0], n_old - 1)
        gaps = np.diff(out)
        changed = np.flatnonzero(gaps != n_old)
        self.assertTrue(changed.size, "set_n() took effect after the stream ended")
        switch = changed[0]
        # One transition gap: what was left of the old group, then n_new.
        self.assertGreaterEqual(gaps[switch], n_new)
        self.assertLess(gaps[switch], n_old + n_new)
        np.testing.assert_array_equal(gaps[switch + 1 :], n_new)
        self.assertGreater(len(gaps) - switch, 1000)

        # Every tag lands on or just before the output kept for its input
        # item, never on output written before that item arrived.
        result_tags = sink.tags()
        self.assertEqual(len(result_tags), len(tags))
        for tag in result_tags:
            input_offset = pmt.to_long(tag.value)
            kept = out[tag.offset]
            self.assertGreaterEqual(kept, input_offset)
            self.assertLess(kept, input_offset + n_old + n_new)


if __name__ == "__main__":
    gr_unittest.run(qa_keep_one_in_n)
