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


def run_select(block, data, vlen, num_selected, dtype, tags=()):
    source_type, sink_type = BLOCKS_BY_DTYPE[np.dtype(dtype)]
    tb = gr.top_block()
    source = source_type(data, repeat=False, vlen=vlen, tags=tags)
    sink = sink_type(vlen=num_selected)
    tb.connect(source, block, sink)
    tb.run()
    return np.asarray(sink.data(), dtype=dtype), sink.tags()


def expected_select(data, vlen, indices):
    return data.reshape(-1, vlen)[:, indices].reshape(-1)


class qa_vector_select(gr_unittest.TestCase):
    def test_001_matches_reference(self):
        cases = [
            (np.complex64, 8, [3]),
            (np.complex64, 8, [0, 3, 5]),
            (np.float32, 16, [15, 0, 7, 7]),
            (np.int32, 4, [1, 2]),
            (np.int16, 3, [2, 1, 0]),
            (np.uint8, 5, [4]),
            (np.float32, 1, [0, 0]),
        ]
        for dtype, vlen, indices in cases:
            with self.subTest(dtype=dtype, vlen=vlen, indices=indices):
                values = make_values(1031, vlen, dtype)
                block = cuda.vector_select(np.dtype(dtype).itemsize, vlen, indices)
                result, _ = run_select(block, values, vlen, len(indices), dtype)
                np.testing.assert_array_equal(
                    result, expected_select(values, vlen, indices)
                )

    def test_001b_single_integer_index(self):
        values = make_values(300, 8, np.complex64)
        block = cuda.vector_select(gr.sizeof_gr_complex, 8, 0)
        result, _ = run_select(block, values, 8, 1, np.complex64)
        np.testing.assert_array_equal(result, expected_select(values, 8, [0]))
        block.set_indices(np.int64(3))
        result, _ = run_select(block, values, 8, 1, np.complex64)
        np.testing.assert_array_equal(result, expected_select(values, 8, [3]))

    def test_002_tags_pass_through(self):
        tags = [
            gr.tag_utils.python_to_tag(
                (offset, pmt.intern("marker"), pmt.from_long(offset), pmt.PMT_NIL)
            )
            for offset in (0, 5, 99)
        ]
        values = make_values(100, 4, np.complex64)
        block = cuda.vector_select(gr.sizeof_gr_complex, 4, [2])
        _, result = run_select(block, values, 4, 1, np.complex64, tags=tags)
        self.assertEqual([t.offset for t in result], [0, 5, 99])

    def test_003_set_indices_before_run(self):
        values = make_values(200, 6, np.complex64)
        block = cuda.vector_select(gr.sizeof_gr_complex, 6, [0, 1])
        block.set_indices([5, 2])
        result, _ = run_select(block, values, 6, 2, np.complex64)
        np.testing.assert_array_equal(result, expected_select(values, 6, [5, 2]))

    def test_004_set_indices_while_running(self):
        vlen = 8
        block = cuda.vector_select(gr.sizeof_float, vlen, [1, 2])
        tb = gr.top_block()
        source = blocks.vector_source_f(
            np.arange(vlen, dtype=np.float32), repeat=True, vlen=vlen
        )
        throttle = blocks.throttle(gr.sizeof_float * vlen, 100_000)
        throttle.set_max_noutput_items(1000)
        sink = blocks.vector_sink_f(vlen=2)
        tb.connect(source, throttle, block, sink)
        tb.start()
        try:
            time.sleep(0.1)
            block.set_indices([6, 4])
            time.sleep(0.2)
        finally:
            tb.stop()
            tb.wait()
        pairs = np.asarray(sink.data(), dtype=np.float32).reshape(-1, 2)
        valid = (pairs == [1, 2]).all(axis=1) | (pairs == [6, 4]).all(axis=1)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(pairs[-1], [6, 4])

    def test_004b_output_multiple(self):
        # As with any GNU Radio block, the end of a finite stream that does not
        # fill a whole output batch may be dropped.
        values = make_values(12_007, 8, np.complex64)
        expected = expected_select(values, 8, [6, 1]).reshape(-1, 2)
        for output_multiple in (1, 64, 5000):
            with self.subTest(output_multiple=output_multiple):
                block = cuda.vector_select(gr.sizeof_gr_complex, 8, [6, 1])
                block.set_output_multiple(output_multiple)
                self.assertEqual(block.output_multiple(), output_multiple)
                result, _ = run_select(block, values, 8, 2, np.complex64)
                result = result.reshape(-1, 2)
                self.assertGreater(len(result), len(expected) - output_multiple)
                np.testing.assert_array_equal(result, expected[: len(result)])

    def test_005_invalid_arguments(self):
        invalid_arguments = [
            {"vlen": 0, "indices": [0]},
            {"vlen": 4, "indices": []},
            {"vlen": 4, "indices": [4]},
            {"vlen": 4, "indices": [-1]},
            {"vlen": 4, "indices": [0], "itemsize": 0},
        ]
        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    cuda.vector_select(**{"itemsize": 8, **kwargs})
        with self.assertRaises(TypeError):
            cuda.vector_select(8, 4, [1.5])

        block = cuda.vector_select(gr.sizeof_gr_complex, 4, [0, 1])
        for indices in ([0], [0, 1, 2], [0, 4]):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    block.set_indices(indices)


if __name__ == "__main__":
    gr_unittest.run(qa_vector_select)
