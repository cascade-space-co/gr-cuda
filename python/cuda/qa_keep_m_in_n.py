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
    if max_noutput_items is None:
        tb.run()
    else:
        tb.run(max_noutput_items=max_noutput_items)
    return np.asarray(sink.data(), dtype=dtype), sink.tags()


def cpu_block(m, n, offset, vlen, dtype):
    return blocks.keep_m_in_n(np.dtype(dtype).itemsize * vlen, m, n, offset)


def gpu_block(m, n, offset, vlen, dtype):
    return cuda.keep_m_in_n(np.dtype(dtype).itemsize * vlen, m, n, offset)


class qa_keep_m_in_n(gr_unittest.TestCase):
    def test_001_matches_cpu(self):
        dtype_vlen_pairs = [
            (np.complex64, 1),
            (np.float32, 8),
            (np.int32, 3),
            (np.int16, 4),
            (np.uint8, 1),
        ]
        # (m, n, offset): contiguous, keep-all, single item, and wrapping selections.
        params = [(1, 1, 0), (2, 5, 0), (2, 5, 3), (3, 5, 4), (5, 5, 2), (7, 64, 60)]
        for m, n, offset in params:
            for dtype, vlen in dtype_vlen_pairs:
                for max_noutput_items in (None, 3 * m):
                    with self.subTest(
                        m=m,
                        n=n,
                        offset=offset,
                        dtype=dtype,
                        vlen=vlen,
                        max_noutput_items=max_noutput_items,
                    ):
                        values = make_values(n * 41 + 3, vlen, dtype)
                        expected, _ = run_keep(
                            cpu_block(m, n, offset, vlen, dtype), values, vlen, dtype
                        )
                        result, _ = run_keep(
                            gpu_block(m, n, offset, vlen, dtype),
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
        values = np.arange(125, dtype=np.float32)
        _, expected = run_keep(
            cpu_block(2, 5, 1, 1, np.float32), values, 1, np.float32, tags=tags
        )
        _, result = run_keep(
            gpu_block(2, 5, 1, 1, np.float32),
            values,
            1,
            np.float32,
            tags=tags,
            max_noutput_items=4,
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

    def test_003_setters_match_cpu(self):
        values = make_values(700, 1, np.float32)
        cpu = cpu_block(2, 5, 0, 1, np.float32)
        gpu = gpu_block(2, 5, 0, 1, np.float32)
        for block in (cpu, gpu):
            block.set_n(9)
            block.set_m(4)
            block.set_offset(7)
        expected, _ = run_keep(cpu, values, 1, np.float32)
        result, _ = run_keep(gpu, values, 1, np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(gpu.output_multiple(), 4)

    def test_004_invalid_arguments(self):
        invalid_arguments = [
            {"m": 0, "n": 2},
            {"m": 1, "n": 0},
            {"m": 3, "n": 2},
            {"m": 1, "n": 2, "offset": -1},
            {"m": 1, "n": 2, "offset": 2},
            {"m": 1, "n": 2, "itemsize": 0},
        ]
        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    cuda.keep_m_in_n(**{"itemsize": 4, **kwargs})
        with self.assertRaises(TypeError):
            cuda.keep_m_in_n(4, 1.5, 2)

    def test_005_invalid_setters_keep_state(self):
        block = cuda.keep_m_in_n(4, 2, 4, 1)
        for setter, value in (
            (block.set_m, 5),
            (block.set_n, 1),
            (block.set_n, 0),
            (block.set_offset, 4),
        ):
            with self.subTest(setter=setter.__name__, value=value):
                with self.assertRaises(ValueError):
                    setter(value)
        values = make_values(400, 1, np.float32)
        expected, _ = run_keep(cpu_block(2, 4, 1, 1, np.float32), values, 1, np.float32)
        result, _ = run_keep(block, values, 1, np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_006_output_multiple(self):
        # As with any GNU Radio block, the end of a finite stream that does not
        # fill a whole output batch may be dropped.
        values = make_values(5 * 2000 + 3, 1, np.float32)
        expected, _ = run_keep(cpu_block(3, 5, 4, 1, np.float32), values, 1, np.float32)
        for output_multiple in (1, 4, 64, 1000):
            with self.subTest(output_multiple=output_multiple):
                block = cuda.keep_m_in_n(4, 3, 5, 4, output_multiple)
                batch = np.lcm(3, output_multiple)
                self.assertEqual(block.output_multiple(), batch)
                result, _ = run_keep(block, values, 1, np.float32)
                self.assertGreater(len(result), len(expected) - batch)
                np.testing.assert_array_equal(result, expected[: len(result)])

        block = cuda.keep_m_in_n(4, 3, 5, output_multiple=4)
        block.set_m(2)
        self.assertEqual(block.output_multiple(), 4)

    def test_007_invalid_output_multiple(self):
        with self.assertRaises(ValueError):
            cuda.keep_m_in_n(4, 1, 2, output_multiple=0)
        with self.assertRaises(TypeError):
            cuda.keep_m_in_n(4, 1, 2, output_multiple=1.5)
        # lcm(65537, 65536) does not fit in an int.
        with self.assertRaises(ValueError):
            cuda.keep_m_in_n(4, 65537, 65537, output_multiple=65536)
        block = cuda.keep_m_in_n(4, 1, 65537, output_multiple=65536)
        with self.assertRaises(ValueError):
            block.set_m(65537)
        self.assertEqual(block.output_multiple(), 65536)

    def run_while_changing(self, m, n, offset, change):
        """Run an int32 ramp through the block, call change(block) mid-run,
        and return the kept positions of each input block, in order."""
        num_items = n * 80_000
        block = cuda.keep_m_in_n(gr.sizeof_int, m, n, offset)
        tb = gr.top_block()
        source = blocks.vector_source_i(np.arange(num_items, dtype=np.int32))
        throttle = blocks.throttle(gr.sizeof_int, 400_000)
        sink = blocks.vector_sink_i()
        tb.connect(source, throttle, block, sink)
        tb.start()
        time.sleep(0.3)
        change(block)
        tb.wait()

        out = np.asarray(sink.data())
        # The input is a ramp, so out // n is the input block and out % n the
        # position kept from it. Output must cover consecutive input blocks.
        ids, starts = np.unique(out // n, return_index=True)
        np.testing.assert_array_equal(ids, np.arange(len(ids)))
        return [tuple(p) for p in np.split(out % n, starts[1:])]

    def assert_switches_once(self, patterns, old, new):
        is_new = [p == new for p in patterns]
        self.assertTrue(any(is_new), "the change took effect after the stream ended")
        switch = is_new.index(True)
        self.assertGreater(switch, 0, "the change took effect before the run")
        self.assertEqual(set(patterns[:switch]), {old})
        self.assertEqual(set(patterns[switch:]), {new})

    def test_008_set_offset_while_running(self):
        patterns = self.run_while_changing(2, 5, 0, lambda b: b.set_offset(3))
        self.assert_switches_once(patterns, (0, 1), (3, 4))

    def test_009_set_m_while_running(self):
        patterns = self.run_while_changing(2, 5, 0, lambda b: b.set_m(3))
        self.assert_switches_once(patterns, (0, 1), (0, 1, 2))


if __name__ == "__main__":
    gr_unittest.run(qa_keep_m_in_n)
