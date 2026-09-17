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


def run_repeat(
    make_block,
    data,
    interp,
    vlen,
    dtype,
    tags=(),
    max_noutput_items=None,
):
    source_type, sink_type = BLOCKS_BY_DTYPE[np.dtype(dtype)]
    tb = gr.top_block()
    source = source_type(data, repeat=False, vlen=vlen, tags=tags)
    repeat = make_block(interp, vlen, dtype)
    sink = sink_type(vlen=vlen)
    tb.connect(source, repeat, sink)
    if max_noutput_items is None:
        tb.run()
    else:
        tb.run(max_noutput_items=max_noutput_items)
    return np.asarray(sink.data(), dtype=dtype), sink.tags()


def make_cpu_repeat(interp, vlen, dtype):
    return blocks.repeat(np.dtype(dtype).itemsize * vlen, interp)


def make_gpu_repeat(interp, vlen, dtype):
    return cuda.repeat_cupy(interp=interp, vlen=vlen, dtype=dtype)


class qa_repeat_cupy(gr_unittest.TestCase):
    def test_001_matches_cpu(self):
        dtype_vlen_pairs = [
            (np.complex64, 1),
            (np.float32, 8),
            (np.int32, 3),
            (np.int16, 4),
            (np.uint8, 1),
        ]
        for interp in (1, 3, 17, 1000):
            for dtype, vlen in dtype_vlen_pairs:
                with self.subTest(interp=interp, dtype=dtype, vlen=vlen):
                    num_items = 257
                    values = np.arange(num_items * vlen).astype(dtype)
                    if np.issubdtype(dtype, np.complexfloating):
                        values += 1j * values[::-1]

                    expected, _ = run_repeat(
                        make_cpu_repeat,
                        values,
                        interp,
                        vlen,
                        dtype,
                    )
                    result, _ = run_repeat(
                        make_gpu_repeat,
                        values,
                        interp,
                        vlen,
                        dtype,
                        max_noutput_items=19 * interp,
                    )
                    np.testing.assert_array_equal(result, expected)

    def test_002_tags_match_cpu(self):
        interp = 7
        tags = [
            gr.tag_utils.python_to_tag(
                (
                    offset,
                    pmt.intern("marker"),
                    pmt.from_long(offset),
                    pmt.PMT_NIL,
                )
            )
            for offset in (0, 3, 19, 63)
        ]
        values = np.arange(64, dtype=np.float32)

        _, expected_tags = run_repeat(
            make_cpu_repeat,
            values,
            interp,
            1,
            np.float32,
            tags=tags,
        )
        _, result_tags = run_repeat(
            make_gpu_repeat,
            values,
            interp,
            1,
            np.float32,
            tags=tags,
            max_noutput_items=5,
        )

        self.assertEqual(
            [tag.offset for tag in result_tags],
            [tag.offset for tag in expected_tags],
        )
        self.assertEqual(
            [pmt.to_long(tag.value) for tag in result_tags],
            [pmt.to_long(tag.value) for tag in expected_tags],
        )

    def test_003_partial_repetitions_across_work_calls(self):
        interp = 7
        vlen = 3
        values = np.arange(31 * vlen, dtype=np.complex64)
        expected, _ = run_repeat(
            make_cpu_repeat,
            values,
            interp,
            vlen,
            np.complex64,
        )
        result, _ = run_repeat(
            make_gpu_repeat,
            values,
            interp,
            vlen,
            np.complex64,
            max_noutput_items=5,
        )
        np.testing.assert_array_equal(result, expected)

    def test_004_output_multiple_and_runtime_interpolation(self):
        repeat = cuda.repeat_cupy(interp=7, output_multiple=16)
        self.assertEqual(repeat.output_multiple(), 16)
        self.assertEqual(repeat.interpolation(), 7)

        repeat.set_interpolation(11)
        self.assertEqual(repeat.interpolation(), 11)

        tb = gr.top_block()
        source = blocks.vector_source_f([1.0], repeat=True)
        sink = blocks.null_sink(gr.sizeof_float)
        tb.connect(source, repeat, sink)
        tb.start()
        repeat.to_basic_block()._post(
            pmt.intern("interpolation"),
            pmt.cons(pmt.intern("interpolation"), pmt.from_long(13)),
        )
        deadline = time.monotonic() + 1
        try:
            while repeat.interpolation() != 13 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(repeat.interpolation(), 13)
        finally:
            tb.stop()
            tb.wait()

    def test_005_invalid_arguments(self):
        invalid_arguments = [
            {"interp": 0},
            {"interp": 1.5},
            {"interp": 1, "vlen": 0},
            {"interp": 1, "vlen": 1.5},
            {"interp": 1, "output_multiple": 0},
            {"interp": 1, "output_multiple": 1.5},
            {"interp": 1, "dtype": object},
            {"interp": 1, "dtype": "V0"},
        ]
        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    cuda.repeat_cupy(**kwargs)

    def test_006_invalid_runtime_interpolation(self):
        repeat = cuda.repeat_cupy(interp=1)
        with self.assertRaises(ValueError):
            repeat.set_interpolation(1.5)


if __name__ == "__main__":
    gr_unittest.run(qa_repeat_cupy)
