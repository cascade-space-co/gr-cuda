#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA for the CUDA type converters, compared against gr::blocks."""

import numpy as np
from gnuradio import blocks, cuda, gr, gr_unittest

HOST_BLOCKS = {
    np.dtype(np.complex64): (blocks.vector_source_c, blocks.vector_sink_c),
    np.dtype(np.float32): (blocks.vector_source_f, blocks.vector_sink_f),
    np.dtype(np.int32): (blocks.vector_source_i, blocks.vector_sink_i),
    np.dtype(np.int16): (blocks.vector_source_s, blocks.vector_sink_s),
    np.dtype(np.uint8): (blocks.vector_source_b, blocks.vector_sink_b),
    np.dtype(np.int8): (blocks.vector_source_b, blocks.vector_sink_b),
}

N = 4099  # not a multiple of any batch or vector length used below
rng = np.random.default_rng(1234)


def run(block, inputs, outputs, max_noutput_items=None):
    """Run block with host-side vector sources and sinks.

    inputs: list of (array, vlen); outputs: list of (dtype, vlen).
    Returns a list of output arrays.
    """
    tb = gr.top_block()
    for port, (data, vlen) in enumerate(inputs):
        source_type = HOST_BLOCKS[data.dtype][0]
        host = data.view(np.uint8) if data.dtype == np.int8 else data
        tb.connect(source_type(host.tolist(), False, vlen), (block, port))
    sinks = []
    for port, (dtype, vlen) in enumerate(outputs):
        sink = HOST_BLOCKS[np.dtype(dtype)][1](vlen)
        tb.connect((block, port), sink)
        sinks.append((sink, dtype))
    if max_noutput_items is None:
        tb.run()
    else:
        tb.run(max_noutput_items=max_noutput_items)
    results = []
    for sink, dtype in sinks:
        data = np.asarray(sink.data())
        if np.dtype(dtype) == np.int8:
            data = data.astype(np.uint8).view(np.int8)
        results.append(data.astype(dtype))
    return results


def floats(n=N):
    """Random floats plus rounding ties and values beyond every integer range."""
    edges = np.array(
        [
            0.5,
            1.5,
            2.5,
            -0.5,
            -1.5,
            -2.5,
            127.5,
            128.0,
            -128.5,
            -129.0,
            255.5,
            256.0,
            32767.5,
            -32768.5,
            40000.0,
            -40000.0,
            3e9,
            -3e9,
            0.0,
            -0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([edges, (rng.standard_normal(n) * 300).astype(np.float32)])


def complexes(n=N):
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    z[:4] = [0, 1, -1j, -2 + 0j]
    return z.astype(np.complex64)


def ints(dtype, n=N):
    info = np.iinfo(dtype)
    edges = np.array([info.min, info.max, 0, -1 if info.min < 0 else 1], dtype=dtype)
    values = rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True)
    return np.concatenate([edges, values])


def whole_vectors(data, vlen):
    return data[: len(data) // vlen * vlen]


class qa_type_converters(gr_unittest.TestCase):
    def check(self, name, args, inputs, outputs, rtol=0, atol=0, maxulp=0):
        """Compare cuda.<name>(*args) against blocks.<name>(*args), both in one
        run and with small work calls.

        maxulp allows for GNU Radio's own results depending on which VOLK
        kernel the machine uses (dividing by a scale or multiplying by its
        reciprocal).
        """
        expected = run(getattr(blocks, name)(*args), inputs, outputs)
        for max_noutput_items in (None, 7):
            actual = run(getattr(cuda, name)(*args), inputs, outputs, max_noutput_items)
            for exp, act in zip(expected, actual, strict=True):
                self.assertEqual(len(act), len(exp))
                if maxulp:
                    np.testing.assert_array_max_ulp(
                        act.view(np.float32), exp.view(np.float32), maxulp
                    )
                elif rtol or atol:
                    np.testing.assert_allclose(act, exp, rtol=rtol, atol=atol)
                else:
                    np.testing.assert_array_equal(act, exp)

    def test_int_to_float(self):
        cases = [
            ("char_to_float", np.int8),
            ("short_to_float", np.int16),
            ("int_to_float", np.int32),
        ]
        for name, dtype in cases:
            for vlen, scale in ((1, 1.0), (3, 7.0), (2, 0.25)):
                with self.subTest(name=name, vlen=vlen, scale=scale):
                    self.check(
                        name,
                        (vlen, scale),
                        [(whole_vectors(ints(dtype, N * vlen), vlen), vlen)],
                        [(np.float32, vlen)],
                        maxulp=1,
                    )

    def test_uchar_to_float(self):
        data = ints(np.uint8)
        self.check(
            "uchar_to_float",
            (),
            [(data, 1)],
            [(np.float32, 1)],
        )
        out = run(cuda.uchar_to_float(4, 2.0), [(data[:4000], 4)], [(np.float32, 4)])[0]
        np.testing.assert_array_equal(out, data[:4000].astype(np.float32) / 2)

    def test_float_to_int(self):
        cases = [
            ("float_to_char", np.int8),
            ("float_to_short", np.int16),
            ("float_to_int", np.int32),
        ]
        for name, dtype in cases:
            for vlen, scale in ((1, 1.0), (3, 0.5), (2, 100.0)):
                with self.subTest(name=name, vlen=vlen, scale=scale):
                    data = whole_vectors(floats(N * vlen), vlen)
                    self.check(
                        name,
                        (vlen, scale),
                        [(data, vlen)],
                        [(dtype, vlen)],
                    )

    def test_float_to_uchar(self):
        for vlen, scale, bias in ((1, 1.0, 0.0), (2, 0.5, 128.0), (1, 100.0, -3.0)):
            with self.subTest(vlen=vlen, scale=scale, bias=bias):
                data = whole_vectors(floats(N * vlen), vlen)
                self.check(
                    "float_to_uchar",
                    (vlen, scale, bias),
                    [(data, vlen)],
                    [(np.uint8, vlen)],
                )

    def test_float_to_int_saturates_huge_values(self):
        # gr::blocks::float_to_int gives INT_MIN for values beyond the int64
        # range, +inf and NaN; the CUDA block saturates and maps NaN to 0.
        data = np.array(
            [1e19, -1e19, np.inf, -np.inf, 3e9, -3e9, np.nan], dtype=np.float32
        )
        out = run(cuda.float_to_int(), [(data, 1)], [(np.int32, 1)])[0]
        info = np.iinfo(np.int32)
        np.testing.assert_array_equal(out, [info.max, info.min] * 3 + [0])

    def test_char_short(self):
        for vlen in (1, 5):
            with self.subTest(vlen=vlen):
                self.check(
                    "char_to_short",
                    (vlen,),
                    [(whole_vectors(ints(np.int8, N * vlen), vlen), vlen)],
                    [(np.int16, vlen)],
                )
                self.check(
                    "short_to_char",
                    (vlen,),
                    [(whole_vectors(ints(np.int16, N * vlen), vlen), vlen)],
                    [(np.int8, vlen)],
                )

    def test_complex_parts(self):
        exact = ["complex_to_real", "complex_to_imag"]
        rounded = ["complex_to_mag", "complex_to_mag_squared"]
        for vlen in (1, 3):
            z = complexes(N * vlen)
            for name in exact + rounded:
                with self.subTest(name=name, vlen=vlen):
                    self.check(
                        name,
                        (vlen,),
                        [(z, vlen)],
                        [(np.float32, vlen)],
                        rtol=0 if name in exact else 1e-6,
                    )

    def test_complex_to_mag_extreme_values(self):
        # re^2 + im^2 overflows or underflows here; hypotf does not.
        z = np.array(
            [3e19, 1e19 + 1e19j, 1e-25, 1e-30 + 1e-30j, 1e-45, 3e38 + 3e38j],
            dtype=np.complex64,
        )
        expected = np.abs(z.astype(np.complex128)).astype(np.float32)
        for block in (cuda.complex_to_mag(), cuda.complex_to_magphase()):
            with self.subTest(block=block.name()):
                outputs = [(np.float32, 1)] * block.output_signature().max_streams()
                mag = run(block, [(z, 1)], outputs)[0]
                np.testing.assert_allclose(mag, expected, rtol=1e-6)

    def test_complex_to_arg(self):
        # gr::blocks uses a fast atan2 approximation; the CUDA block is exact.
        z = complexes()
        gpu = run(cuda.complex_to_arg(), [(z, 1)], [(np.float32, 1)])[0]
        np.testing.assert_allclose(gpu, np.angle(z), rtol=0, atol=2e-6)
        self.check(
            "complex_to_arg",
            (),
            [(z, 1)],
            [(np.float32, 1)],
            atol=1e-4,
        )

    def test_complex_to_magphase(self):
        z = complexes()
        mag, phase = run(
            cuda.complex_to_magphase(), [(z, 1)], [(np.float32, 1), (np.float32, 1)]
        )
        np.testing.assert_allclose(mag, np.abs(z), rtol=1e-6)
        np.testing.assert_allclose(phase, np.angle(z), rtol=0, atol=2e-6)

    def test_magphase_to_complex(self):
        mag = np.abs(rng.standard_normal(N)).astype(np.float32)
        phase = rng.uniform(-10, 10, N).astype(np.float32)
        self.check(
            "magphase_to_complex",
            (),
            [(mag, 1), (phase, 1)],
            [(np.complex64, 1)],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_complex_float_ports(self):
        z = complexes()
        re, im = z.real.copy(), z.imag.copy()
        for vlen in (1, 2):
            with self.subTest(vlen=vlen):
                self.check(
                    "complex_to_float",
                    (vlen,),
                    [(z[: N // vlen * vlen], vlen)],
                    [(np.float32, vlen), (np.float32, vlen)],
                )
                self.check(
                    "complex_to_float",
                    (vlen,),
                    [(z[: N // vlen * vlen], vlen)],
                    [(np.float32, vlen)],
                )
                self.check(
                    "float_to_complex",
                    (vlen,),
                    [(re[: N // vlen * vlen], vlen), (im[: N // vlen * vlen], vlen)],
                    [(np.complex64, vlen)],
                )
                self.check(
                    "float_to_complex",
                    (vlen,),
                    [(re[: N // vlen * vlen], vlen)],
                    [(np.complex64, vlen)],
                )

    def test_complex_to_interleaved(self):
        z = (complexes() * 50).astype(np.complex64)
        z[4:8] = [0.5 + 1.5j, 2.5 - 0.5j, 300 - 300j, 1e6 + 0j]
        cases = [("char", np.int8), ("short", np.int16)]
        for kind, dtype in cases:
            for vector in (False, True):
                for scale in (1.0, 2.0):
                    with self.subTest(kind=kind, vector=vector, scale=scale):
                        name = f"complex_to_interleaved_{kind}"
                        vlen = 2 if vector else 1
                        self.check(
                            name,
                            (vector, scale),
                            [(z, 1)],
                            [(dtype, vlen)],
                        )

    def test_interleaved_to_complex(self):
        for vector in (False, True):
            vlen = 2 if vector else 1
            for scale in (1.0, 127.0):
                with self.subTest(kind="char", vector=vector, scale=scale):
                    data = ints(np.int8, 2 * N)
                    self.check(
                        "interleaved_char_to_complex",
                        (vector, scale),
                        [(data, vlen)],
                        [(np.complex64, 1)],
                        maxulp=1,
                    )
                for swap in (False, True):
                    with self.subTest(
                        kind="short", vector=vector, scale=scale, swap=swap
                    ):
                        data = ints(np.int16, 2 * N)
                        self.check(
                            "interleaved_short_to_complex",
                            (vector, swap, scale),
                            [(data, vlen)],
                            [(np.complex64, 1)],
                            maxulp=1,
                        )

    def test_interleaved_char_swap(self):
        data = ints(np.int8, 2 * N)
        out = run(
            cuda.interleaved_char_to_complex(False, 2.0, True),
            [(data, 1)],
            [(np.complex64, 1)],
        )[0]
        pairs = data.astype(np.float32).reshape(-1, 2) / 2
        np.testing.assert_array_equal(out, pairs[:, 1] + 1j * pairs[:, 0])

    def test_setters(self):
        block = cuda.float_to_uchar(1, 1.0, 0.0)
        block.set_scale(2.0)
        block.set_bias(10.0)
        self.assertEqual((block.scale(), block.bias()), (2.0, 10.0))
        out = run(block, [(np.array([1.0, 2.5], np.float32), 1)], [(np.uint8, 1)])[0]
        np.testing.assert_array_equal(out, [12, 15])

        block = cuda.interleaved_short_to_complex()
        block.set_swap(True)
        block.set_scale_factor(2.0)
        self.assertEqual((block.swap(), block.scale_factor()), (True, 2.0))
        out = run(block, [(np.array([2, 4], np.int16), 1)], [(np.complex64, 1)])[0]
        np.testing.assert_array_equal(out, [2 + 1j])


if __name__ == "__main__":
    gr_unittest.run(qa_type_converters)
