#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA for the CuPy Signal Source.

Combines two strategies:

* Deterministic known-value tests ported from the in-tree
  ``gr-analog/python/analog/qa_sig_source.py``.
* End-to-end comparison against ``analog.sig_source_x`` for every waveform.
"""

import math

import numpy as np
from gnuradio import analog, blocks, cuda, gr, gr_unittest

# Map cuda waveform constants to the equivalent analog ones (values are equal,
# but this makes the correspondence explicit).
_WAVEFORMS = {
    "const": (cuda.GR_CONST_WAVE, analog.GR_CONST_WAVE),
    "sin": (cuda.GR_SIN_WAVE, analog.GR_SIN_WAVE),
    "cos": (cuda.GR_COS_WAVE, analog.GR_COS_WAVE),
    "square": (cuda.GR_SQR_WAVE, analog.GR_SQR_WAVE),
    "triangle": (cuda.GR_TRI_WAVE, analog.GR_TRI_WAVE),
    "saw": (cuda.GR_SAW_WAVE, analog.GR_SAW_WAVE),
}

SAMP_RATE = 32000.0
FREQ = 1001.0
AMPL = 2.0
PHASE = 0.4
N = 8000


class qa_sig_source_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def _run_cuda(self, src, itemsize, sink):
        head = blocks.head(itemsize, N)
        self.tb.connect(src, head, sink)
        self.tb.run()

    def _compare_complex(self, name):
        cuda_wf, analog_wf = _WAVEFORMS[name]
        offset = 0.5 + 0.25j

        # Reference: CPU signal source.
        ref_tb = gr.top_block()
        ref_src = analog.sig_source_c(SAMP_RATE, analog_wf, FREQ, AMPL, offset, PHASE)
        ref_head = blocks.head(gr.sizeof_gr_complex, N)
        ref_snk = blocks.vector_sink_c()
        ref_tb.connect(ref_src, ref_head, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.complex64)

        # DUT: GPU signal source.
        dut = cuda.sig_source_cupy(
            SAMP_RATE, cuda_wf, FREQ, AMPL, offset, PHASE, dtype=np.complex64
        )
        snk = blocks.vector_sink_c()
        self._run_cuda(dut, gr.sizeof_gr_complex, snk)
        result = np.array(snk.data(), dtype=np.complex64)

        self.assertEqual(len(result), len(expected))
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

    def _compare_float(self, name):
        cuda_wf, analog_wf = _WAVEFORMS[name]
        offset = 0.5

        ref_tb = gr.top_block()
        ref_src = analog.sig_source_f(SAMP_RATE, analog_wf, FREQ, AMPL, offset, PHASE)
        ref_head = blocks.head(gr.sizeof_float, N)
        ref_snk = blocks.vector_sink_f()
        ref_tb.connect(ref_src, ref_head, ref_snk)
        ref_tb.run()
        expected = np.array(ref_snk.data(), dtype=np.float32)

        dut = cuda.sig_source_cupy(
            SAMP_RATE, cuda_wf, FREQ, AMPL, offset, PHASE, dtype=np.float32
        )
        snk = blocks.vector_sink_f()
        self._run_cuda(dut, gr.sizeof_float, snk)
        result = np.array(snk.data(), dtype=np.float32)

        self.assertEqual(len(result), len(expected))
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-3)

    def test_complex_const(self):
        self._compare_complex("const")

    def test_complex_sin(self):
        self._compare_complex("sin")

    def test_complex_cos(self):
        self._compare_complex("cos")

    def test_complex_square(self):
        self._compare_complex("square")

    def test_complex_triangle(self):
        self._compare_complex("triangle")

    def test_complex_saw(self):
        self._compare_complex("saw")

    def test_float_const(self):
        self._compare_float("const")

    def test_float_sin(self):
        self._compare_float("sin")

    def test_float_cos(self):
        self._compare_float("cos")

    def test_float_square(self):
        self._compare_float("square")

    def test_float_triangle(self):
        self._compare_float("triangle")

    def test_float_saw(self):
        self._compare_float("saw")

    # -- Known-value tests ported from the in-tree qa_sig_source.py ----------

    def _known_float(self, waveform, ampl, expected, places=5):
        n = len(expected)
        dut = cuda.sig_source_cupy(8, waveform, 1.0, ampl, 0, 0.0, dtype=np.float32)
        head = blocks.head(gr.sizeof_float, n)
        snk = blocks.vector_sink_f()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        self.assertFloatTuplesAlmostEqual(list(snk.data()), expected, places)

    def _known_complex(self, waveform, ampl, expected, places=5):
        n = len(expected)
        dut = cuda.sig_source_cupy(8, waveform, 1.0, ampl, 0, 0.0, dtype=np.complex64)
        head = blocks.head(gr.sizeof_gr_complex, n)
        snk = blocks.vector_sink_c()
        self.tb.connect(dut, head, snk)
        self.tb.run()
        self.assertComplexTuplesAlmostEqual(list(snk.data()), expected, places)

    def test_known_const_f(self):
        self._known_float(cuda.GR_CONST_WAVE, 1.5, [1.5] * 10)

    def test_known_sine_f(self):
        s = math.sqrt(2) / 2
        self._known_float(cuda.GR_SIN_WAVE, 1.0, [0, s, 1, s, 0, -s, -1, -s, 0])

    def test_known_cosine_f(self):
        s = math.sqrt(2) / 2
        self._known_float(cuda.GR_COS_WAVE, 1.0, [1, s, 0, -s, -1, -s, 0, s, 1])

    def test_known_cosine_c(self):
        s = math.sqrt(2) / 2
        sj = 1j * math.sqrt(2) / 2
        self._known_complex(
            cuda.GR_COS_WAVE,
            1.0,
            [1, s + sj, 1j, -s + sj, -1, -s - sj, -1j, s - sj, 1],
        )

    def test_known_sqr_f(self):
        self._known_float(cuda.GR_SQR_WAVE, 1.0, [0, 0, 0, 0, 1, 1, 1, 1, 0])

    def test_known_tri_f(self):
        self._known_float(
            cuda.GR_TRI_WAVE, 1.0, [1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1]
        )

    def test_known_saw_f(self):
        self._known_float(
            cuda.GR_SAW_WAVE, 1.0, [0.5, 0.625, 0.75, 0.875, 0, 0.125, 0.25, 0.375, 0.5]
        )

    # Note: the in-tree complex square/saw known-value arrays sample exactly on
    # phase boundaries (+-pi/2, pi) where the published values reflect the C++
    # NCO's floating-point drift rather than the waveform formula. Those two
    # waveforms are instead covered by the samp_rate=32000 / phase=0.4
    # comparison tests above, which avoid the exact-boundary ambiguity.

    def test_known_tri_c(self):
        self._known_complex(
            cuda.GR_TRI_WAVE,
            1.0,
            [
                1 + 0.5j,
                0.75 + 0.75j,
                0.5 + 1j,
                0.25 + 0.75j,
                0 + 0.5j,
                0.25 + 0.25j,
                0.5 + 0j,
                0.75 + 0.25j,
                1 + 0.5j,
            ],
        )

    def test_accessors(self):
        dut = cuda.sig_source_cupy(8, cuda.GR_SIN_WAVE, 1.0, 2.5, -1.0, 0.0)
        self.assertAlmostEqual(dut.frequency(), 1.0)
        self.assertAlmostEqual(dut.amplitude(), 2.5)
        self.assertAlmostEqual(dut.offset(), -1.0)
        dut.set_frequency(3.0)
        dut.set_amplitude(4.0)
        self.assertAlmostEqual(dut.frequency(), 3.0)
        self.assertAlmostEqual(dut.amplitude(), 4.0)


if __name__ == "__main__":
    gr_unittest.run(qa_sig_source_cupy)
