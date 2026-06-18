#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA for the CuPy Noise Source.

Combines the in-tree ``qa_noise.py`` checks (instantiation with assorted seeds,
getters) with analytic-moment checks adapted from ``qa_fastnoise.py``. The RNG
differs from the in-tree block (cuRAND vs. xoroshiro128+), so outputs are
compared statistically, never sample-for-sample.
"""

import numpy as np
from gnuradio import analog, blocks, cuda, gr, gr_unittest

N = 500000
AMPL = 3.0


def _collect(tb, src, itemsize, sink, n=N):
    head = blocks.head(itemsize, n)
    tb.connect(src, head, sink)
    tb.run()
    return np.array(sink.data())


class qa_noise_source_cupy(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    # -- in-tree qa_noise.py parity -----------------------------------------

    def test_instantiate(self):
        cuda.noise_source_cupy(cuda.GR_GAUSSIAN, 10, 10)
        cuda.noise_source_cupy(cuda.GR_GAUSSIAN, 10, -10)
        cuda.noise_source_cupy(cuda.GR_GAUSSIAN, 10, -(2**63))
        cuda.noise_source_cupy(cuda.GR_GAUSSIAN, 10, 2**64 - 1)

    def test_getters(self):
        op = cuda.noise_source_cupy(cuda.GR_GAUSSIAN, 10, 10, dtype=np.float32)
        self.assertEqual(op.type(), cuda.GR_GAUSSIAN)
        self.assertEqual(op.amplitude(), 10)

    # -- distribution moments (amplitude = 1) -------------------------------

    def _moments(self, dtype, noise_type, n=N):
        sink = (
            blocks.vector_sink_c()
            if np.dtype(dtype).kind == "c"
            else blocks.vector_sink_f()
        )
        itemsize = (
            gr.sizeof_gr_complex if np.dtype(dtype).kind == "c" else gr.sizeof_float
        )
        dut = cuda.noise_source_cupy(noise_type, 1.0, seed=43, dtype=dtype)
        return _collect(self.tb, dut, itemsize, sink, n)

    def test_moments_real_uniform(self):
        data = self._moments(np.float32, cuda.GR_UNIFORM)
        self.assertAlmostEqual(data.min(), -1, places=3)
        self.assertAlmostEqual(data.max(), 1, places=3)
        self.assertAlmostEqual(data.mean(), 0, delta=0.02)
        self.assertAlmostEqual(data.var(), (2.0**2) / 12, delta=0.01)

    def test_moments_real_gaussian(self):
        data = self._moments(np.float32, cuda.GR_GAUSSIAN)
        self.assertAlmostEqual(data.mean(), 0, delta=0.02)
        self.assertAlmostEqual(data.var(), 1.0, delta=0.02)

    def test_moments_real_laplacian(self):
        data = self._moments(np.float32, cuda.GR_LAPLACIAN)
        self.assertAlmostEqual(data.mean(), 0, delta=0.05)
        self.assertAlmostEqual(data.var(), 2.0, delta=0.05)

    def test_moments_real_impulse(self):
        data = self._moments(np.float32, cuda.GR_IMPULSE)
        # Mostly zeros; nonzero values are large (|z| > factor).
        nonzero = np.abs(data[data != 0])
        self.assertGreater(nonzero.size, 0)
        self.assertGreater(nonzero.min(), 9.0 * 0.99)

    def test_moments_complex_uniform(self):
        data = self._moments(np.complex64, cuda.GR_UNIFORM)
        self.assertAlmostEqual(data.real.var(), 0.5 * (2.0**2) / 12, delta=0.01)
        self.assertAlmostEqual(data.imag.var(), 0.5 * (2.0**2) / 12, delta=0.01)

    def test_moments_complex_gaussian(self):
        data = self._moments(np.complex64, cuda.GR_GAUSSIAN)
        self.assertAlmostEqual(data.real.var(), 0.5, delta=0.02)
        self.assertAlmostEqual(data.imag.var(), 0.5, delta=0.02)

    # -- statistical comparison vs the CPU block ----------------------------

    def test_vs_cpu_real_gaussian(self):
        ref = _collect(
            gr.top_block(),
            analog.noise_source_f(analog.GR_GAUSSIAN, AMPL, 0),
            gr.sizeof_float,
            blocks.vector_sink_f(),
        )
        res = _collect(
            self.tb,
            cuda.noise_source_cupy(cuda.GR_GAUSSIAN, AMPL, seed=7, dtype=np.float32),
            gr.sizeof_float,
            blocks.vector_sink_f(),
        )
        self.assertAlmostEqual(res.mean(), 0.0, delta=0.05)
        self.assertAlmostEqual(res.std(), AMPL, delta=0.05 * AMPL)
        self.assertAlmostEqual(res.std(), ref.std(), delta=0.05 * AMPL)

    def test_vs_cpu_complex_gaussian(self):
        ref = _collect(
            gr.top_block(),
            analog.noise_source_c(analog.GR_GAUSSIAN, AMPL, 0),
            gr.sizeof_gr_complex,
            blocks.vector_sink_c(),
        )
        res = _collect(
            self.tb,
            cuda.noise_source_cupy(cuda.GR_GAUSSIAN, AMPL, seed=7, dtype=np.complex64),
            gr.sizeof_gr_complex,
            blocks.vector_sink_c(),
        )
        self.assertAlmostEqual(res.real.std(), AMPL / np.sqrt(2), delta=0.05 * AMPL)
        self.assertAlmostEqual(res.real.std(), ref.real.std(), delta=0.05 * AMPL)
        self.assertAlmostEqual(res.imag.std(), ref.imag.std(), delta=0.05 * AMPL)

    # -- misc ---------------------------------------------------------------

    def test_reproducibility(self):
        def run():
            dut = cuda.noise_source_cupy(
                cuda.GR_GAUSSIAN, AMPL, seed=43, dtype=np.float32
            )
            return _collect(
                gr.top_block(), dut, gr.sizeof_float, blocks.vector_sink_f(), 100000
            )

        np.testing.assert_array_equal(run(), run())

    def test_complex_rejects_laplacian(self):
        with self.assertRaises(ValueError):
            cuda.noise_source_cupy(cuda.GR_LAPLACIAN, AMPL, seed=1, dtype=np.complex64)


if __name__ == "__main__":
    gr_unittest.run(qa_noise_source_cupy)
