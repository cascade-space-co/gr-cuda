#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr, gr_unittest, blocks, fft as gr_fft

from gnuradio import cuda
import cupy as cp


def _run_flowgraph(src, block, sink, fft_size):
    # Single-block flowgraph; input length drives buffer wrap behavior.
    tb = gr.top_block()
    tb.connect(src, block)
    tb.connect(block, sink)
    tb.run()
    data = np.array(sink.data(), dtype=np.complex64)
    # Prevent CuPy plan-cache teardown crashes during flowgraph unwinding.
    cp.fft.config.get_plan_cache().clear()
    if data.size == 0:
        return data
    return data.reshape((-1, fft_size))


def _make_src(data, fft_size, real_input):
    if real_input:
        return blocks.vector_source_f(data.astype(np.float32), False, fft_size)
    return blocks.vector_source_c(data.astype(np.complex64), False, fft_size)


def _make_sink(fft_size):
    return blocks.vector_sink_c(fft_size)


def _make_native_block(fft_size, forward, window, shift, real_input):
    if real_input:
        return gr_fft.fft_vfc(fft_size, forward, window, shift, 1)
    return gr_fft.fft_vcc(fft_size, forward, window, shift, 1)


def _make_cuda_block(fft_size, forward, window, shift, real_input):
    return cuda.fft(
        fft_size, forward=forward, window=window, shift=shift, real_input=real_input
    )


def _make_cupy_block(fft_size, forward, window, shift, real_input):
    return cuda.fft_cupy(
        fft_size, forward=forward, window=window, shift=shift, real_input=real_input
    )


def _run_native_fft(data, fft_size, forward, window, shift, real_input):
    if real_input and not forward:
        # Workaround: fft_vfc reverse expects real input (not a complex spectrum),
        # so it is not a true inverse of the forward real FFT. Use fft_vcc here
        # to get a complex-to-complex inverse for reference comparisons.
        src = blocks.vector_source_c(data.astype(np.complex64), False, fft_size)
        real_input_for_block = False
    else:
        src = _make_src(data, fft_size, real_input)
        real_input_for_block = real_input

    sink = _make_sink(fft_size)
    block = _make_native_block(fft_size, forward, window, shift, real_input_for_block)
    return _run_flowgraph(src, block, sink, fft_size)


def _run_cuda_fft(data, fft_size, forward, window, shift, real_input):
    src = _make_src(data, fft_size, real_input)
    sink = _make_sink(fft_size)
    block = _make_cuda_block(fft_size, forward, window, shift, real_input)
    return _run_flowgraph(src, block, sink, fft_size)


def _run_cupy_fft(data, fft_size, forward, window, shift, real_input):
    src = _make_src(data, fft_size, real_input)
    sink = _make_sink(fft_size)
    block = _make_cupy_block(fft_size, forward, window, shift, real_input)
    return _run_flowgraph(src, block, sink, fft_size)


def _run_cuda_fft_shift(data, fft_size):
    src = blocks.vector_source_c(data.astype(np.complex64), False, fft_size)
    sink = _make_sink(fft_size)
    block = cuda.fft_shift(fft_size)
    return _run_flowgraph(src, block, sink, fft_size)


class qa_fft(gr_unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.fft_sizes = [256, 2048, 16384, 131072, 1048576]

    def _make_input(self, fft_size, real_input):
        # Feed multiple FFT vectors so buffers wrap.
        nvec = 16 if fft_size < 32768 else 2  # Makes test run faster
        total = fft_size * nvec
        if real_input:
            return self.rng.standard_normal(total).astype(np.float32)
        return (
            self.rng.standard_normal(total)
            + 1j * self.rng.standard_normal(total)
        ).astype(np.complex64)

    def _compare_fft(self, runner, fft_size, forward, window, shift, real_input):
        data = self._make_input(fft_size, real_input)
        # Compare CUDA/CuPy outputs against the native FFT reference.
        expected = _run_native_fft(
            data, fft_size, forward, window, shift, real_input
        )
        actual = runner(
            data, fft_size, forward, window, shift, real_input
        )
        # 1e-3 tolerance is enough for algorithm differences between native and CUDA.
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    def test_cuda_fft_sizes_forward(self):
        for fft_size in self.fft_sizes:
            for real_input in (False, True):
                with self.subTest(fft_size=fft_size, real_input=real_input):
                    self._compare_fft(_run_cuda_fft, fft_size, True, [], False, real_input)

    def test_cuda_fft_sizes_inverse(self):
        for fft_size in self.fft_sizes:
            for real_input in (False, True):
                with self.subTest(fft_size=fft_size, real_input=real_input):
                    self._compare_fft(_run_cuda_fft, fft_size, False, [], False, real_input)

    def test_cuda_fft_window(self):
        fft_size = 2048
        window = self.rng.standard_normal(fft_size).astype(np.float32).tolist()
        for real_input in (False, True):
            with self.subTest(real_input=real_input):
                self._compare_fft(_run_cuda_fft, fft_size, True, window, False, real_input)

    def test_cuda_fft_shift(self):
        fft_size = 257
        for real_input in (False, True):
            with self.subTest(real_input=real_input, direction="forward"):
                self._compare_fft(_run_cuda_fft, fft_size, True, [], True, real_input)
            with self.subTest(real_input=real_input, direction="inverse"):
                self._compare_fft(_run_cuda_fft, fft_size, False, [], True, real_input)

    def test_cupy_fft_sizes_forward(self):
        for fft_size in self.fft_sizes:
            for real_input in (False, True):
                with self.subTest(fft_size=fft_size, real_input=real_input):
                    self._compare_fft(_run_cupy_fft, fft_size, True, [], False, real_input)

    def test_cupy_fft_sizes_inverse(self):
        for fft_size in self.fft_sizes:
            for real_input in (False, True):
                with self.subTest(fft_size=fft_size, real_input=real_input):
                    self._compare_fft(_run_cupy_fft, fft_size, False, [], False, real_input)

    def test_cupy_fft_window(self):
        fft_size = 2048
        window = self.rng.standard_normal(fft_size).astype(np.float32).tolist()
        for real_input in (False, True):
            with self.subTest(real_input=real_input):
                self._compare_fft(_run_cupy_fft, fft_size, True, window, False, real_input)

    def test_cupy_fft_shift(self):
        fft_size = 257
        for real_input in (False, True):
            with self.subTest(real_input=real_input, direction="forward"):
                self._compare_fft(_run_cupy_fft, fft_size, True, [], True, real_input)
            with self.subTest(real_input=real_input, direction="inverse"):
                self._compare_fft(_run_cupy_fft, fft_size, False, [], True, real_input)

    def test_cuda_fft_shift_block(self):
        fft_size = 257
        data = self._make_input(fft_size, real_input=False)
        expected = np.fft.fftshift(data.reshape((-1, fft_size)), axes=1)
        actual = _run_cuda_fft_shift(data, fft_size)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    gr_unittest.run(qa_fft)
