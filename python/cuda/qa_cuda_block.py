#!/usr/bin/env python3
#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""QA tests for cuda_block base classes (sync_block, decim_block, interp_block, basic_block)."""

import numpy as np
import cupy as cp
from gnuradio import gr, gr_unittest, blocks, cuda


# ---------------------------------------------------------------------------
# Test cuda_blocks — one per base class / configuration
# ---------------------------------------------------------------------------

class _scale_f32(cuda.sync_block):
    """sync_block: float32 scalar multiply."""
    def __init__(self, k):
        self.k = k
        cuda.sync_block.__init__(self, "scale_f32",
            [np.float32], [np.float32])

    def work(self, input_items, output_items):
        cp.multiply(input_items[0], self.k, out=output_items[0])
        return len(output_items[0])


class _scale_c64(cuda.sync_block):
    """sync_block: complex64 scalar multiply."""
    def __init__(self, k):
        self.k = k
        cuda.sync_block.__init__(self, "scale_c64",
            [np.complex64], [np.complex64])

    def work(self, input_items, output_items):
        cp.multiply(input_items[0], self.k, out=output_items[0])
        return len(output_items[0])


class _scale_vec_f32(cuda.sync_block):
    """sync_block: float32 with vlen > 1."""
    def __init__(self, k, vlen):
        self.k = k
        cuda.sync_block.__init__(self, "scale_vec_f32",
            [(np.float32, vlen)], [(np.float32, vlen)])

    def work(self, input_items, output_items):
        cp.multiply(input_items[0], self.k, out=output_items[0])
        return len(output_items[0])


class _add_two_f32(cuda.sync_block):
    """sync_block: two float32 inputs -> one output."""
    def __init__(self):
        cuda.sync_block.__init__(self, "add_two_f32",
            [np.float32, np.float32], [np.float32])

    def work(self, input_items, output_items):
        cp.add(input_items[0], input_items[1], out=output_items[0])
        return len(output_items[0])


class _scale_i16(cuda.sync_block):
    """sync_block: int16 scalar multiply."""
    def __init__(self, k):
        self.k = np.int16(k)
        cuda.sync_block.__init__(self, "scale_i16",
            [np.int16], [np.int16])

    def work(self, input_items, output_items):
        cp.multiply(input_items[0], self.k, out=output_items[0])
        return len(output_items[0])


class _downsample_f32(cuda.decim_block):
    """decim_block: keep every Nth sample."""
    def __init__(self, decim):
        self.decim = decim
        cuda.decim_block.__init__(self, "downsample_f32",
            [np.float32], [np.float32], decim)

    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0][::self.decim]
        return len(output_items[0])


class _upsample_f32(cuda.interp_block):
    """interp_block: repeat each sample N times."""
    def __init__(self, interp):
        self.interp = interp
        cuda.interp_block.__init__(self, "upsample_f32",
            [np.float32], [np.float32], interp)

    def work(self, input_items, output_items):
        output_items[0][:] = cp.repeat(input_items[0], self.interp)
        return len(output_items[0])


class _passthrough_f32(cuda.basic_block):
    """basic_block: passthrough via general_work + consume_each."""
    def __init__(self):
        cuda.basic_block.__init__(self, "passthrough_f32",
            [np.float32], [np.float32])

    def forecast(self, noutput_items, ninputs):
        return [noutput_items] * ninputs

    def general_work(self, input_items, output_items):
        n = min(len(input_items[0]), len(output_items[0]))
        output_items[0][:n] = input_items[0][:n]
        self.consume_each(n)
        return n


class _convolve_f32(cuda.sync_block):
    """sync_block: FIR filter using set_history()."""
    def __init__(self, taps):
        self._taps = cp.asarray(taps, dtype=np.float32)
        cuda.sync_block.__init__(self, "convolve_f32",
            [np.float32], [np.float32])
        self.set_history(len(taps))

    def work(self, input_items, output_items):
        output_items[0][:] = cp.convolve(
            input_items[0], self._taps, mode='valid')
        return len(output_items[0])


class _c64_to_mag_f32(cuda.sync_block):
    """sync_block: complex64 input → float32 magnitude output."""
    def __init__(self):
        cuda.sync_block.__init__(self, "c64_to_mag_f32",
            [np.complex64], [np.float32])

    def work(self, input_items, output_items):
        cp.abs(input_items[0], out=output_items[0])
        return len(output_items[0])


class _accumulate_f32(cuda.basic_block):
    """basic_block: sum every N input samples into one output (non-1:1 rate)."""
    def __init__(self, n):
        self.n = n
        cuda.basic_block.__init__(self, "accumulate_f32",
            [np.float32], [np.float32])

    def forecast(self, noutput_items, ninputs):
        return [noutput_items * self.n] * ninputs

    def general_work(self, input_items, output_items):
        n_in = len(input_items[0])
        n_out = len(output_items[0])
        n_produce = min(n_out, n_in // self.n)
        if n_produce == 0:
            return 0
        reshaped = input_items[0][:n_produce * self.n].reshape(n_produce, self.n)
        output_items[0][:n_produce] = cp.sum(reshaped, axis=1)
        self.consume_each(n_produce * self.n)
        return n_produce


class _const_source_f32(cuda.sync_block):
    """sync_block source (in_sig=None): produces a constant value."""
    def __init__(self, value: float, count: int):
        self._value = np.float32(value)
        self._count = count
        self._produced = 0
        cuda.sync_block.__init__(self, "const_source_f32",
            None, [np.float32])

    def work(self, input_items, output_items):
        remaining = self._count - self._produced
        if remaining <= 0:
            return -1
        n = min(len(output_items[0]), remaining)
        output_items[0][:n] = self._value
        self._produced += n
        return n


class _sum_sink_f32(cuda.sync_block):
    """sync_block sink (out_sig=None): accumulates sum of all samples."""
    def __init__(self):
        self.total = cp.float32(0.0)
        cuda.sync_block.__init__(self, "sum_sink_f32",
            [np.float32], None)

    def work(self, input_items, output_items):
        self.total += cp.sum(input_items[0])
        return len(input_items[0])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class qa_cuda_block(gr_unittest.TestCase):
    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_sync_float32(self):
        N = 10000
        src_data = np.random.randn(N).astype(np.float32)
        src = blocks.vector_source_f(src_data, False)
        dut = _scale_f32(3.0)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data * 3.0)

    def test_002_sync_complex64(self):
        N = 10000
        src_data = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
        src = blocks.vector_source_c(src_data, False)
        dut = _scale_c64(2.0 + 1j)
        snk = blocks.vector_sink_c()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.complex64)
        # Complex multiply may differ by 1 ULP due to FMA on GPU vs separate ops on CPU
        np.testing.assert_allclose(result, src_data * (2.0 + 1j), rtol=1e-7)

    def test_003_sync_vlen(self):
        N = 10000
        vlen = 4
        src_data = np.random.randn(N * vlen).astype(np.float32)
        src = blocks.vector_source_f(src_data, False, vlen)
        dut = _scale_vec_f32(5.0, vlen)
        snk = blocks.vector_sink_f(vlen)
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data * 5.0)

    def test_004_sync_multi_input(self):
        N = 10000
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        src_a = blocks.vector_source_f(a, False)
        src_b = blocks.vector_source_f(b, False)
        dut = _add_two_f32()
        snk = blocks.vector_sink_f()
        self.tb.connect(src_a, (dut, 0))
        self.tb.connect(src_b, (dut, 1))
        self.tb.connect(dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, a + b)

    def test_005_sync_int16(self):
        N = 10000
        src_data = np.random.randint(-100, 100, N, dtype=np.int16)
        src = blocks.vector_source_s(src_data.tolist(), False)
        dut = _scale_i16(3)
        snk = blocks.vector_sink_s()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.int16)
        expected = (src_data.astype(np.int32) * 3).astype(np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_006_decim_block(self):
        N = 10000
        decim = 4
        src_data = np.arange(N, dtype=np.float32)
        src = blocks.vector_source_f(src_data, False)
        dut = _downsample_f32(decim)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        expected = src_data[::decim]
        np.testing.assert_array_equal(result, expected)

    def test_007_interp_block(self):
        N = 10000
        interp = 3
        src_data = np.arange(N, dtype=np.float32)
        src = blocks.vector_source_f(src_data, False)
        dut = _upsample_f32(interp)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        expected = np.repeat(src_data, interp)
        np.testing.assert_array_equal(result, expected)

    def test_008_basic_general_work(self):
        N = 10000
        src_data = np.random.randn(N).astype(np.float32)
        src = blocks.vector_source_f(src_data, False)
        dut = _passthrough_f32()
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        np.testing.assert_array_equal(result, src_data)

    def test_009_sync_history(self):
        taps = [1.0, 0.0, 0.0, 0.0]
        src_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        src = blocks.vector_source_f(src_data, False)
        dut = _convolve_f32(taps)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        # cp.convolve may differ by 1 ULP from numpy due to FMA
        np.testing.assert_allclose(result, src_data, rtol=1e-6)

    def test_010_sync_different_io_dtypes(self):
        N = 10000
        src_data = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
        src = blocks.vector_source_c(src_data, False)
        dut = _c64_to_mag_f32()
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        expected = np.abs(src_data).astype(np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_011_basic_accumulate(self):
        acc = 8
        N = 10000 * acc
        src_data = np.arange(N, dtype=np.float32)
        src = blocks.vector_source_f(src_data, False)
        dut = _accumulate_f32(acc)
        snk = blocks.vector_sink_f()
        self.tb.connect(src, dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        expected = src_data.reshape(-1, acc).sum(axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_012_sync_source(self):
        N = 10000
        value = 42.0
        dut = _const_source_f32(value, N)
        snk = blocks.vector_sink_f()
        self.tb.connect(dut, snk)
        self.tb.run()
        result = np.array(snk.data(), dtype=np.float32)
        self.assertEqual(len(result), N)
        np.testing.assert_array_equal(result, np.full(N, value, dtype=np.float32))

    def test_013_sync_sink(self):
        # Use a CUDA source so the sink receives device pointers
        N = 10000
        src = _const_source_f32(1.0, N)
        dut = _sum_sink_f32()
        self.tb.connect(src, dut)
        self.tb.run()
        np.testing.assert_allclose(float(dut.total), float(N), rtol=1e-6)


if __name__ == '__main__':
    gr_unittest.run(qa_cuda_block)
