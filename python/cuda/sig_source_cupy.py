#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""GPU signal source (CuPy port of ``analog.sig_source_x``)."""

import math

import cupy as cp
import numpy as np
from gnuradio import cuda

# Waveform identifiers matching gr::analog::gr_waveform_t so flowgraphs may use
# either ``cuda.GR_*_WAVE`` or ``analog.GR_*_WAVE`` interchangeably.
GR_CONST_WAVE = 100
GR_SIN_WAVE = 101
GR_COS_WAVE = 102
GR_SQR_WAVE = 103
GR_TRI_WAVE = 104
GR_SAW_WAVE = 105

# The in-tree ``sig_source`` does not use a floating-point oscillator; it uses
# ``gr::fxpt_nco``, a 32-bit fixed-point NCO. The phase is a ``uint32`` that maps
# [-pi, pi) onto the full integer range and wraps by plain integer overflow; the
# scaling constants are single precision. We replicate that exactly so the phase
# (and therefore the square/saw threshold decisions, which are discontinuous and
# thus sensitive to sub-sample phase) matches GR bit-for-bit. A plain
# double-precision phase would be MORE accurate, but the fixed-point NCO has a
# ~5e-8 relative frequency error that accumulates and shifts the discontinuity
# samples relative to GR; see gnuradio-runtime/include/gnuradio/fxpt.h.
_TWO_PI = (
    2.0 * np.pi
)  # double; GR computes the angle rate in double before the float cast
_F_PI = np.float32(np.pi)
_F_TAU = np.float32(2.0 * np.pi)
_F_TWO31 = np.float32(2147483648.0)  # 2**31, exact in float32
# get_phase() scale: gr::fxpt::fixed_to_float multiplies by PI/2**31 in float32.
_PHASE_SCALE = np.float32(_F_PI / _F_TWO31)


def _float_to_fixed(angle):
    """Port of ``gr::fxpt::float_to_fixed``: radians -> int32 fixed-point phase.

    All arithmetic is single precision and the final conversion truncates
    toward zero, matching the C++ exactly.
    """
    x = np.float32(angle)
    # Fold into [-pi, pi): d = (int)floor(x / TAU + 0.5); x -= d * TAU.
    d = math.floor(float(np.float32(np.float32(x / _F_TAU) + np.float32(0.5))))
    x = np.float32(x - np.float32(np.float32(d) * _F_TAU))
    prod = np.float32(np.float32(x * _F_TWO31) / _F_PI)
    return int(np.int32(prod))


class sig_source_cupy(cuda.sync_block):
    """Generates a configurable waveform on the GPU using CuPy.

    Drop-in GPU counterpart of ``analog.sig_source_x`` supporting constant,
    sine, cosine, square, triangle and saw-tooth waveforms. Phase is tracked
    across ``work()`` calls with a numerically controlled oscillator
    equivalent to the in-tree block, so output is phase-continuous.
    """

    def __init__(
        self,
        sampling_freq: float = 1.0,
        waveform: int = GR_COS_WAVE,
        frequency: float = 1000.0,
        amplitude: float = 1.0,
        offset=0.0,
        phase: float = 0.0,
        dtype: np.dtype = np.complex64,
    ):
        """
        Parameters
        ----------
        sampling_freq : float
            Sample rate in Hz.
        waveform : int
            One of ``GR_CONST_WAVE``, ``GR_SIN_WAVE``, ``GR_COS_WAVE``,
            ``GR_SQR_WAVE``, ``GR_TRI_WAVE``, ``GR_SAW_WAVE``.
        frequency : float
            Signal frequency in Hz.
        amplitude : float
            Signal amplitude.
        offset : float or complex
            DC offset added to every sample.
        phase : float
            Initial phase in radians.
        dtype : numpy.dtype
            Output sample type (complex64, float32, int32, int16, int8).
        """
        self._dtype = np.dtype(dtype)
        self._is_complex = self._dtype.kind == "c"
        self._sampling_freq = float(sampling_freq)
        self._waveform = int(waveform)
        self._frequency = float(frequency)
        self._amplitude = float(amplitude)
        self._offset = offset
        self._phase = float(phase)
        # Fixed-point NCO state (see gr::fxpt_nco): uint32 accumulator and int32
        # increment.
        self._phase_acc = _float_to_fixed(self._phase) & 0xFFFFFFFF
        self._update_phase_inc()
        cuda.sync_block.__init__(self, "sig_source_cupy", None, [self._dtype])

    def _update_phase_inc(self):
        rate = (
            _TWO_PI * self._frequency / self._sampling_freq
            if self._sampling_freq
            else 0.0
        )
        self._phase_inc = _float_to_fixed(rate)

    def _phases(self, n):
        """Per-sample NCO phase in [-pi, pi), bit-matching ``fxpt_nco``.

        The fixed-point accumulator advances by integer addition and wraps by
        ``uint32`` overflow; ``get_phase`` reinterprets it as a signed int32 and
        scales by ``PI/2**31`` in single precision.
        """
        acc = self._phase_acc + cp.arange(n, dtype=cp.int64) * self._phase_inc
        signed = ((acc + 0x80000000) & 0xFFFFFFFF) - 0x80000000
        return signed.astype(cp.float32) * _PHASE_SCALE

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        if n == 0:
            return 0

        waveform = self._waveform
        amplitude = self._amplitude
        offset = self._offset

        if waveform == GR_CONST_WAVE:
            out[:] = self._dtype.type(amplitude + offset)
        elif waveform in (GR_SIN_WAVE, GR_COS_WAVE):
            phases = self._phases(n)
            if self._is_complex:
                vals = amplitude * (cp.cos(phases) + 1j * cp.sin(phases)) + offset
            elif waveform == GR_SIN_WAVE:
                vals = amplitude * cp.sin(phases) + offset
            else:
                vals = amplitude * cp.cos(phases) + offset
            out[:] = vals.astype(self._dtype)
        elif waveform == GR_SQR_WAVE:
            out[:] = self._square(n).astype(self._dtype)
        elif waveform == GR_TRI_WAVE:
            out[:] = self._triangle(n).astype(self._dtype)
        elif waveform == GR_SAW_WAVE:
            out[:] = self._sawtooth(n).astype(self._dtype)
        else:
            raise ValueError(f"sig_source_cupy: invalid waveform {waveform}")

        # Advance the fixed-point accumulator with uint32 overflow wrap.
        self._phase_acc = (self._phase_acc + n * self._phase_inc) & 0xFFFFFFFF
        return n

    # The square/triangle/saw shapes read the NCO phase in [-pi, pi), exactly as
    # the in-tree block reads ``d_nco.get_phase()``.
    def _square(self, n):
        phases = self._phases(n)
        amplitude = self._amplitude
        offset = self._offset
        if self._is_complex:
            # real high from -pi to 0; imaginary leads by 90 deg.
            re = cp.where(phases < 0, amplitude, 0.0)
            im = cp.where((phases >= -np.pi / 2) & (phases < np.pi / 2), amplitude, 0.0)
            return (re + 1j * im) + offset
        return cp.where(phases < 0, amplitude, 0.0) + offset

    def _triangle(self, n):
        phases = self._phases(n)
        amplitude = self._amplitude
        offset = self._offset
        # A triangle is just amplitude*(1 - |phase|/pi): a single abs+fma instead
        # of cp.where branches (which materialize every branch as a full array).
        inv_pi = amplitude / np.pi
        re = amplitude - cp.abs(phases) * inv_pi
        if not self._is_complex:
            return re + offset
        # Imaginary part leads by 90 deg: same triangle of the phase shifted by
        # pi/2 and wrapped back into [-pi, pi).
        shifted = (phases + np.pi / 2) % (2 * np.pi) - np.pi
        im = amplitude - cp.abs(shifted) * inv_pi
        # Assemble complex64 directly so we never spill through a complex128 temp
        # (the Python ``1j`` literal would promote the whole expression).
        vals = cp.empty(phases.shape, dtype=cp.complex64)
        vals.real = re
        vals.imag = im
        return vals + offset

    def _sawtooth(self, n):
        phases = self._phases(n)
        amplitude = self._amplitude
        offset = self._offset
        if self._is_complex:
            base = amplitude * phases / (2 * np.pi)
            re = base + amplitude / 2
            im = cp.where(
                phases < -np.pi / 2, base + 5 * amplitude / 4, base + amplitude / 4
            )
            return (re + 1j * im) + offset
        return amplitude * phases / (2 * np.pi) + amplitude / 2 + offset

    def set_sampling_freq(self, sampling_freq: float):
        self._sampling_freq = float(sampling_freq)
        self._update_phase_inc()

    def set_waveform(self, waveform: int):
        self._waveform = int(waveform)

    def set_frequency(self, frequency: float):
        self._frequency = float(frequency)
        self._update_phase_inc()

    def set_amplitude(self, amplitude: float):
        self._amplitude = float(amplitude)

    def set_offset(self, offset):
        self._offset = offset

    def set_phase(self, phase: float):
        self._phase = float(phase)
        self._phase_acc = _float_to_fixed(self._phase) & 0xFFFFFFFF

    def frequency(self) -> float:
        return self._frequency

    def amplitude(self) -> float:
        return self._amplitude

    def offset(self):
        return self._offset
