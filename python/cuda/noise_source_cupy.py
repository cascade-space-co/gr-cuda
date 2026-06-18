#
# Copyright 2026 Cascade Space.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

"""GPU noise source (CuPy port of ``analog.noise_source_x``)."""

import cupy as cp
import numpy as np
from gnuradio import cuda

# Noise type identifiers matching gr::analog::noise_type_t so flowgraphs may use
# either ``cuda.GR_*`` or ``analog.GR_*`` interchangeably.
GR_UNIFORM = 200
GR_GAUSSIAN = 201
GR_LAPLACIAN = 202
GR_IMPULSE = 203

# Threshold used by gr::random::impulse(), matching the in-tree noise source
# which calls ``d_rng.impulse(9)``.
_IMPULSE_FACTOR = 9.0


def _normalize_seed(seed):
    """Map a GR-style seed to a CuPy RandomState seed.

    A seed of 0 selects a nondeterministic seed (matching the in-tree blocks,
    which seed from the system time). Any other value -- including negative and
    full 64-bit seeds -- is folded into the valid unsigned 64-bit range.
    """
    if seed == 0:
        return None
    return int(seed) & 0xFFFFFFFFFFFFFFFF


def _laplacian(rng, n, dtype):
    """Laplace(0, 1) variates, matching gr::random::laplacian()."""
    z = rng.random_sample(n, dtype=dtype)
    return cp.where(z > 0.5, -cp.log(2.0 * (1.0 - z)), cp.log(2.0 * z))


def _impulse(rng, n, factor, dtype):
    """Impulse variates, matching gr::random::impulse(factor)."""
    z = -np.sqrt(2.0) * cp.log(rng.random_sample(n, dtype=dtype))
    return cp.where(cp.abs(z) <= factor, 0.0, z)


class noise_source_cupy(cuda.sync_block):
    """Noise source on the GPU using CuPy.

    GPU-native counterpart of ``analog.noise_source_x``: fresh random variates
    are generated for every output sample (unlike the fast noise source, which
    samples from a fixed pool -- a CPU optimization that offers no benefit on
    the GPU, where generating fresh variates is just as cheap and higher
    quality).

    Distributions mirror the in-tree block:

    * Uniform   : ``amplitude * (U(0,1) * 2 - 1)``  -> ``U(-amplitude, amplitude)``
    * Gaussian  : ``amplitude * N(0, 1)``
    * Laplacian : ``amplitude * Laplace(0, 1)``  (real types only)
    * Impulse   : ``amplitude * impulse(9)``     (real types only)

    For complex output the amplitude is scaled by ``1/sqrt(2)`` per component
    (so total power matches ``amplitude``), and only Uniform and Gaussian are
    supported, exactly as in the in-tree block.

    The RNG differs from the CPU block (cuRAND vs. GNU Radio's xoroshiro128+),
    so output is not bit-for-bit identical; the statistical distribution
    matches.
    """

    def __init__(
        self,
        noise_type: int = GR_GAUSSIAN,
        amplitude: float = 1.0,
        seed: int = 0,
        dtype: np.dtype = np.complex64,
    ):
        """
        Parameters
        ----------
        noise_type : int
            One of ``GR_UNIFORM``, ``GR_GAUSSIAN``, ``GR_LAPLACIAN``,
            ``GR_IMPULSE``.
        amplitude : float
            Noise amplitude.
        seed : int
            RNG seed. 0 selects a nondeterministic seed.
        dtype : numpy.dtype
            Output sample type (complex64, float32, int32, int16).
        """
        self._dtype = np.dtype(dtype)
        self._is_complex = self._dtype.kind == "c"
        self._noise_type = int(noise_type)
        self._ampl = float(amplitude)
        if self._is_complex and self._noise_type not in (GR_UNIFORM, GR_GAUSSIAN):
            raise ValueError(
                "complex noise_source supports only GR_UNIFORM and GR_GAUSSIAN"
            )
        # Generate variates at the output's component precision (float32 for
        # the supported types) to halve RNG bandwidth vs. the float64 default
        # and skip a cast. complex64 has 4-byte components -> float32.
        comp_bytes = self._dtype.itemsize // (2 if self._is_complex else 1)
        self._gdtype = cp.float64 if comp_bytes > 4 else cp.float32
        self._rng = cp.random.RandomState(_normalize_seed(seed))
        cuda.sync_block.__init__(self, "noise_source_cupy", None, [self._dtype])

    def _variates(self, count, scale):
        """``count`` scaled variates at ``self._gdtype`` precision (no cast).

        For complex output the real and imaginary parts are i.i.d. from the
        same distribution, so a single length-``2n`` draw fills the interleaved
        ``[re, im, re, im, ...]`` buffer -- one RNG call and one contiguous
        write instead of two of each.
        """
        rng = self._rng
        noise_type = self._noise_type
        if noise_type == GR_UNIFORM:
            samples = rng.random_sample(count, dtype=self._gdtype)
            samples *= 2.0 * scale
            samples -= scale
            return samples
        if noise_type == GR_GAUSSIAN:
            samples = rng.standard_normal(count, dtype=self._gdtype)
            samples *= scale
            return samples
        if noise_type == GR_LAPLACIAN:
            samples = _laplacian(rng, count, self._gdtype)
            samples *= scale
            return samples
        if noise_type == GR_IMPULSE:
            samples = _impulse(rng, count, _IMPULSE_FACTOR, self._gdtype)
            samples *= scale
            return samples
        raise ValueError(f"invalid noise type {self._noise_type}")

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        if n == 0:
            return 0
        if self._is_complex:
            # Reinterpret the complex buffer as 2n interleaved float components
            # and fill it with one contiguous draw (re/im share the same dist).
            out.view(self._gdtype)[:] = self._variates(2 * n, self._ampl / np.sqrt(2.0))
        else:
            out[:] = self._variates(n, self._ampl)
        return n

    def set_type(self, noise_type: int):
        noise_type = int(noise_type)
        if self._is_complex and noise_type not in (GR_UNIFORM, GR_GAUSSIAN):
            raise ValueError(
                "complex noise_source supports only GR_UNIFORM and GR_GAUSSIAN"
            )
        self._noise_type = noise_type

    def set_amplitude(self, amplitude: float):
        self._ampl = float(amplitude)

    def type(self) -> int:
        return self._noise_type

    def amplitude(self) -> float:
        return self._ampl
