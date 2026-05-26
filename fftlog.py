"""Minimal FFTLog correlator utilities.
- ``xi00``
- ``xi0m2``
- ``xi2m2``
- ``xi20``
- ``xi40``
- ``xi11``
- ``qv``
- ``xi_l_n(...)``
"""

from __future__ import annotations

import numpy as np
from scipy.special import loggamma


class _SphericalFFTLog:
    """Internal FFTLog spherical-Bessel transform engine."""

    def __init__(self, source_grid: np.ndarray, max_order: int, fourier_space: bool):
        self._source_grid = source_grid
        self._max_order = max_order
        self._prefactor = np.sqrt(np.pi) if not fourier_space else np.sqrt(np.pi) / (2 * np.pi**2)

        n_input = len(source_grid)
        log_step = np.log(source_grid[-1] / source_grid[0]) / (n_input - 1)
        n_fft = 2 ** (int(np.ceil(np.log2(n_input))) + 1)
        pad_count = n_fft - n_input

        self._n_fft = n_fft
        self._log_step = log_step
        self._pad = np.zeros((n_fft - n_input) // 2)
        self._slice = np.arange(pad_count - pad_count // 2, n_fft - pad_count // 2)

        modes = np.arange(0, n_fft // 2 + 1)
        self._tilt_by_order = {}
        self._target_by_order = {}
        self._kernel_by_order = {}

        for ell in range(max_order):
            tilt = max(0, 1.5 - ell)
            phase = log_step / np.pi * np.angle(self._mellin_kernel(ell, tilt + 1j * np.pi / log_step))
            target_grid = np.exp(phase - log_step) * source_grid / (source_grid[0] * source_grid[-1])
            kernel = self._mellin_kernel(ell, tilt + 2j * np.pi / n_fft / log_step * modes)
            kernel = kernel * np.exp(-2j * np.pi * phase / n_fft / log_step * modes)

            self._tilt_by_order[ell] = tilt
            self._target_by_order[ell] = target_grid
            self._kernel_by_order[ell] = kernel

    def apply(self, order: int, values: np.ndarray):
        """Apply spherical transform of order ``order`` to ``values``."""
        tilt = self._tilt_by_order[order]
        target_grid = self._target_by_order[order]
        padded = np.concatenate((self._pad, self._source_grid ** (3 - tilt) * values, self._pad))
        coeffs = np.fft.rfft(padded)
        mapped = self._kernel_by_order[order] * coeffs
        transformed = np.fft.hfft(mapped) / self._n_fft
        return target_grid, target_grid ** (-tilt) * transformed[self._slice]

    def _mellin_kernel(self, order: int, z):
        return self._prefactor * np.exp(
            np.log(2) * (z - 2)
            + loggamma(0.5 * (order + z))
            - loggamma(0.5 * (3 + order - z))
        )


class CorrelationFFTLog:
    """Compute a minimal set of FFTLog generalized correlation functions.

    Args:
        k_values: Monotonic Fourier grid ``k``.
        power_spectrum: ``P(k)`` sampled on ``k_values``.
        q_values: Optional output separation grid ``q``. If omitted, uses
            ``logspace(-5, 5, 20000)``.
    """

    def __init__(self, k_values: np.ndarray, power_spectrum: np.ndarray, q_values: np.ndarray | None = None):
        self.k = k_values
        self.p = power_spectrum
        self.qv = np.logspace(-5, 5, int(2e4)) if q_values is None else q_values

        self._k_to_q = _SphericalFFTLog(self.k, max_order=5, fourier_space=True)
        self._compute_required_correlators()

    def xi_l_n(self, ell: int, n_power: int) -> np.ndarray:
        """Return generalized correlation function ``xi_{ell,n}(q)``.

        Args:
            ell: Spherical-Bessel order.
            n_power: Power of ``k`` multiplying the spectrum integrand.
        """
        return self._compute_xi(ell=ell, n_power=n_power)

    def _compute_required_correlators(self):
        self.xi00 = self.xi_l_n(0, 0)
        self.xi0m2 = self.xi_l_n(0, -2)
        self.xi2m2 = self.xi_l_n(2, -2)
        self.xi20 = self.xi_l_n(2, 0)
        self.xi40 = self.xi_l_n(4, 0)
        self.xi11 = self.xi_l_n(1, 1)

    def _compute_xi(self, ell: int, n_power: int) -> np.ndarray:
        integrand = self.p * self.k**n_power
        q_grid, transformed = self._k_to_q.apply(ell, integrand)
        return np.interp(self.qv, q_grid, transformed)
