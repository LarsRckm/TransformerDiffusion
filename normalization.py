"""normalization.py
=================

Robust per-window normalization utilities.

We normalize a 1D series y to approximately [-1, 1] using quantiles
(robust to outliers). For masked series, quantiles are computed from
observed points only.

Returned parameters are sufficient to apply the same transform to x0 and y
and to convert a sigma in original units to sigma in normalized units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormParams:
    lo: float
    hi: float

    @property
    def span(self) -> float:
        s = self.hi - self.lo
        return s if not np.isclose(s, 0.0) else 1.0

    @property
    def scale(self) -> float:
        # y_norm = 2*(y - lo)/span - 1  -> linear scale factor = 2/span
        return 2.0 / self.span


def robust_normalize_masked(
    y: np.ndarray,
    mask: np.ndarray,
    q_low: float = 0.02,
    q_high: float = 0.98,
    clip: float = 1.5,
) -> tuple[np.ndarray, NormParams]:
    """Normalize y using quantiles computed from observed entries.

    y: (L,)
    mask: (L,) float/bool, 1=observed, 0=missing
    """
    y = np.asarray(y)
    mask = np.asarray(mask)
    obs = y[mask.astype(bool)]

    if obs.size < 4:
        # fallback: use full y (or zeros if degenerate)
        obs = y
    if obs.size == 0:
        params = NormParams(lo=0.0, hi=1.0)
        return np.zeros_like(y, dtype=np.float32), params

    lo = float(np.quantile(obs, q_low))
    hi = float(np.quantile(obs, q_high))
    params = NormParams(lo=lo, hi=hi)
    y_norm = (2.0 * (y - params.lo) / params.span) - 1.0
    y_norm = np.clip(y_norm, -clip, clip)
    return y_norm.astype(np.float32), params


def apply_norm(y: np.ndarray, params: NormParams, clip: float = 1.5) -> np.ndarray:
    y = np.asarray(y)
    y_norm = (2.0 * (y - params.lo) / params.span) - 1.0
    return np.clip(y_norm, -clip, clip).astype(np.float32)


def denormalize(y_norm: np.ndarray, params: NormParams) -> np.ndarray:
    y_norm = np.asarray(y_norm)
    return ((y_norm + 1.0) / 2.0) * params.span + params.lo
