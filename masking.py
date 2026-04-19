"""masking.py
============

Utilities for generating missingness masks for time-series imputation.

Mask convention:
    mask[i] = 1.0 -> observed
    mask[i] = 0.0 -> missing

The generator supports:
  - up to K missing blocks (random), each with length in [min_block_len, max_block_len]
  - minimum gap (in observed points) between blocks
  - additional isolated missing points
  - a total missing budget B (count), so overall missingness stays controlled
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MaskConfig:
    max_missing_frac: float = 0.70
    max_blocks: int = 20
    min_block_len: int = 2
    max_block_len: int = 150
    min_gap: int = 10


def _valid_starts(L: int, length: int, forbidden: np.ndarray) -> np.ndarray:
    """Return all start indices s such that [s, s+length) does not touch forbidden."""
    if length > L:
        return np.array([], dtype=np.int64)
    # valid if forbidden[s:s+length].any() is False
    # do this with convolution over boolean mask for speed
    f = forbidden.astype(np.int32)
    window = np.ones(length, dtype=np.int32)
    # 'valid' convolution output length = L-length+1
    hits = np.convolve(f, window, mode="valid")
    return np.where(hits == 0)[0].astype(np.int64)


def generate_missing_mask(
    L: int,
    rng: np.random.Generator,
    cfg: MaskConfig = MaskConfig(),
    *,
    enforce_missing: bool = True,
) -> np.ndarray:
    """Generate a 0/1 mask of length L.

    Missingness is sampled with total missing fraction f ~ U(0, cfg.max_missing_frac).
    Then missing budget B = round(f*L) is split into blocks + isolated points.

    Blocks are separated by at least cfg.min_gap observed points (blocks only).
    """
    if L <= 0:
        raise ValueError("L must be positive")

    f = float(rng.uniform(0.0, cfg.max_missing_frac))
    B = int(round(f * L))
    if enforce_missing:
        B = max(B, 1)
    B = min(B, L)  # cannot miss more than L

    mask = np.ones(L, dtype=np.float32)
    if B == 0:
        return mask

    # split missing budget
    r_block = float(rng.uniform(0.0, 1.0))
    B_block = int(round(r_block * B))
    B_points = max(B - B_block, 0)

    # place blocks
    K = int(rng.integers(0, cfg.max_blocks + 1))
    forbidden = np.zeros(L, dtype=bool)
    remaining_block = B_block

    for _ in range(K):
        if remaining_block < cfg.min_block_len:
            break

        length = int(rng.integers(cfg.min_block_len, cfg.max_block_len + 1))
        length = min(length, remaining_block)
        if length < cfg.min_block_len:
            break

        starts = _valid_starts(L, length, forbidden)
        if starts.size == 0:
            break

        s = int(rng.choice(starts))
        e = s + length
        mask[s:e] = 0.0
        remaining_block -= length

        # update forbidden region for block spacing (blocks only)
        fs = max(0, s - cfg.min_gap)
        fe = min(L, e + cfg.min_gap)
        forbidden[fs:fe] = True

    # add isolated points
    if B_points > 0:
        obs_idx = np.where(mask > 0.5)[0]
        if obs_idx.size > 0:
            n = min(B_points, obs_idx.size)
            pick = rng.choice(obs_idx, size=n, replace=False)
            mask[pick] = 0.0

    # if blocks didn't consume their budget, we don't force it; total missing will be <= B.
    return mask
