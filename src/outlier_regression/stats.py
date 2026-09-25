"""Paired statistics on per-dataset differences (e.g. method A - method B)."""

from __future__ import annotations

import numpy as np


def sign_flip_p(diff, n_flips, rng):
    """Two-sided paired permutation (sign-flip) test of mean(diff) = 0.

    Uses the ``(count + 1) / (n_flips + 1)`` form, so the p-value is never 0.
    """
    d = np.asarray(diff, dtype=float)
    flips = rng.choice([-1.0, 1.0], size=(n_flips, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean())) + 1) / (n_flips + 1))


def bootstrap_ci(diff, n_boot, rng, level=0.95):
    """Percentile bootstrap confidence interval of mean(diff)."""
    d = np.asarray(diff, dtype=float)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(boot, [tail, 100 - tail])
    return float(lo), float(hi)
