"""Paired tests on per-dataset differences (e.g. method A - method B).

All tests are two-sided. Zero differences are ties: the sign test and the
Wilcoxon test drop them, as usual.
"""

from __future__ import annotations

import math

import numpy as np


def sign_test_p(diff):
    """Two-sided exact sign test of P(diff < 0) = 0.5 (ties dropped)."""
    d = np.asarray(diff)
    d = d[d != 0]
    n = len(d)
    k = min(int(np.sum(d < 0)), int(np.sum(d > 0)))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


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


def wilcoxon_signed_rank(diff):
    """Two-sided Wilcoxon signed-rank test, normal approximation with tie
    correction (zero differences dropped). Returns ``(w_plus, z, p)``."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    n = len(d)
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    abs_sorted = np.abs(d)[order]
    i = 0
    while i < n:  # average ranks over ties
        j = i
        while j + 1 < n and abs_sorted[j + 1] == abs_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4
    _, counts = np.unique(abs_sorted, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - np.sum(counts ** 3 - counts) / 48
    z = (w_plus - mean) / math.sqrt(var)
    return w_plus, float(z), float(math.erfc(abs(z) / math.sqrt(2)))
