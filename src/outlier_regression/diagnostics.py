"""Diagnostics of an inlier selection, using the true population labels.

The labels ``z`` (1 = clean, 2 = outlier) are used here only to *describe*
what a method did. No method uses them.
"""

from __future__ import annotations

import numpy as np


def true_noise(X, y, z, w1):
    """Noise of each sample under its own population's weights,
    ``epsilon = y - x . w`` with ``w = w1`` (clean) or ``-w1`` (outlier)."""
    return y - np.where(z == 1, 1.0, -1.0) * (X @ w1)


def excluded_clean_noise_signs(noise, z, kept_idx):
    """Counts of clean samples *not* in ``kept_idx``, split by noise sign.

    Returns ``(n_positive, n_negative)``.
    """
    excluded = np.setdiff1d(np.flatnonzero(z == 1), kept_idx)
    return int(np.sum(noise[excluded] > 0)), int(np.sum(noise[excluded] < 0))


def outlier_xw1(X, w1, z, kept_idx):
    """``x . w1`` of the outliers that were kept and of those removed.

    Returns ``(kept_values, removed_values)`` as arrays.
    """
    outliers = np.flatnonzero(z == 2)
    kept = np.isin(outliers, kept_idx)
    xw = X[outliers] @ w1
    return xw[kept], xw[~kept]


def set_changes(sets):
    """Number of samples that change between consecutive inlier sets."""
    return [int(len(np.setxor1d(np.asarray(list(b)), np.asarray(list(a)))))
            for a, b in zip(sets[:-1], sets[1:])]
