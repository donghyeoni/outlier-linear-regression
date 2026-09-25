"""Descriptions of a sample selection using the true population labels.

The labels ``z`` (1 = clean, 2 = outlier) are used here only to describe
what a method did. No method uses them.
"""

from __future__ import annotations

import numpy as np


def true_noise(X, y, z, w1):
    """Noise of each sample under its own population's weights,
    ``y - x . w`` with ``w = w1`` (clean) or ``-w1`` (outlier)."""
    return y - np.where(z == 1, 1.0, -1.0) * (X @ w1)


def selection_summary(X, y, z, w1, kept_idx):
    """Counts describing a kept set.

    Returns a dict with ``clean_kept``, ``outliers_kept`` and the numbers of
    clean samples left out whose true noise is positive / negative
    (``clean_out_pos`` / ``clean_out_neg``).
    """
    in_kept = np.zeros(len(z), dtype=bool)
    in_kept[kept_idx] = True
    noise = true_noise(X, y, z, w1)
    clean_out = (z == 1) & ~in_kept
    return {
        "clean_kept": int(np.sum((z == 1) & in_kept)),
        "outliers_kept": int(np.sum((z == 2) & in_kept)),
        "clean_out_pos": int(np.sum(clean_out & (noise > 0))),
        "clean_out_neg": int(np.sum(clean_out & (noise < 0))),
    }
