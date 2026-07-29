"""Synthetic data generators.

Everything here is generated in-code from a fixed seed, so results are fully
reproducible and there is nothing to download.
"""

from __future__ import annotations

import numpy as np


def generate_linear_data(N: int = 1000, D: int = 4, noise_std: float = 0.1,
                         seed: int = 0):
    """Generate a clean synthetic linear-regression dataset.

    The model is ``y = X @ w_true + noise`` with features drawn uniformly on
    ``[0, 1)`` and Gaussian noise.

    Parameters
    ----------
    N, D : int
        Number of samples and features.
    noise_std : float
        Standard deviation multiplier of the additive Gaussian noise.
    seed : int
        Seed for ``numpy.random`` (fixed for reproducibility).

    Returns
    -------
    X : (N, D) ndarray
    y : (N,) ndarray
    w_true : (D,) ndarray
        The ground-truth weight vector.

    Notes
    -----
    The exact NumPy RNG call order (``rand`` -> ``rand`` -> ``randn``) is kept
    identical to the original notebook so the produced numbers match.
    """
    np.random.seed(seed)
    X = np.random.rand(N, D)
    w_true = np.random.rand(D)
    noise = noise_std * np.random.randn(N)
    y = X @ w_true + noise
    return X, y, w_true


def generate_mixture_data(N: int = 1000, D: int = 4, p: float = 0.9,
                          noise_std: float = 0.1, seed: int = 0):
    """Generate a two-population mixture dataset.

    A fraction ``p`` of the samples follow weight vector ``w1`` (the clean
    population, ``z == 1``); the remaining ``1 - p`` follow ``w2 = -w1`` (the
    outlier population, ``z == 2``).

    Parameters
    ----------
    N, D : int
        Number of samples and features.
    p : float
        Probability that a sample belongs to the clean population.
    noise_std : float
        Standard deviation multiplier of the additive Gaussian noise.
    seed : int
        Seed for ``numpy.random`` (fixed for reproducibility).

    Returns
    -------
    X : (N, D) ndarray
    y : (N,) ndarray
    z : (N,) int ndarray
        Population label per sample: ``1`` = clean, ``2`` = outlier.
    w1 : (D,) ndarray
        Ground-truth weights of the clean population.
    w2 : (D,) ndarray
        Ground-truth weights of the outlier population (``= -w1``).

    Notes
    -----
    The per-sample assignment loop is preserved verbatim so the RNG stream and
    resulting dataset are identical to the original notebook.
    """
    np.random.seed(seed)
    X = np.random.rand(N, D)
    w1 = np.random.rand(D)
    w2 = -w1
    noise = noise_std * np.random.randn(N)

    y = np.zeros(N)
    z = np.zeros(N, dtype=int)
    for i in range(N):
        if np.random.rand() < p:
            y[i] = X[i] @ w1 + noise[i]
            z[i] = 1
        else:
            y[i] = X[i] @ w2 + noise[i]
            z[i] = 2
    return X, y, z, w1, w2
