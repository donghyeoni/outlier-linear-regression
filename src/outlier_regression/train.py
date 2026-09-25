"""Trainer and optimizer-grid runner.

One iteration is one parameter update. Full batch uses all samples,
mini-batch 32 samples drawn without replacement, SGD one sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimizers
from .optimizers import OPTIMIZERS

INIT_TYPES = ("random", "zero", "sparse")
BATCH_TYPES = ("full", "mini-batch", "SGD")


def initialize_weights(init_type: str, D: int) -> np.ndarray:
    """Initialise a weight vector from the global NumPy RNG.

    ``"random"`` draws from ``N(0, 1)``, ``"zero"`` is all zeros, and
    ``"sparse"`` draws from ``N(0, 1)`` and sets each entry to 0 with
    probability 0.8.
    """
    if init_type == "random":
        return np.random.randn(D)
    if init_type == "zero":
        return np.zeros(D)
    if init_type == "sparse":
        w = np.random.randn(D)
        w[np.random.rand(D) < 0.8] = 0
        return w
    raise ValueError(f"Unknown init_type: {init_type!r}. "
                     f"Expected one of {INIT_TYPES}.")


def select_batch(X, y, batch_type: str, batch_size: int):
    """Return the ``(X_batch, y_batch)`` used for one update."""
    N = X.shape[0]
    if batch_type == "full":
        return X, y
    if batch_type == "mini-batch":
        idx = np.random.choice(N, batch_size, replace=False)
        return X[idx], y[idx]
    if batch_type == "SGD":
        i = np.random.randint(0, N)
        return X[i:i + 1], y[i:i + 1]
    raise ValueError(f"Unknown batch_type: {batch_type!r}. "
                     f"Expected one of {BATCH_TYPES}.")


def train(X, y, init_type, batch_type, optimizer_type, w_ref, *,
          learning_rate=0.1, num_iters=1000, batch_size=32, seed=0):
    """Train a linear model on the MSE loss with a hand-coded optimizer.

    The global NumPy RNG is seeded with ``seed`` before the weights are
    initialised; batch sampling continues from the same stream.

    Returns
    -------
    w : (D,) ndarray
        Final weights.
    weight_history : list[float]
        ``||w - w_ref||`` after every iteration.
    """
    np.random.seed(seed)
    w = initialize_weights(init_type, X.shape[1])
    state = optimizers.init_state(w)
    weight_history = []
    for t in range(1, num_iters + 1):
        X_b, y_b = select_batch(X, y, batch_type, batch_size)
        grad = optimizers.mse_gradient(X_b, y_b, w)
        w = optimizers.step(optimizer_type, w, grad, state, t, learning_rate)
        weight_history.append(float(np.linalg.norm(w - w_ref)))
    return w, weight_history


def run_grid(X, y, w_ref, w_star, *, learning_rates=(0.01, 0.1),
             num_iters=1000, batch_size=32, seed=0, keep_histories=False):
    """Run every ``optimizer x batch x init x learning rate`` combination.

    Parameters
    ----------
    w_ref : ndarray
        Reference weights of the weight error.
    w_star : ndarray
        Closed-form least-squares solution on ``(X, y)``; the distance
        ``||w - w_star||`` measures how close an optimizer got to the
        minimiser of its loss.

    Returns
    -------
    df : DataFrame
        One row per combination with ``weight_error`` and ``dist_to_star``.
    histories : dict
        ``{(optimizer, batch, init, lr): weight_history}`` if
        ``keep_histories``, else empty.
    """
    rows, histories = [], {}
    for lr in learning_rates:
        for opt in OPTIMIZERS:
            for batch in BATCH_TYPES:
                for init in INIT_TYPES:
                    w, hist = train(X, y, init, batch, opt, w_ref,
                                    learning_rate=lr, num_iters=num_iters,
                                    batch_size=batch_size, seed=seed)
                    rows.append({
                        "optimizer": opt, "batch": batch, "init": init,
                        "lr": lr, "weight_error": hist[-1],
                        "dist_to_star": float(np.linalg.norm(w - w_star)),
                    })
                    if keep_histories:
                        histories[(opt, batch, init, lr)] = hist
    return pd.DataFrame(rows), histories
