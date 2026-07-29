"""Unified trainer and experiment-grid runner.

The two original notebooks each contained a near-identical training loop
(``run_experiment`` in notebook 1, ``gradient_descent`` in notebook 2). They
differed only in how the per-epoch metrics were recorded:

* notebook 1 recorded the batch training MSE (using the *pre-update*
  prediction) and the weight error against ``w_true``;
* notebook 2 recorded the MSE of the *post-update* full-data prediction
  restricted to the clean population (``z == 1``) and the weight error against
  the clean weights ``w1``.

Both behaviours are preserved here through the ``record`` / ``eval_*``
parameters, so a single :func:`train` reproduces either notebook exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimizers
from .optimizers import OPTIMIZERS

INIT_TYPES = ("random", "zero", "sparse")
BATCH_TYPES = ("full", "mini-batch", "SGD")


def initialize_weights(init_type: str, D: int) -> np.ndarray:
    """Initialise a weight vector.

    ``"random"`` draws from a standard normal, ``"zero"`` is all zeros, and
    ``"sparse"`` is a random vector with ~80% of its entries zeroed out.
    The RNG must already be seeded by the caller.
    """
    if init_type == "random":
        w = np.random.randn(D)
    elif init_type == "zero":
        w = np.zeros(D)
    elif init_type == "sparse":
        w = np.random.randn(D)
        w[np.random.rand(D) < 0.8] = 0
    else:
        raise ValueError(f"Unknown init_type: {init_type!r}. "
                         f"Expected one of {INIT_TYPES}.")
    return w


def select_batch(X: np.ndarray, y: np.ndarray, batch_type: str,
                 batch_size: int):
    """Return the ``(X_batch, y_batch)`` used for one update step."""
    N = X.shape[0]
    if batch_type == "full":
        return X, y
    if batch_type == "mini-batch":
        idx = np.random.choice(N, batch_size, replace=False)
        return X[idx], y[idx]
    if batch_type == "SGD":
        idx = np.random.randint(0, N)
        return X[idx:idx + 1], y[idx:idx + 1]
    raise ValueError(f"Unknown batch_type: {batch_type!r}. "
                     f"Expected one of {BATCH_TYPES}.")


def train(X, y, init_type, batch_type, optimizer_type, w_ref, *,
          learning_rate=0.1, num_epochs=1000, batch_size=32,
          record="batch", eval_X=None, eval_y=None, eval_mask=None,
          seed=0):
    """Train a linear model with a hand-coded optimizer.

    Parameters
    ----------
    X, y : ndarray
        Training data.
    init_type : {"random", "zero", "sparse"}
    batch_type : {"full", "mini-batch", "SGD"}
    optimizer_type : {"GD", "AdaGrad", "RMSProp", "Adam"}
    w_ref : ndarray
        Reference weights used for the weight-error metric.
    learning_rate, num_epochs, batch_size : see notebooks.
    record : {"batch", "eval"}
        ``"batch"`` (notebook 1) records the pre-update batch MSE.
        ``"eval"`` (notebook 2) records the post-update MSE evaluated on
        ``eval_X``/``eval_y`` restricted to ``eval_mask``.
    eval_X, eval_y, eval_mask : optional ndarray
        Evaluation data / boolean mask, required when ``record == "eval"``.
        Defaults to ``X``/``y``/all-True.
    seed : int
        Seed applied at the start of training (weight init + batch sampling).

    Returns
    -------
    w : ndarray
        Final weights.
    est_history : list[float]
        Per-epoch estimation error.
    weight_history : list[float]
        Per-epoch weight error (``||w - w_ref||``).
    """
    np.random.seed(seed)
    D = X.shape[1]

    w = initialize_weights(init_type, D)
    state = optimizers.init_state(w)

    if eval_X is None:
        eval_X = X
    if eval_y is None:
        eval_y = y
    if eval_mask is None:
        eval_mask = np.ones(eval_X.shape[0], dtype=bool)

    est_history = []
    weight_history = []

    for epoch in range(1, num_epochs + 1):
        X_batch, y_batch = select_batch(X, y, batch_type, batch_size)

        # pre-update prediction (used by notebook-1 style recording)
        y_pred_batch = X_batch @ w
        grad = optimizers.mse_gradient(X_batch, y_batch, w)

        w = optimizers.step(optimizer_type, w, grad, state, epoch,
                             learning_rate)

        if record == "batch":
            est_history.append(float(np.mean((y_batch - y_pred_batch) ** 2)))
        elif record == "eval":
            y_pred_eval = eval_X @ w
            est_history.append(float(np.mean(
                (eval_y[eval_mask] - y_pred_eval[eval_mask]) ** 2)))
        else:
            raise ValueError(f"Unknown record mode: {record!r}. "
                             f"Expected 'batch' or 'eval'.")

        weight_history.append(float(np.linalg.norm(w - w_ref)))

    return w, est_history, weight_history


def run_experiment(X, y, w_ref, *, inits=INIT_TYPES, batches=BATCH_TYPES,
                   optimizer_list=OPTIMIZERS, learning_rate=0.1,
                   num_epochs=1000, batch_size=32, record="batch",
                   eval_X=None, eval_y=None, eval_mask=None, seed=0):
    """Run the full ``init x batch x optimizer`` grid.

    Returns
    -------
    df : pandas.DataFrame
        Final estimation/weight error per configuration, sorted the same way
        as the notebooks (optimizer, then batch, then init).
    histories : dict
        ``{(optimizer, batch, init): {"estimation": [...], "weight": [...]}}``.
    """
    results = []
    histories = {}

    for init in inits:
        for batch in batches:
            for opt in optimizer_list:
                w, est_hist, w_hist = train(
                    X, y, init, batch, opt, w_ref,
                    learning_rate=learning_rate, num_epochs=num_epochs,
                    batch_size=batch_size, record=record,
                    eval_X=eval_X, eval_y=eval_y, eval_mask=eval_mask,
                    seed=seed)
                results.append({
                    "Init Type": init,
                    "Batch Type": batch,
                    "Optimizer": opt,
                    "Estimation Error": est_hist[-1],
                    "Weight Error": w_hist[-1],
                })
                histories[(opt, batch, init)] = {
                    "estimation": est_hist,
                    "weight": w_hist,
                }

    df = pd.DataFrame(results)[[
        "Optimizer", "Batch Type", "Init Type",
        "Estimation Error", "Weight Error"]]
    df["Optimizer"] = pd.Categorical(
        df["Optimizer"], categories=list(OPTIMIZERS), ordered=True)
    df = df.sort_values(
        by=["Optimizer", "Batch Type", "Init Type"]).reset_index(drop=True)
    return df, histories
