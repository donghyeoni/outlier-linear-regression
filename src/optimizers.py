"""Hand-coded gradient-based optimizer update steps (pure NumPy).

The four optimizers -- vanilla Gradient Descent, AdaGrad, RMSProp and Adam --
share the same MSE gradient; only their per-step parameter update differs.
This module isolates that update logic so a single trainer (``src.train``) can
drive any of them.
"""

from __future__ import annotations

import numpy as np

OPTIMIZERS = ("GD", "AdaGrad", "RMSProp", "Adam")


def mse_gradient(X_batch: np.ndarray, y_batch: np.ndarray,
                 w: np.ndarray) -> np.ndarray:
    """Gradient of the mean-squared-error loss w.r.t. ``w`` on a batch.

    ``grad = (2 / n) * X_batch.T @ (X_batch @ w - y_batch)``
    """
    error = X_batch @ w - y_batch
    return (2.0 / X_batch.shape[0]) * X_batch.T @ error


def init_state(w: np.ndarray) -> dict:
    """Create a fresh optimizer state for weight vector ``w``.

    Each accumulator is an independent zero array (this fixes the aliasing in
    the original notebook where ``m = v = grad_squared = running_avg`` all
    referenced the *same* array object).
    """
    return {
        "m": np.zeros_like(w),             # Adam first moment
        "v": np.zeros_like(w),             # Adam second moment
        "grad_squared": np.zeros_like(w),  # AdaGrad accumulator
        "running_avg": np.zeros_like(w),   # RMSProp accumulator
    }


def step(optimizer_type: str, w: np.ndarray, grad: np.ndarray, state: dict,
         epoch: int, learning_rate: float,
         beta1: float = 0.9, beta2: float = 0.999,
         epsilon: float = 1e-8) -> np.ndarray:
    """Apply one optimizer update and return the new weights.

    Parameters
    ----------
    optimizer_type : {"GD", "AdaGrad", "RMSProp", "Adam"}
    w : (D,) ndarray
        Current weights.
    grad : (D,) ndarray
        Loss gradient at ``w``.
    state : dict
        Mutable optimizer state from :func:`init_state`.
    epoch : int
        1-based step counter (used for Adam bias correction).
    learning_rate : float
    beta1, beta2, epsilon : float
        Adam / RMSProp hyper-parameters.

    Returns
    -------
    (D,) ndarray
        The updated weights.
    """
    if optimizer_type == "GD":
        w = w - learning_rate * grad

    elif optimizer_type == "AdaGrad":
        state["grad_squared"] += grad ** 2
        w = w - learning_rate * grad / (np.sqrt(state["grad_squared"]) + epsilon)

    elif optimizer_type == "RMSProp":
        state["running_avg"] = 0.9 * state["running_avg"] + 0.1 * (grad ** 2)
        w = w - learning_rate * grad / (np.sqrt(state["running_avg"]) + epsilon)

    elif optimizer_type == "Adam":
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
        m_hat = state["m"] / (1 - beta1 ** epoch)
        v_hat = state["v"] / (1 - beta2 ** epoch)
        w = w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}. "
                         f"Expected one of {OPTIMIZERS}.")

    return w
