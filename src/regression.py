"""Closed-form linear-regression solvers (normal equation).

Provides the analytic least-squares solution used as the reference baseline
that the iterative optimizers are compared against.
"""

from __future__ import annotations

import numpy as np


def closed_form_solution(X: np.ndarray, y: np.ndarray,
                         method: str = "pinv") -> np.ndarray:
    """Solve ``min_w ||X w - y||^2`` in closed form.

    Parameters
    ----------
    X : (N, D) ndarray
    y : (N,) ndarray
    method : {"pinv", "inv"}
        ``"pinv"`` uses ``np.linalg.pinv(X) @ y`` (numerically robust,
        as in notebook 2).
        ``"inv"`` uses ``np.linalg.inv(X.T @ X) @ X.T @ y`` (the explicit
        normal equation, as in notebook 1).

    Returns
    -------
    (D,) ndarray
        The least-squares weight estimate.
    """
    if method == "pinv":
        return np.linalg.pinv(X) @ y
    if method == "inv":
        return np.linalg.inv(X.T @ X) @ X.T @ y
    raise ValueError(f"Unknown method: {method!r}. Expected 'pinv' or 'inv'.")


def estimation_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean-squared prediction error."""
    return float(np.mean((y_true - y_pred) ** 2))


def weight_error(w_pred: np.ndarray, w_true: np.ndarray) -> float:
    """L2 norm of the weight-estimation error."""
    return float(np.linalg.norm(w_pred - w_true))
