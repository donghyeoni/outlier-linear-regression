"""Closed-form least squares and the weight-error metric."""

from __future__ import annotations

import numpy as np


def closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares solution ``argmin_w ||X w - y||^2`` via the
    Moore-Penrose pseudo-inverse, ``pinv(X) @ y``."""
    return np.linalg.pinv(X) @ y


def weight_error(w_pred: np.ndarray, w_ref: np.ndarray) -> float:
    """L2 norm of the weight-estimation error, ``||w_pred - w_ref||``."""
    return float(np.linalg.norm(w_pred - w_ref))
