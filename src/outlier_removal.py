"""The custom outlier-removal training method ("ours").

Idea
----
Train a full-batch Adam linear regressor on the mixture data. Every
``reset_interval`` epochs, discard the samples with the largest absolute
residuals (the top ``outlier_ratio`` fraction) and re-initialise the weights
and optimizer state. Repeatedly pruning high-residual points and restarting
lets the model shed the ``z == 2`` outlier population and converge toward the
clean-population weights ``w1``.
"""

from __future__ import annotations

import numpy as np

from .train import initialize_weights


def ours(X, y, w_ref, *, init_type="random", learning_rate=0.01,
         num_epochs=1000, reset_interval=200, outlier_ratio=0.1,
         eval_X=None, eval_y=None, eval_mask=None,
         beta1=0.9, beta2=0.999, epsilon=1e-8, seed=0):
    """Periodic residual-cutoff + re-init full-batch Adam.

    Parameters
    ----------
    X, y : ndarray
        Full training data (a working copy is pruned over time).
    w_ref : ndarray
        Reference weights for the weight-error metric (the clean weights).
    init_type : {"random", "zero", "sparse"}
        Re-initialisation scheme applied at every reset.
    learning_rate : float
    num_epochs : int
    reset_interval : int
        Epochs between residual-based pruning + re-initialisation events.
    outlier_ratio : float
        Fraction of highest-residual samples removed at each pruning event.
    eval_X, eval_y, eval_mask : optional ndarray
        Evaluation data / boolean mask. Metrics are computed on the *original*
        (un-pruned) evaluation data. Default: ``X``/``y``/all-True.
    beta1, beta2, epsilon : float
        Adam hyper-parameters.
    seed : int

    Returns
    -------
    w : ndarray
        Final weights.
    est_history : list[float]
        Per-epoch estimation error on the (masked) evaluation set.
    weight_history : list[float]
        Per-epoch weight error ``||w - w_ref||``.

    Notes
    -----
    The Adam timestep ``t`` is reset to 0 on every re-initialisation (so bias
    correction restarts each cycle), exactly as in the original notebook.
    """
    np.random.seed(seed)
    D = X.shape[1]

    X_current = X.copy()
    y_current = y.copy()

    if eval_X is None:
        eval_X = X
    if eval_y is None:
        eval_y = y
    if eval_mask is None:
        eval_mask = np.ones(eval_X.shape[0], dtype=bool)

    est_history = []
    weight_history = []

    w = None
    m = v = None
    t = 0

    for epoch in range(1, num_epochs + 1):
        # (re-)initialise weights + optimizer state at start and every cycle
        if epoch == 1 or (epoch - 1) % reset_interval == 0:
            w = initialize_weights(init_type, D)
            m = np.zeros_like(w)
            v = np.zeros_like(w)
            t = 0

        # full-batch Adam update
        error = X_current @ w - y_current
        grad = (2.0 / X_current.shape[0]) * X_current.T @ error

        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # metrics on the original evaluation data
        y_pred_eval = eval_X @ w
        est_history.append(float(np.mean(
            (eval_y[eval_mask] - y_pred_eval[eval_mask]) ** 2)))
        weight_history.append(float(np.linalg.norm(w - w_ref)))

        # prune highest-residual samples every reset_interval epochs
        if epoch % reset_interval == 0 and epoch != num_epochs:
            residuals = np.abs(X_current @ w - y_current)
            cutoff = np.percentile(residuals, 100 * (1 - outlier_ratio))
            inlier_mask = residuals <= cutoff
            X_current = X_current[inlier_mask]
            y_current = y_current[inlier_mask]

    return w, est_history, weight_history
