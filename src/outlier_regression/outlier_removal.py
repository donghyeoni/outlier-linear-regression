"""Outlier-removal training methods.

* :func:`ours` -- the final method: full-batch Adam trained to convergence in
  each cycle, with the inlier set re-selected from all samples as
  ``|r| <= 3 * sigma_MAD`` until it no longer changes (at most 5 cycles).
* :func:`ours_v2` -- the general implementation used during development. Its
  options cover the stopping rule, per-cycle convergence and the ratio /
  threshold / readmit pruning rules; :func:`ours` is one setting of it.
* :func:`ours_v1` -- the first version: fixed-length cycles, removal of the
  top ``outlier_ratio`` fraction of absolute residuals, re-initialisation.
"""

from __future__ import annotations

import numpy as np

from . import optimizers
from .train import initialize_weights


def ours_v1(X, y, w_ref, *, init_type="random", learning_rate=0.01,
         num_epochs=1000, reset_interval=200, outlier_ratio=0.1,
         eval_X=None, eval_y=None, eval_mask=None,
         beta1=0.9, beta2=0.999, epsilon=1e-8, seed=0, on_prune=None):
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
    on_prune : callable, optional
        Called as ``on_prune(epoch, removed_idx, kept_idx)`` after every
        pruning event, with indices into the *original* ``X``. Used for
        diagnostics (e.g. how many true outliers each prune removes).

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
    The Adam timestep ``t`` is reset to 0 on every re-initialisation, so bias
    correction restarts each cycle.
    """
    np.random.seed(seed)
    D = X.shape[1]

    kept_idx = np.arange(X.shape[0])

    if eval_X is None:
        eval_X = X
    if eval_y is None:
        eval_y = y
    if eval_mask is None:
        eval_mask = np.ones(eval_X.shape[0], dtype=bool)

    est_history = []
    weight_history = []

    w = None
    state = None
    t = 0

    for epoch in range(1, num_epochs + 1):
        # (re-)initialise weights + optimizer state at start and every cycle
        if epoch == 1 or (epoch - 1) % reset_interval == 0:
            w = initialize_weights(init_type, D)
            state = optimizers.init_state(w)
            t = 0

        # full-batch Adam update on the surviving samples
        X_current, y_current = X[kept_idx], y[kept_idx]
        grad = optimizers.mse_gradient(X_current, y_current, w)
        t += 1
        w = optimizers.step("Adam", w, grad, state, t, learning_rate,
                            beta1=beta1, beta2=beta2, epsilon=epsilon)

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
            removed_idx = kept_idx[~inlier_mask]
            kept_idx = kept_idx[inlier_mask]
            if on_prune is not None:
                on_prune(epoch, removed_idx, kept_idx)

    return w, est_history, weight_history


def robust_scale(r):
    """MAD-based estimate of the residual standard deviation.

    ``1.4826 * median(|r - median(r)|)`` is consistent for Gaussian noise and
    tolerates up to 50% contamination, so it can be computed while outliers
    are still in the data.
    """
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def ours_v2(X, y, w_ref, *, init_type="random", learning_rate=0.1,
            num_cycles=5, cycle_epochs=200, converge_tol=None,
            max_cycle_epochs=20000, outlier_ratio=0.1, stop_k=None,
            prune_rule="ratio",
            eval_X=None, eval_y=None, eval_mask=None,
            beta1=0.9, beta2=0.999, epsilon=1e-8, seed=0, on_prune=None):
    """:func:`ours_v1` extended with a stopping rule and per-cycle convergence.

    Training is organised in cycles. Each cycle re-initialises the weights
    and optimizer state, trains full-batch Adam on the surviving samples, and
    (except after the last cycle) prunes high-residual samples.

    Parameters
    ----------
    num_cycles : int
        Maximum number of training cycles (pruning happens between cycles).
    cycle_epochs : int
        Epochs per cycle when ``converge_tol`` is None (fixed-length cycles).
    converge_tol : float, optional
        If given, each cycle instead runs until the gradient norm drops below
        this value (capped at ``max_cycle_epochs``).
    stop_k : float, optional
        Stopping rule. Before each prune, if no surviving sample has
        ``|residual| > stop_k * robust_scale(residuals)``, pruning stops. The
        remaining fixed-length budget (if any) is then spent training the
        current weights without re-initialisation.
    prune_rule : {"ratio", "threshold", "readmit"}
        ``"ratio"`` removes the ``outlier_ratio`` fraction with the largest
        absolute residuals. ``"threshold"`` removes exactly the samples that
        fail the stopping rule, ``|residual| > stop_k * robust_scale``
        (requires ``stop_k``), so pruning ends once none are left.
        ``"readmit"`` recomputes the inlier set from *all* samples after each
        cycle: residuals of every sample under the current fit are compared
        with ``stop_k * robust_scale`` of the current inliers' residuals, so
        wrongly removed samples can return. Training stops when the inlier
        set no longer changes (requires ``stop_k``; the ratio stopping check
        is not used).
    on_prune : callable, optional
        ``on_prune(epoch, removed_idx, kept_idx)`` with original indices.

    With ``converge_tol=None``, ``stop_k=None``, ``prune_rule="ratio"`` and 5
    cycles of 200 epochs (the defaults), this reproduces :func:`ours_v1` with
    ``num_epochs=1000`` *at the same* ``learning_rate``. Note that the default
    ``learning_rate`` differs (0.1 here, 0.01 in :func:`ours_v1`).

    ``outlier_ratio`` is used only by ``prune_rule="ratio"``. ``cycle_epochs``
    is used only when ``converge_tol`` is None, and ``max_cycle_epochs`` only
    when it is set.

    Returns
    -------
    w, est_history, weight_history
        As in :func:`ours_v1`.
    info : dict
        ``cycle_epochs`` (epochs actually run per cycle), ``stopped_at_cycle``
        (index of the cycle after which the stopping rule fired, or None) and
        ``kept_idx`` (surviving original indices).
    """
    if prune_rule not in ("ratio", "threshold", "readmit"):
        raise ValueError(f"Unknown prune_rule: {prune_rule!r}. "
                         f"Expected 'ratio', 'threshold' or 'readmit'.")
    if prune_rule in ("threshold", "readmit") and stop_k is None:
        raise ValueError(f"prune_rule={prune_rule!r} requires stop_k.")

    np.random.seed(seed)
    D = X.shape[1]
    kept_idx = np.arange(X.shape[0])

    if eval_X is None:
        eval_X = X
    if eval_y is None:
        eval_y = y
    if eval_mask is None:
        eval_mask = np.ones(eval_X.shape[0], dtype=bool)

    est_history = []
    weight_history = []
    info = {"cycle_epochs": [], "stopped_at_cycle": None}
    epoch = 0

    def train_epochs(w, state, t, max_epochs, tol):
        nonlocal epoch
        X_cur, y_cur = X[kept_idx], y[kept_idx]
        n = 0
        while n < max_epochs:
            grad = optimizers.mse_gradient(X_cur, y_cur, w)
            if tol is not None and np.linalg.norm(grad) < tol:
                break
            t += 1
            n += 1
            epoch += 1
            w = optimizers.step("Adam", w, grad, state, t, learning_rate,
                                beta1=beta1, beta2=beta2, epsilon=epsilon)
            y_pred_eval = eval_X @ w
            est_history.append(float(np.mean(
                (eval_y[eval_mask] - y_pred_eval[eval_mask]) ** 2)))
            weight_history.append(float(np.linalg.norm(w - w_ref)))
        return w, t, n

    budget = cycle_epochs if converge_tol is None else max_cycle_epochs
    w = None
    for cycle in range(num_cycles):
        w = initialize_weights(init_type, D)
        state = optimizers.init_state(w)
        w, t, n = train_epochs(w, state, 0, budget, converge_tol)
        info["cycle_epochs"].append(n)

        if cycle == num_cycles - 1:
            break

        if prune_rule == "readmit":
            scale = robust_scale(X[kept_idx] @ w - y[kept_idx])
            new_kept = np.flatnonzero(np.abs(X @ w - y) <= stop_k * scale)
            if np.array_equal(new_kept, kept_idx):
                info["stopped_at_cycle"] = cycle
                break
            removed_idx = np.setdiff1d(np.arange(X.shape[0]), new_kept)
            kept_idx = new_kept
            if on_prune is not None:
                on_prune(epoch, removed_idx, kept_idx)
            continue

        residuals = X[kept_idx] @ w - y[kept_idx]
        abs_r = np.abs(residuals)
        if stop_k is not None:
            flagged = abs_r > stop_k * robust_scale(residuals)
        if stop_k is not None and not np.any(flagged):
            info["stopped_at_cycle"] = cycle
            if converge_tol is None:
                remaining = (num_cycles - cycle - 1) * cycle_epochs
                w, t, n = train_epochs(w, state, t, remaining, None)
                info["cycle_epochs"][-1] += n
            break

        if prune_rule == "threshold":
            inlier_mask = ~flagged
        else:
            cutoff = np.percentile(abs_r, 100 * (1 - outlier_ratio))
            inlier_mask = abs_r <= cutoff
        removed_idx = kept_idx[~inlier_mask]
        kept_idx = kept_idx[inlier_mask]
        if on_prune is not None:
            on_prune(epoch, removed_idx, kept_idx)

    info["kept_idx"] = kept_idx
    return w, est_history, weight_history, info


FINAL_CONFIG = {"prune_rule": "readmit", "stop_k": 3.0, "converge_tol": 1e-5,
                "num_cycles": 5, "learning_rate": 0.1}


def ours(X, y, w_ref, **kwargs):
    """The final method.

    Equivalent to ``ours_v2(X, y, w_ref, prune_rule="readmit", stop_k=3,
    converge_tol=1e-5, num_cycles=5, learning_rate=0.1)``. Any keyword of
    :func:`ours_v2` can be passed to override a setting (e.g.
    ``learning_rate``) or to add evaluation data / callbacks.

    Returns
    -------
    w, est_history, weight_history, info
        As in :func:`ours_v2`.
    """
    return ours_v2(X, y, w_ref, **{**FINAL_CONFIG, **kwargs})
