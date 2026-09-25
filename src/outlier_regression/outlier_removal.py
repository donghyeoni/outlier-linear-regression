"""Outlier-removal training methods.

* :func:`ours_v1` -- the starting method: fixed-length cycles of full-batch
  Adam, removal of the top ``outlier_ratio`` fraction of absolute residuals
  after every cycle but the last, re-initialisation.
* :func:`ours_v2` -- the general implementation used in development: optional
  per-cycle convergence, a MAD-based stopping rule, and the ratio / threshold
  / readmit pruning rules.
* :func:`ours` -- the final method, one setting of :func:`ours_v2`.

All methods train on residuals only; the population labels are never used.
"""

from __future__ import annotations

import numpy as np

from . import optimizers
from .train import initialize_weights

PRUNE_RULES = ("ratio", "threshold", "readmit")


def robust_scale(r):
    """MAD estimate of the residual standard deviation,
    ``1.4826 * median(|r - median(r)|)``."""
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def _adam_cycle(X, y, w_ref, w, learning_rate, max_iters, tol, history):
    """Full-batch Adam from a fresh state until ``||grad|| < tol`` (if
    ``tol`` is given) or ``max_iters`` updates. Appends the weight error
    after each update to ``history``. Returns ``(w, n_iters, grad_norm)``
    with the gradient norm at the returned weights."""
    state = optimizers.init_state(w)
    n = 0
    grad = optimizers.mse_gradient(X, y, w)
    while n < max_iters and not (tol is not None
                                 and np.linalg.norm(grad) < tol):
        n += 1
        w = optimizers.step("Adam", w, grad, state, n, learning_rate)
        history.append(float(np.linalg.norm(w - w_ref)))
        grad = optimizers.mse_gradient(X, y, w)
    return w, n, float(np.linalg.norm(grad))


def ours_v1(X, y, w_ref, *, learning_rate=0.01, num_iters=1000,
            reset_interval=200, outlier_ratio=0.1, seed=0):
    """Periodic residual cutoff with re-initialisation.

    Runs ``num_iters / reset_interval`` cycles of ``reset_interval``
    full-batch Adam updates. Each cycle starts from freshly initialised
    ``N(0, 1)`` weights (global RNG seeded once with ``seed``). After every
    cycle but the last, the ``outlier_ratio`` fraction of kept samples with
    the largest ``|residual|`` is removed permanently.

    Returns
    -------
    w : ndarray
    weight_history : list[float]
        ``||w - w_ref||`` after every update.
    info : dict
        ``kept_idx`` (final kept indices), ``removed`` (list of removed index
        arrays, one per prune), ``grad_norm`` (gradient norm at the end of
        each cycle).
    """
    return ours_v2(X, y, w_ref, prune_rule="ratio",
                   learning_rate=learning_rate,
                   max_cycles=num_iters // reset_interval,
                   cycle_iters=reset_interval, converge_tol=None,
                   outlier_ratio=outlier_ratio, stop_k=None, seed=seed)


def ours_v2(X, y, w_ref, *, prune_rule="ratio", learning_rate=0.1,
            max_cycles=5, cycle_iters=200, converge_tol=None,
            max_cycle_iters=20000, outlier_ratio=0.1, stop_k=None, seed=0):
    """Cyclic full-batch Adam with residual-based sample selection.

    Every cycle re-initialises the weights (``N(0, 1)``, global RNG seeded
    once with ``seed``) and the Adam state, then trains on the current kept
    set: for ``cycle_iters`` updates if ``converge_tol`` is None, otherwise
    until ``||grad|| < converge_tol`` (at most ``max_cycle_iters`` updates).
    After training, the kept set for the next cycle is chosen by
    ``prune_rule``:

    ``"ratio"``
        Remove the ``outlier_ratio`` fraction of kept samples with the
        largest ``|r|``. With ``stop_k``, first check the stopping rule: if
        no kept sample has ``|r| > stop_k * robust_scale(r_kept)``, stop.
    ``"threshold"``
        Remove the kept samples with ``|r| > stop_k * robust_scale(r_kept)``;
        stop when there are none.
    ``"readmit"``
        Select, from *all* samples, those with
        ``|r| <= stop_k * robust_scale(r_kept)``; samples removed earlier can
        return. Stop when the selected set equals the current kept set.

    The check is also made after the last cycle, but no further cycle is
    trained. Residuals are ``r = X w - y`` under the weights just trained.

    Returns
    -------
    w : ndarray
        Weights trained in the last cycle.
    weight_history : list[float]
        ``||w - w_ref||`` after every update.
    info : dict
        ``kept_idx`` (set used in the last cycle), ``kept_sets`` (set used
        in each cycle), ``removed`` (for ratio / threshold: indices removed
        at each prune), ``cycle_iters`` and ``grad_norm`` (per cycle),
        ``settled`` (True if the rule found nothing left to change after the
        last trained cycle; None for ratio without ``stop_k``).
    """
    if prune_rule not in PRUNE_RULES:
        raise ValueError(f"Unknown prune_rule: {prune_rule!r}. "
                         f"Expected one of {PRUNE_RULES}.")
    if prune_rule != "ratio" and stop_k is None:
        raise ValueError(f"prune_rule={prune_rule!r} requires stop_k.")

    np.random.seed(seed)
    N, D = X.shape
    kept = np.arange(N)
    history = []
    info = {"kept_sets": [], "removed": [], "cycle_iters": [],
            "grad_norm": [], "settled": None if (prune_rule == "ratio"
                                                 and stop_k is None) else False}
    budget = cycle_iters if converge_tol is None else max_cycle_iters

    for cycle in range(max_cycles):
        w0 = initialize_weights("random", D)
        w, n, g = _adam_cycle(X[kept], y[kept], w_ref, w0, learning_rate,
                              budget, converge_tol, history)
        info["kept_sets"].append(kept)
        info["cycle_iters"].append(n)
        info["grad_norm"].append(g)
        last = cycle == max_cycles - 1

        r_kept = X[kept] @ w - y[kept]
        if prune_rule == "readmit":
            new = np.flatnonzero(np.abs(X @ w - y)
                                 <= stop_k * robust_scale(r_kept))
            if np.array_equal(new, kept):
                info["settled"] = True
                break
            if last:
                break
            kept = new
            continue

        if stop_k is not None:
            flagged = np.abs(r_kept) > stop_k * robust_scale(r_kept)
            if not np.any(flagged):
                info["settled"] = True
                break
        if last:
            break
        if prune_rule == "threshold":
            keep_mask = ~flagged
        else:
            cutoff = np.percentile(np.abs(r_kept), 100 * (1 - outlier_ratio))
            keep_mask = np.abs(r_kept) <= cutoff
        info["removed"].append(kept[~keep_mask])
        kept = kept[keep_mask]

    info["kept_idx"] = kept
    return w, history, info


FINAL_CONFIG = {"prune_rule": "readmit", "stop_k": 3.5, "converge_tol": 1e-5,
                "max_cycles": 50, "max_cycle_iters": 20000,
                "learning_rate": 0.1}


def ours(X, y, w_ref, **kwargs):
    """The final method: :func:`ours_v2` with :data:`FINAL_CONFIG`.

    Keywords of :func:`ours_v2` override a setting (e.g. ``learning_rate``).
    """
    return ours_v2(X, y, w_ref, **{**FINAL_CONFIG, **kwargs})
