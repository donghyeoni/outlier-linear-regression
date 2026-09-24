"""Ablation of the two fixes to ``ours_v1`` on the development dataset
(seed 0; log E3).

Step 1 adds only the stopping rule (fixed 200-epoch cycles), sweeping its
threshold ``k``. Step 2 adds only per-cycle convergence (no stopping rule),
sweeping the gradient-norm tolerance. Each fix is tested alone so their
effects are not mixed.

Writes ``results/ablation/step1_stop_rule.json`` and
``results/ablation/step2_converge.json``.

Usage
-----
    python experiments/run_ablation.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "ablation")

LEARNING_RATES = (0.01, 0.1, 0.5)


def run(X, y, z, w1, **kwargs):
    removed_outliers = []
    w, _, w_hist, info = ours_v2(
        X, y, w1, eval_X=X, eval_y=y, eval_mask=(z == 1), seed=0,
        on_prune=lambda ep, rem, kept: removed_outliers.append(
            int(np.sum(z[rem] == 2))),
        **kwargs)
    kept = info["kept_idx"]
    return {
        "final_weight_error": w_hist[-1],
        "total_epochs": len(w_hist),
        "cycle_epochs": info["cycle_epochs"],
        "stopped_at_cycle": info["stopped_at_cycle"],
        "outliers_removed_per_prune": removed_outliers,
        "kept": int(len(kept)),
        "outliers_left": int(np.sum(z[kept] == 2)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=0)

    print("=== Step 1: stopping rule only (fixed 200-epoch cycles) ===")
    step1 = []
    for lr in LEARNING_RATES:
        for k in (None, 2.5, 3.0, 3.5, 4.0, 5.0):
            r = {"learning_rate": lr, "stop_k": k,
                 **run(X, y, z, w1, learning_rate=lr, stop_k=k)}
            step1.append(r)
            print(f"  lr={lr:<4} k={str(k):<4} final {r['final_weight_error']:.4f}"
                  f"  removed outliers {r['outliers_removed_per_prune']}"
                  f"  kept {r['kept']} (outliers left {r['outliers_left']})")
    with open(os.path.join(OUT_DIR, "step1_stop_rule.json"), "w") as f:
        json.dump(step1, f, indent=2)

    print("\n=== Step 2: per-cycle convergence only (no stopping rule) ===")
    step2 = []
    for lr in LEARNING_RATES:
        for tol in (1e-3, 1e-4, 1e-5, 1e-6):
            r = {"learning_rate": lr, "converge_tol": tol,
                 **run(X, y, z, w1, learning_rate=lr, converge_tol=tol)}
            step2.append(r)
            print(f"  lr={lr:<4} tol={tol:<6g} final {r['final_weight_error']:.4f}"
                  f"  epochs/cycle {r['cycle_epochs']}"
                  f"  removed outliers {r['outliers_removed_per_prune']}")
    with open(os.path.join(OUT_DIR, "step2_converge.json"), "w") as f:
        json.dump(step2, f, indent=2)

    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
