"""Why ``ours_v1`` (lr 0.01) has a near-oracle MSE but a large
weight error (experiment log, E2).

Decomposes the weight error ``w - w1`` into its component along the
all-ones direction and the orthogonal remainder. Computes the eigenvalues of
the clean-input second-moment matrix ``S = E[x x^T]``, and checks that the
excess MSE ``(w - w_oracle)^T S (w - w_oracle)`` equals the observed MSE gap
between "ours" and the oracle.

Writes ``results/outlier/mse_vs_weight_error.json``.

Usage
-----
    python experiments/analyze_mse_vs_weight.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v1
from outlier_regression.regression import closed_form_solution

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "outlier")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=0)
    clean = z == 1
    # same configuration as the reported "ours" result (run_outlier.py Part 4)
    w, est_hist, _ = ours_v1(X, y, w1, init_type="random",
                             learning_rate=0.01, num_epochs=1000,
                             reset_interval=200, outlier_ratio=0.1,
                             eval_X=X, eval_y=y, eval_mask=clean, seed=0)
    w_oracle = closed_form_solution(X[clean], y[clean], method="pinv")

    dw = w - w1
    ones = np.ones_like(w) / np.sqrt(len(w))
    along = float(dw @ ones)
    S = X[clean].T @ X[clean] / int(np.sum(clean))
    mse_ours = est_hist[-1]
    mse_oracle = float(np.mean((y[clean] - X[clean] @ w_oracle) ** 2))
    d_or = w - w_oracle

    result = {
        "config": {"learning_rate": 0.01, "num_epochs": 1000, "seed": 0},
        "weight_error": float(np.linalg.norm(dw)),
        "component_along_ones": along,
        "component_orthogonal_norm": float(np.linalg.norm(dw - along * ones)),
        "S_eigenvalues": [float(v) for v in np.linalg.eigvalsh(S)],
        "excess_mse_vs_w1": float(dw @ S @ dw),
        "excess_mse_vs_oracle": float(d_or @ S @ d_or),
        "mse_ours": float(mse_ours),
        "mse_oracle": mse_oracle,
        "mse_gap": float(mse_ours - mse_oracle),
    }
    for k, v in result.items():
        print(f"{k:28s} {v}")
    with open(os.path.join(OUT_DIR, "mse_vs_weight_error.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
