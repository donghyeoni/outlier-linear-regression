"""Reproduces notebook 2: outlier study on a two-population mixture.

Parts
-----
1. Oracle fit on the clean population only (``z == 1``).
2. Naive fit on all data (contaminated by the outlier population).
3. Optimizer grid (GD/AdaGrad/RMSProp/Adam) evaluated on the clean population.
4. "ours": periodic residual-cutoff + re-init Adam.
5. Learning-rate sweep of "ours" (weight-error convergence plot).

Usage
-----
    python experiments/run_outlier.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_mixture_data
from src.regression import closed_form_solution, estimation_error, weight_error
from src.train import run_experiment
from src.outlier_removal import ours
from src.plots import plot_curves

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "outlier")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- data ---------------------------------------------------------------
    X, y, z, w1, w2 = generate_mixture_data(N=1000, D=4, p=0.9, seed=0)
    clean_mask = z == 1

    # --- Part 1: oracle fit on the clean population only --------------------
    w_y1 = closed_form_solution(X[clean_mask], y[clean_mask], method="pinv")
    y_pred = X @ w_y1
    part1 = {
        "estimation_error_clean": float(estimation_error(y[clean_mask], y_pred[clean_mask])),
        "weight_error_l2": float(weight_error(w_y1, w1)),
    }
    print("=== Part 1: fit on clean population (z==1) only ===")
    print(f"Estimation Error (z==1 only): {part1['estimation_error_clean']:.6f}")
    print(f"Weight Error (L2 norm):      {part1['weight_error_l2']:.6f}\n")

    # --- Part 2: naive fit on all (contaminated) data -----------------------
    w_all = closed_form_solution(X, y, method="pinv")
    y_pred = X @ w_all
    part2 = {
        "estimation_error_clean": float(estimation_error(y[clean_mask], y_pred[clean_mask])),
        "weight_error_l2": float(weight_error(w_all, w1)),
    }
    print("=== Part 2: fit on all data (contaminated) ===")
    print(f"Estimation Error (z==1 only): {part2['estimation_error_clean']:.6f}")
    print(f"Weight Error (L2 norm):      {part2['weight_error_l2']:.6f}\n")

    # --- Part 3: optimizer grid, evaluated on the clean population ----------
    print("=== Part 3: optimizer grid (evaluated on z==1) ===")
    df, _ = run_experiment(
        X, y, w1, learning_rate=0.01, num_epochs=1000, batch_size=32,
        record="eval", eval_X=X, eval_y=y, eval_mask=clean_mask, seed=0)
    print(df.to_string())
    df.to_csv(os.path.join(OUT_DIR, "optimizer_grid.csv"), index=False)
    print()

    # --- Part 4: "ours" (periodic residual removal + re-init) ---------------
    print("=== Part 4: ours (outlier removal) ===")
    w, est_hist, w_hist = ours(
        X, y, w1, init_type="random", learning_rate=0.01, num_epochs=999,
        reset_interval=200, outlier_ratio=0.1,
        eval_X=X, eval_y=y, eval_mask=clean_mask, seed=0)
    summary = pd.DataFrame([{
        "Optimizer": "Outlier Removal",
        "Final MSE": round(est_hist[-1], 6),
        "Final Weight Error": round(w_hist[-1], 6),
    }])
    print(summary.to_string(index=False))

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "part1_oracle_clean": part1,
            "part2_naive_all": part2,
            "part4_ours": {
                "final_mse": float(est_hist[-1]),
                "final_weight_error": float(w_hist[-1]),
            },
        }, f, indent=2)
    print()

    # --- Part 5: learning-rate sweep of "ours" ------------------------------
    print("=== Part 5: learning-rate sweep (weight error) ===")
    learning_rates = [0.01, 0.1, 0.5]
    curves = {}
    for lr in learning_rates:
        _, _, w_hist = ours(
            X, y, w1, init_type="random", learning_rate=lr, num_epochs=1000,
            reset_interval=200, outlier_ratio=0.1,
            eval_X=X, eval_y=y, eval_mask=clean_mask, seed=0)
        curves[f"lr = {lr}"] = w_hist
    plot_curves(curves, xlabel="Epoch", ylabel="Weight Error (L2 norm)",
                title="ours: weight-error convergence by learning rate",
                save_path=os.path.join(OUT_DIR, "lr_sweep.png"))
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
