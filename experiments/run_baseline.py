"""Reproduces notebook 1: benchmark of four hand-coded optimizers on clean
synthetic linear-regression data, compared to the closed-form solution.

Running this writes reproducible artifacts to ``results/baseline/``:

* ``closed_form.json``   -- ground-truth vs. estimated weights and errors
* ``optimizer_grid.csv`` -- full (optimizer x batch x init) error grid
* ``estimation_error.png`` / ``weight_error.png`` -- convergence plots

Usage
-----
    python experiments/run_baseline.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_linear_data
from src.regression import closed_form_solution, estimation_error, weight_error
from src.train import run_experiment
from src.plots import plot_optimizer_convergence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "baseline")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- data ---------------------------------------------------------------
    X, y, w_true = generate_linear_data(N=1000, D=4, seed=0)

    # --- closed-form reference (normal equation, explicit inverse) ----------
    w_pred = closed_form_solution(X, y, method="inv")
    y_pred = X @ w_pred
    closed_form = {
        "true_weights": w_true.tolist(),
        "estimated_weights": w_pred.tolist(),
        "weight_error_l2": float(weight_error(w_pred, w_true)),
        "estimation_error": float(estimation_error(y, y_pred)),
    }
    print("True Weights:      ", w_true)
    print("Estimated Weights: ", w_pred)
    print("Weight Error (L2): ", closed_form["weight_error_l2"])
    print("Estimation Error:  ", closed_form["estimation_error"])
    with open(os.path.join(OUT_DIR, "closed_form.json"), "w") as f:
        json.dump(closed_form, f, indent=2)
    print()

    # --- optimizer grid -----------------------------------------------------
    df, histories = run_experiment(
        X, y, w_true, learning_rate=0.1, num_epochs=1000, batch_size=32,
        record="batch", seed=0)
    print(df.to_string())
    df.to_csv(os.path.join(OUT_DIR, "optimizer_grid.csv"), index=False)

    # --- convergence plots (zero init, full batch) --------------------------
    plot_optimizer_convergence(
        histories, "estimation", batch="full", init="zero",
        title="Estimation Error (init=zero, batch=full)",
        save_path=os.path.join(OUT_DIR, "estimation_error.png"))
    plot_optimizer_convergence(
        histories, "weight", batch="full", init="zero",
        title="Weight Error (init=zero, batch=full)",
        save_path=os.path.join(OUT_DIR, "weight_error.png"))
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
