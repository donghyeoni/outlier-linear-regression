"""Reproduces notebook 1: benchmark of four hand-coded optimizers on clean
synthetic linear-regression data, compared to the closed-form solution.

Usage
-----
    python experiments/run_baseline.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_linear_data
from src.regression import closed_form_solution, estimation_error, weight_error
from src.train import run_experiment
from src.plots import plot_optimizer_convergence

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- data ---------------------------------------------------------------
    X, y, w_true = generate_linear_data(N=1000, D=4, seed=0)

    # --- closed-form reference (normal equation, explicit inverse) ----------
    w_pred = closed_form_solution(X, y, method="inv")
    y_pred = X @ w_pred
    print("True Weights:      ", w_true)
    print("Estimated Weights: ", w_pred)
    print("Weight Error (L2): ", weight_error(w_pred, w_true))
    print("Estimation Error:  ", estimation_error(y, y_pred))
    print()

    # --- optimizer grid -----------------------------------------------------
    df, histories = run_experiment(
        X, y, w_true, learning_rate=0.1, num_epochs=1000, batch_size=32,
        record="batch", seed=0)
    print(df.to_string())

    # --- convergence plots (zero init, full batch) --------------------------
    plot_optimizer_convergence(
        histories, "estimation", batch="full", init="zero",
        title="Estimation Error (init=zero, batch=full)",
        save_path=os.path.join(OUT_DIR, "baseline_estimation_error.png"))
    plot_optimizer_convergence(
        histories, "weight", batch="full", init="zero",
        title="Weight Error (init=zero, batch=full)",
        save_path=os.path.join(OUT_DIR, "baseline_weight_error.png"))
    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
