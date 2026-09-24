"""Diagnose the pruning behaviour of "ours".

For each pruning event, records how many removed samples were true outliers
(``z == 2``), how many outliers remain, and the weight error of the model that
made the pruning decision. It also fits the closed form on the final kept set,
which separates the error caused by *which* samples were kept from the error
caused by training not converging. Runs several learning rates and lengths.

Writes ``results/outlier/pruning_diagnostic.json``.

Usage
-----
    python experiments/diagnose_pruning.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v1
from outlier_regression.regression import closed_form_solution, weight_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "outlier")


def diagnose(X, y, z, w1, learning_rate, num_epochs):
    events = []
    final = {"kept_idx": np.arange(X.shape[0])}

    def on_prune(epoch, removed_idx, kept_idx):
        events.append({
            "epoch": epoch,
            "removed": int(len(removed_idx)),
            "removed_outliers": int(np.sum(z[removed_idx] == 2)),
            "kept": int(len(kept_idx)),
            "kept_outliers": int(np.sum(z[kept_idx] == 2)),
        })
        final["kept_idx"] = kept_idx

    w, _, w_hist = ours_v1(X, y, w1, init_type="random",
                           learning_rate=learning_rate,
                           num_epochs=num_epochs, reset_interval=200,
                           outlier_ratio=0.1, eval_X=X, eval_y=y,
                           eval_mask=(z == 1), seed=0, on_prune=on_prune)
    # weight error of the model that made each pruning decision
    for e in events:
        e["weight_error_before_prune"] = w_hist[e["epoch"] - 1]

    kept = final["kept_idx"]
    w_cf = closed_form_solution(X[kept], y[kept], method="pinv")
    return {
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "prune_events": events,
        "final_weight_error_trained": weight_error(w, w1),
        "final_weight_error_closed_form_on_kept": weight_error(w_cf, w1),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=0)
    print(f"true outliers: {int(np.sum(z == 2))} / {len(z)}")

    configs = [(0.01, 1000), (0.01, 3000),
               (0.1, 1000), (0.5, 1000)]
    runs = [diagnose(X, y, z, w1, lr, n) for lr, n in configs]
    for r in runs:
        print(f"\n--- lr = {r['learning_rate']}, "
              f"num_epochs = {r['num_epochs']} ---")
        for e in r["prune_events"]:
            print(f"  prune@{e['epoch']:>4}: "
                  f"w_err {e['weight_error_before_prune']:.4f}, "
                  f"removed {e['removed']:>3} "
                  f"(outliers {e['removed_outliers']:>2}), kept {e['kept']:>3} "
                  f"(outliers {e['kept_outliers']:>2})")
        print(f"  weight error, trained:              "
              f"{r['final_weight_error_trained']:.4f}")
        print(f"  weight error, closed form on kept:  "
              f"{r['final_weight_error_closed_form_on_kept']:.4f}")

    with open(os.path.join(OUT_DIR, "pruning_diagnostic.json"), "w") as f:
        json.dump({"true_outliers": int(np.sum(z == 2)), "runs": runs},
                  f, indent=2)
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
