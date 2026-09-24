"""Evaluate the ratio-based method on held-out datasets (seeds 1-20).

Experiment-log entry E3. Ratio pruning was later replaced by re-admission
after the confirmatory test (E6, ``run_confirmatory.py``).

The ratio-based method is ``ours_v2`` with per-cycle convergence
(``converge_tol=1e-5``) and the stopping rule (``stop_k=3``). Both values were
fixed before this evaluation. Seed 0 was used during development, so the
evaluation uses fresh mixture datasets with seeds 1-20. The seed-0 result is
also saved, separately, for reference.

Writes to ``results/final/``:

* ``per_seed.csv``          -- weight error per dataset seed and method
                               (``v1_*`` = ``ours_v1``)
* ``summary.json``          -- mean / std / min / max per method
* ``weight_error_by_seed.png``

Usage
-----
    python experiments/run_final.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v1, ours_v2
from outlier_regression.plots import plot_weight_error_strip
from outlier_regression.regression import closed_form_solution, weight_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "final")

SEEDS = range(1, 21)
LEARNING_RATES = (0.01, 0.1, 0.5)
FINAL = {"converge_tol": 1e-5, "stop_k": 3.0}

# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
COLORS = {"Oracle": "#1baf7a", "Naive": "#eb6834", "ours": "#2a78d6"}


def evaluate(seed):
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    clean = z == 1
    row = {
        "seed": seed,
        "n_outliers": int(np.sum(z == 2)),
        "Oracle": weight_error(closed_form_solution(X[clean], y[clean]), w1),
        "Naive": weight_error(closed_form_solution(X, y), w1),
    }
    for lr in LEARNING_RATES:
        w, est, w_hist, info = ours_v2(
            X, y, w1, learning_rate=lr, eval_X=X, eval_y=y, eval_mask=clean,
            seed=0, **FINAL)
        kept = info["kept_idx"]
        row[f"ours_lr{lr}"] = w_hist[-1]
        row[f"ours_lr{lr}_mse"] = est[-1]
        row[f"ours_lr{lr}_kept"] = int(len(kept))
        row[f"ours_lr{lr}_outliers_left"] = int(np.sum(z[kept] == 2))
        row[f"ours_lr{lr}_epochs"] = len(w_hist)
    # ours_v1, for comparison in the experiment log
    for lr in LEARNING_RATES:
        _, est, w_hist = ours_v1(X, y, w1, learning_rate=lr,
                                 num_epochs=1000, eval_X=X, eval_y=y,
                                 eval_mask=clean, seed=0)
        row[f"v1_lr{lr}"] = w_hist[-1]
        row[f"v1_lr{lr}_mse"] = est[-1]
    row["Oracle_mse"] = float(np.mean(
        (y[clean] - X[clean] @ closed_form_solution(X[clean], y[clean])) ** 2))
    row["Naive_mse"] = float(np.mean(
        (y[clean] - X[clean] @ closed_form_solution(X, y)) ** 2))
    return row


def plot(df, path):
    plot_weight_error_strip(
        [("Oracle", df["Oracle"], COLORS["Oracle"]),
         ("Naive", df["Naive"], COLORS["Naive"]),
         ("ours (final)", df["ours_lr0.1"], COLORS["ours"])],
        save_path=path,
        title=f"Weight error on {len(df)} held-out datasets",
        point_size=28, jitter=0.12, edge_width=1.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame([evaluate(s) for s in SEEDS])
    df.to_csv(os.path.join(OUT_DIR, "per_seed.csv"), index=False)

    cols = (["Oracle", "Naive"] + [f"ours_lr{lr}" for lr in LEARNING_RATES]
            + [f"v1_lr{lr}" for lr in LEARNING_RATES])
    summary = {"seeds": list(SEEDS), "final_config": FINAL, "methods": {}}
    for c in cols:
        v = df[c].to_numpy()
        summary["methods"][c] = {
            "weight_error_mean": float(v.mean()),
            "weight_error_std": float(v.std()),
            "weight_error_min": float(v.min()),
            "weight_error_max": float(v.max()),
            "mse_mean": float(df[f"{c}_mse"].mean()),
        }
        print(f"{c:16s} weight error {v.mean():.4f} ± {v.std():.4f} "
              f"(min {v.min():.4f}, max {v.max():.4f}), "
              f"MSE {df[f'{c}_mse'].mean():.5f}")
    for lr in LEARNING_RATES:
        summary["methods"][f"ours_lr{lr}"].update({
            "kept_mean": float(df[f"ours_lr{lr}_kept"].mean()),
            "outliers_left_total": int(df[f"ours_lr{lr}_outliers_left"].sum()),
            "epochs_mean": float(df[f"ours_lr{lr}_epochs"].mean()),
        })
    gap = df["ours_lr0.1"] - df["Oracle"]
    summary["ours_minus_oracle_mean"] = float(gap.mean())
    summary["ours_better_than_naive"] = int(np.sum(df["ours_lr0.1"] < df["Naive"]))
    print(f"ours - Oracle: {gap.mean():.4f} on average; "
          f"ours < Naive on {summary['ours_better_than_naive']}/{len(df)} datasets")

    dev = evaluate(0)
    summary["development_seed0"] = {
        k: dev[k] for k in ["n_outliers", "Oracle", "Oracle_mse", "Naive",
                            "Naive_mse"]
        + [f"ours_lr{lr}{sfx}" for lr in LEARNING_RATES
           for sfx in ("", "_mse", "_kept", "_outliers_left")]}
    print(f"development seed 0: ours {dev['ours_lr0.1']:.4f} "
          f"(Oracle {dev['Oracle']:.4f}, Naive {dev['Naive']:.4f})")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot(df, os.path.join(OUT_DIR, "weight_error_by_seed.png"))
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
