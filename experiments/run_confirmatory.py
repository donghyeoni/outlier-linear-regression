"""Pre-registered confirmatory test: readmit vs ratio pruning.

Follows the plan recorded in ``docs/experiment-log.md`` ("Pre-registered
test: readmit vs ratio") before this script was run:

* 100 fresh datasets, seeds 41-140
* primary metric: final weight error at learning rate 0.1
* primary test: paired two-sided sign-flip permutation test (100000 flips)
* secondary: Wilcoxon signed-rank test (normal approximation)
* alpha = 0.05; mean difference reported with a 95% bootstrap CI

Writes ``results/confirmatory/per_seed.csv``,
``results/confirmatory/summary.json`` and
``results/confirmatory/weight_error_by_seed.png`` (the README figure).

Usage
-----
    python experiments/run_confirmatory.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v2
from outlier_regression.plots import plot_weight_error_strip
from outlier_regression.regression import closed_form_solution, weight_error
from outlier_regression.stats import (bootstrap_ci, sign_flip_p,
                                     wilcoxon_signed_rank)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "confirmatory")

SEEDS = range(41, 141)
CONFIG = {"learning_rate": 0.1, "converge_tol": 1e-5, "stop_k": 3.0,
          "num_cycles": 5, "seed": 0}
ALPHA = 0.05
N_FLIPS = 100_000
N_BOOT = 10_000

# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
COLORS = {"Oracle": "#1baf7a", "Naive": "#eb6834", "readmit": "#2a78d6"}


def evaluate(seed):
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    clean = z == 1
    row = {
        "seed": seed,
        "n_outliers": int(np.sum(z == 2)),
        "Oracle": weight_error(closed_form_solution(X[clean], y[clean]), w1),
        "Naive": weight_error(closed_form_solution(X, y), w1),
    }
    for rule in ("ratio", "readmit"):
        _, _, w_hist, info = ours_v2(X, y, w1, prune_rule=rule, eval_X=X,
                                     eval_y=y, eval_mask=clean, **CONFIG)
        kept = info["kept_idx"]
        row[rule] = w_hist[-1]
        row[f"{rule}_clean_kept"] = int(np.sum(z[kept] == 1))
        row[f"{rule}_outliers_left"] = int(np.sum(z[kept] == 2))
        row[f"{rule}_stopped"] = info["stopped_at_cycle"] is not None
    return row


def plot(df, path):
    plot_weight_error_strip(
        [("Oracle", df["Oracle"], COLORS["Oracle"]),
         ("Naive", df["Naive"], COLORS["Naive"]),
         ("readmit", df["readmit"], COLORS["readmit"])],
        save_path=path,
        title=f"Weight error on {len(df)} confirmatory datasets",
        point_size=16, jitter=0.15, edge_width=0.6, alpha=0.9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame([evaluate(s) for s in SEEDS])
    df.to_csv(os.path.join(OUT_DIR, "per_seed.csv"), index=False)

    diff = (df["readmit"] - df["ratio"]).to_numpy()
    p_perm = sign_flip_p(diff, N_FLIPS, np.random.default_rng(0))
    w_plus, z_w, p_wil = wilcoxon_signed_rank(diff)
    ci = bootstrap_ci(diff, N_BOOT, np.random.default_rng(0))
    adopt = bool(p_perm < ALPHA and diff.mean() < 0)

    summary = {
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "config": CONFIG,
        "n": int(len(diff)),
        "mean_diff_readmit_minus_ratio": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "ci95_bootstrap": [float(ci[0]), float(ci[1])],
        "readmit_better": int(np.sum(diff < 0)),
        "p_sign_flip": p_perm,
        "wilcoxon": {"w_plus": w_plus, "z": z_w, "p": p_wil},
        "decision_adopt_readmit": adopt,
        "methods": {
            m: {"weight_error_mean": float(df[m].mean()),
                "weight_error_std": float(df[m].std(ddof=0))}
            for m in ("Oracle", "Naive", "ratio", "readmit")
        },
        "ratio_clean_kept_mean": float(df["ratio_clean_kept"].mean()),
        "readmit_clean_kept_mean": float(df["readmit_clean_kept"].mean()),
        "ratio_outliers_left_total": int(df["ratio_outliers_left"].sum()),
        "readmit_outliers_left_total": int(df["readmit_outliers_left"].sum()),
        "readmit_not_stopped": int((~df["readmit_stopped"]).sum()),
        "readmit_minus_oracle_mean": float((df["readmit"] - df["Oracle"]).mean()),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot(df, os.path.join(OUT_DIR, "weight_error_by_seed.png"))

    for m, v in summary["methods"].items():
        print(f"{m:8s} {v['weight_error_mean']:.4f} ± {v['weight_error_std']:.4f}")
    print(f"readmit - ratio: mean {diff.mean():+.5f}, "
          f"95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}], "
          f"readmit better on {summary['readmit_better']}/{len(diff)}")
    print(f"sign-flip p = {p_perm:.5f}; Wilcoxon z = {z_w:.2f}, p = {p_wil:.5f}")
    print(f"decision: {'adopt readmit' if adopt else 'keep ratio'}")
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
