"""Compare ratio-based and threshold-based pruning in ``ours_v2``.

Both use per-cycle convergence (``converge_tol=1e-5``) and the stopping rule.
``ratio`` removes the top 10% of absolute residuals per cycle; ``threshold``
removes exactly the samples with ``|r| > k * sigma_MAD``. Evaluated on the
development dataset (seed 0) and the held-out datasets (seeds 1-20). For
every run it also records the sign of the true noise of the clean samples
that were removed, which shows whether pruning is one-sided.

Writes ``results/ablation/step3_threshold.csv`` and
``results/ablation/step3_threshold_summary.json``.

Usage
-----
    python experiments/run_threshold.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import ours_v2
from outlier_regression.regression import closed_form_solution, weight_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "ablation")

SEEDS = range(0, 21)
RULES = ("ratio", "threshold")
KS = (2.5, 3.0, 4.0)
LEARNING_RATES = (0.01, 0.1, 0.5)


def evaluate(seed):
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    clean = z == 1
    noise = y - np.where(clean, 1.0, -1.0) * (X @ w1)
    rows = []
    oracle = weight_error(closed_form_solution(X[clean], y[clean]), w1)
    for rule in RULES:
        for k in KS:
            for lr in LEARNING_RATES:
                removed = []
                _, _, w_hist, info = ours_v2(
                    X, y, w1, learning_rate=lr, converge_tol=1e-5, stop_k=k,
                    prune_rule=rule, eval_X=X, eval_y=y, eval_mask=clean,
                    seed=0,
                    on_prune=lambda ep, rem, kept: removed.extend(rem.tolist()))
                removed = np.array(removed, dtype=int)
                removed_clean = removed[z[removed] == 1]
                kept = info["kept_idx"]
                rows.append({
                    "seed": seed, "rule": rule, "k": k, "learning_rate": lr,
                    "weight_error": w_hist[-1], "oracle": oracle,
                    "cycles": len(info["cycle_epochs"]),
                    "clean_total": int(np.sum(clean)),
                    "clean_kept": int(np.sum(z[kept] == 1)),
                    "outliers_left": int(np.sum(z[kept] == 2)),
                    "removed_clean_pos_noise": int(np.sum(noise[removed_clean] > 0)),
                    "removed_clean_neg_noise": int(np.sum(noise[removed_clean] < 0)),
                    "kept_clean_mean_noise": float(
                        noise[kept[z[kept] == 1]].mean()),
                })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame([r for s in SEEDS for r in evaluate(s)])
    df.to_csv(os.path.join(OUT_DIR, "step3_threshold.csv"), index=False)

    held = df[df.seed > 0]
    summary = {"dev_seed": 0, "held_out_seeds": [1, 20], "configs": []}
    print("held-out seeds 1-20 (lr 0.1):")
    for rule in RULES:
        for k in KS:
            h = held[(held.rule == rule) & (held.k == k)
                     & (held.learning_rate == 0.1)]
            d = df[(df.seed == 0) & (df.rule == rule) & (df.k == k)
                   & (df.learning_rate == 0.1)].iloc[0]
            pos = int(h.removed_clean_pos_noise.sum())
            neg = int(h.removed_clean_neg_noise.sum())
            entry = {
                "rule": rule, "k": k,
                "dev_weight_error": float(d.weight_error),
                "held_out_mean": float(h.weight_error.mean()),
                "held_out_std": float(h.weight_error.std(ddof=0)),
                "clean_kept_mean": float(h.clean_kept.mean()),
                "outliers_left_total": int(h.outliers_left.sum()),
                "removed_clean_pos_noise_total": pos,
                "removed_clean_neg_noise_total": neg,
            }
            summary["configs"].append(entry)
            print(f"  {rule:9s} k={k}: {entry['held_out_mean']:.4f} "
                  f"± {entry['held_out_std']:.4f}  clean kept "
                  f"{entry['clean_kept_mean']:.1f}  outliers left "
                  f"{entry['outliers_left_total']}  removed clean +/-: "
                  f"{pos}/{neg}  (dev {entry['dev_weight_error']:.4f})")

    a = held[(held.rule == "threshold") & (held.k == 3.0)
             & (held.learning_rate == 0.1)].set_index("seed").weight_error
    b = held[(held.rule == "ratio") & (held.k == 3.0)
             & (held.learning_rate == 0.1)].set_index("seed").weight_error
    summary["threshold_k3_beats_ratio_k3"] = int(np.sum(a < b))
    print(f"threshold (k=3) < ratio (k=3) on {int(np.sum(a < b))}/20 datasets")

    with open(os.path.join(OUT_DIR, "step3_threshold_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
