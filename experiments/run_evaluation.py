"""Final evaluation of the chosen method on untouched datasets (log E8).

Seeds 141-240 were not used for any decision. The final method is
:func:`outlier_regression.outlier_removal.ours` with its default settings
(``ours_v2`` with ``prune_rule="readmit", stop_k=3, converge_tol=1e-5,
num_cycles=5, learning_rate=0.1``) and ``seed=0``, exactly as adopted in E6.
The plan (what is measured and how) was recorded in
``docs/experiment-log.md`` before this script was run.

Writes to ``results/evaluation/``:

* ``per_seed.csv``          -- per-dataset values of every method/diagnostic
* ``optimizer_grid.csv``    -- the 36-configuration grid, per dataset
* ``summary.json``          -- everything reported in the README
* ``weight_error_by_seed.png`` -- README figure 5

Usage
-----
    python experiments/run_evaluation.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from outlier_regression.data import generate_mixture_data
from outlier_regression.outlier_removal import (FINAL_CONFIG, ours,
                                              robust_scale)
from outlier_regression.plots import plot_weight_error_strip
from outlier_regression.regression import closed_form_solution, weight_error
from outlier_regression.stats import bootstrap_ci, sign_flip_p
from outlier_regression.train import run_experiment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "evaluation")

SEEDS = range(141, 241)
LEARNING_RATES = (0.01, 0.1, 0.5)
FINAL = {"prune_rule": "readmit", "stop_k": 3.0, "converge_tol": 1e-5,
         "num_cycles": 5, "seed": 0}
# the recorded configuration must be the defaults of ours()
assert {k: v for k, v in FINAL.items() if k != "seed"} == {
    k: v for k, v in FINAL_CONFIG.items() if k != "learning_rate"}
assert FINAL_CONFIG["learning_rate"] == 0.1
N_FLIPS = 100_000
N_BOOT = 10_000
NOISE_STD = 0.1

# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
COLORS = {"Oracle": "#1baf7a", "Naive": "#eb6834", "ours": "#2a78d6"}


def run_ours(X, y, z, w1, lr, num_cycles=5):
    """Run the final method and collect the per-cycle inlier sets."""
    sets = []
    w, _, w_hist, info = ours(
        X, y, w1, learning_rate=lr, num_cycles=num_cycles, seed=0,
        eval_X=X, eval_y=y, eval_mask=(z == 1),
        on_prune=lambda ep, rem, kept: sets.append(kept.copy()))
    return w, w_hist, info, sets


def evaluate(seed):
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    clean = z == 1
    noise = y - np.where(clean, 1.0, -1.0) * (X @ w1)
    row = {
        "seed": seed,
        "n_outliers": int(np.sum(~clean)),
        "n_clean": int(np.sum(clean)),
        "Oracle": weight_error(closed_form_solution(X[clean], y[clean]), w1),
        "Naive": weight_error(closed_form_solution(X, y), w1),
    }
    for lr in LEARNING_RATES:
        w, w_hist, info, sets = run_ours(X, y, z, w1, lr)
        row[f"ours_lr{lr}"] = w_hist[-1]
        if lr != 0.1:
            continue
        kept = info["kept_idx"]
        kept_clean = kept[clean[kept]]
        kept_out = kept[~clean[kept]]
        removed_out = np.setdiff1d(np.flatnonzero(~clean), kept)
        first = sets[0] if sets else kept
        excl_first = np.setdiff1d(np.flatnonzero(clean), first)
        excl_final = np.setdiff1d(np.flatnonzero(clean), kept)
        row.update({
            "ours": w_hist[-1],
            "ours_clean_kept": int(len(kept_clean)),
            "ours_outliers_left": int(len(kept_out)),
            "ours_stopped": info["stopped_at_cycle"] is not None,
            "ours_cycles": len(info["cycle_epochs"]),
            "ours_max_cycle_epochs": int(max(info["cycle_epochs"])),
            "ours_sigma_mad_final": robust_scale(X[kept] @ w - y[kept]),
            # effect of the kept outliers: refit the final inlier set
            # with and without them (closed form)
            "refit_with_kept_outliers": weight_error(
                closed_form_solution(X[kept], y[kept]), w1),
            "refit_without_kept_outliers": weight_error(
                closed_form_solution(X[kept_clean], y[kept_clean]), w1),
            "excl_first_pos": int(np.sum(noise[excl_first] > 0)),
            "excl_first_neg": int(np.sum(noise[excl_first] < 0)),
            "excl_final_pos": int(np.sum(noise[excl_final] > 0)),
            "excl_final_neg": int(np.sum(noise[excl_final] < 0)),
            "xw1_kept_outliers": ";".join(f"{v:.6f}" for v in X[kept_out] @ w1),
            "xw1_removed_outliers_sum": float(np.sum(X[removed_out] @ w1)),
            "xw1_removed_outliers_n": int(len(removed_out)),
            "xw1_removed_outliers_min": float(np.min(X[removed_out] @ w1)),
        })
    grid, _ = run_experiment(X, y, w1, learning_rate=0.01, num_epochs=1000,
                             batch_size=32, record="eval", eval_X=X,
                             eval_y=y, eval_mask=clean, seed=0)
    grid.insert(0, "seed", seed)
    return row, grid, (X, y, z, w1)


def stats_of(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, grids, data = [], [], {}
    for s in SEEDS:
        row, grid, d = evaluate(s)
        rows.append(row)
        grids.append(grid)
        data[s] = d
    df = pd.DataFrame(rows)
    grid = pd.concat(grids, ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "per_seed.csv"), index=False)
    grid.to_csv(os.path.join(OUT_DIR, "optimizer_grid.csv"), index=False)

    # optimizer grid per configuration
    g = grid.groupby(["Optimizer", "Batch Type", "Init Type"], observed=True)
    grid_summary = [{"optimizer": k[0], "batch": k[1], "init": k[2],
                     **stats_of(v["Weight Error"])} for k, v in g]
    adam = grid[(grid.Optimizer == "Adam") & (grid["Batch Type"] == "full")
                & (grid["Init Type"] == "zero")].set_index("seed")["Weight Error"]
    adam_vs_naive = float((adam - df.set_index("seed")["Naive"]).abs().max())

    # ours vs Oracle / Naive
    diff = (df["ours"] - df["Oracle"]).to_numpy()
    ci = bootstrap_ci(diff, N_BOOT, np.random.default_rng(0))
    lr_cols = [f"ours_lr{lr}" for lr in LEARNING_RATES]

    # kept vs removed outliers
    kept_xw = np.array([float(v) for s in df["xw1_kept_outliers"].dropna()
                        for v in str(s).split(";") if v != ""])
    removed_mean = float(df["xw1_removed_outliers_sum"].sum()
                         / df["xw1_removed_outliers_n"].sum())

    # runs that hit the 5-cycle cap, re-run with a 20-cycle cap
    capped = []
    for s in df.loc[~df["ours_stopped"], "seed"]:
        X, y, z, w1 = data[int(s)]
        _, w_hist, info, sets = run_ours(X, y, z, w1, 0.1, num_cycles=20)
        capped.append({
            "seed": int(s),
            "stopped_at_cycle_with_cap_20": info["stopped_at_cycle"],
            "inlier_set_sizes": [len(a) for a in sets],
            "changes_between_cycles": [
                int(len(np.setxor1d(sets[i], sets[i - 1])))
                for i in range(1, len(sets))],
            "weight_error_cap_5": float(df.loc[df.seed == s, "ours"].iloc[0]),
            "weight_error_cap_20": float(w_hist[-1]),
        })

    summary = {
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "final_config": {**FINAL, "learning_rate": 0.1},
        "n_datasets": int(len(df)),
        "outliers_per_dataset": [int(df.n_outliers.min()), int(df.n_outliers.max())],
        "clean_per_dataset_mean": float(df.n_clean.mean()),
        "weight_error": {m: stats_of(df[m]) for m in ("Oracle", "Naive", "ours")},
        "ours_by_lr": {c: stats_of(df[c]) for c in lr_cols},
        "ours_max_lr_spread": float(
            (df[lr_cols].max(axis=1) - df[lr_cols].min(axis=1)).max()),
        "ours_better_than_naive": int(np.sum(df["ours"] < df["Naive"])),
        "ours_minus_oracle": {
            "mean": float(diff.mean()),
            "ci95_bootstrap": list(ci),
            "sign_flip_p": sign_flip_p(diff, N_FLIPS, np.random.default_rng(0)),
            "ours_better": int(np.sum(diff < 0)),
        },
        "optimizer_grid": grid_summary,
        "optimizer_grid_min_mean": float(min(r["mean"] for r in grid_summary)),
        "adam_full_zero_max_abs_diff_to_naive": adam_vs_naive,
        "ours_clean_kept_mean": float(df["ours_clean_kept"].mean()),
        "ours_clean_excluded_mean": float((df.n_clean - df.ours_clean_kept).mean()),
        "ours_outliers_left_total": int(df["ours_outliers_left"].sum()),
        "ours_datasets_with_outliers_left": int(np.sum(df["ours_outliers_left"] > 0)),
        "outlier_xw1_kept": {"n": int(len(kept_xw)),
                             "mean": float(kept_xw.mean()) if len(kept_xw) else None,
                             "max": float(kept_xw.max()) if len(kept_xw) else None},
        "outlier_xw1_removed": {"n": int(df["xw1_removed_outliers_n"].sum()),
                                "mean": removed_mean,
                                "min": float(df["xw1_removed_outliers_min"].min())},
        "refit_effect_of_kept_outliers": {
            "max_abs_change": float((df["refit_with_kept_outliers"]
                                     - df["refit_without_kept_outliers"]).abs().max()),
            "mean_change": float((df["refit_with_kept_outliers"]
                                  - df["refit_without_kept_outliers"]).mean()),
        },
        "sigma_mad_final": {**stats_of(df["ours_sigma_mad_final"]),
                            "true_noise_std": NOISE_STD},
        "excluded_clean_noise_sign": {
            "after_first_cycle": {"pos": int(df.excl_first_pos.sum()),
                                  "neg": int(df.excl_first_neg.sum())},
            "final": {"pos": int(df.excl_final_pos.sum()),
                      "neg": int(df.excl_final_neg.sum())},
        },
        "ours_not_stopped_within_5_cycles": int(np.sum(~df["ours_stopped"])),
        "capped_runs_with_20_cycles": capped,
        "ours_max_epochs_in_one_cycle": int(df["ours_max_cycle_epochs"].max()),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_weight_error_strip(
        [("Oracle", df["Oracle"], COLORS["Oracle"]),
         ("Naive", df["Naive"], COLORS["Naive"]),
         ("ours", df["ours"], COLORS["ours"])],
        save_path=os.path.join(OUT_DIR, "weight_error_by_seed.png"),
        title=f"Weight error on {len(df)} evaluation datasets",
        point_size=16, jitter=0.15, edge_width=0.6, alpha=0.9)

    w = summary["weight_error"]
    for m in ("Oracle", "Naive", "ours"):
        print(f"{m:7s} {w[m]['mean']:.4f} ± {w[m]['std']:.4f} "
              f"({w[m]['min']:.4f} - {w[m]['max']:.4f})")
    om = summary["ours_minus_oracle"]
    print(f"ours - Oracle: {om['mean']:+.5f}, CI [{ci[0]:+.5f}, {ci[1]:+.5f}], "
          f"sign-flip p {om['sign_flip_p']:.3f}, ours better {om['ours_better']}")
    print(f"ours < Naive on {summary['ours_better_than_naive']}/{len(df)}")
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
