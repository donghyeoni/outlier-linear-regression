"""Descriptive results on the confirmatory seeds 41-140 (log E7).

Not part of the pre-registered test (``run_confirmatory.py``). These seeds
were used for the E6 choice, so the final method is reported on untouched
seeds by ``run_evaluation.py`` (E8). Computes, on the same 100 datasets:

* the optimizer grid of experiment 2 (4 optimizers x 3 inits x 3 batch
  schemes, learning rate 0.01, 1000 epochs);
* readmit at learning rates 0.01 and 0.5, next to the 0.1 result of the
  confirmatory test;
* from the confirmatory per-seed results: how often readmit beats Oracle
  (with a two-sided sign test and a 95% bootstrap CI of the mean difference,
  10000 resamples, ``default_rng(0)``), and how many clean samples it
  excludes;
* for the runs that hit the 5-cycle cap: the same run with a 20-cycle cap,
  to see whether the inlier set settles or oscillates;
* ``x·w1`` of the outliers readmit keeps vs removes, and the largest number
  of epochs any cycle needed (the per-cycle cap is 20000);
* the sign of the true noise of the clean samples excluded after the first
  cycle and at the end, to check that re-selection removes the one-sided
  bias of the first fit.

Reads ``results/confirmatory/per_seed.csv``, so ``run_confirmatory.py`` must
run first (``run_all.py`` does this). Writes
``results/confirmatory/extra_optimizer_grid.csv`` and
``results/confirmatory/extra_summary.json``.

Usage
-----
    python experiments/run_eval_extra.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from outlier_regression.data import generate_mixture_data
from outlier_regression.diagnostics import (excluded_clean_noise_signs,
                                            outlier_xw1, set_changes,
                                            true_noise)
from outlier_regression.outlier_removal import ours_v2
from outlier_regression.stats import bootstrap_ci, sign_test_p
from outlier_regression.train import run_experiment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "confirmatory")

SEEDS = range(41, 141)
N = 1000
LEARNING_RATES = (0.01, 0.1, 0.5)
CONFIG = {"converge_tol": 1e-5, "stop_k": 3.0, "num_cycles": 5, "seed": 0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grid_rows, lr_rows = [], []
    xw_left, xw_removed, max_cycle_epochs = [], [], 0
    sign_first = {"pos": 0, "neg": 0}
    sign_final = {"pos": 0, "neg": 0}
    for s in SEEDS:
        X, y, z, w1, _ = generate_mixture_data(N=N, D=4, p=0.9, seed=s)
        clean = z == 1
        noise = true_noise(X, y, z, w1)
        df, _ = run_experiment(X, y, w1, learning_rate=0.01, num_epochs=1000,
                               batch_size=32, record="eval", eval_X=X,
                               eval_y=y, eval_mask=clean, seed=0)
        df.insert(0, "seed", s)
        grid_rows.append(df)
        row = {"seed": s}
        for lr in LEARNING_RATES:
            sets = []
            _, _, w_hist, info = ours_v2(
                X, y, w1, learning_rate=lr, prune_rule="readmit", eval_X=X,
                eval_y=y, eval_mask=clean, **CONFIG,
                on_prune=lambda ep, rem, kept: sets.append(kept.copy()))
            row[f"readmit_lr{lr}"] = w_hist[-1]
            if lr == 0.1:
                kept_vals, removed_vals = outlier_xw1(X, w1, z, info["kept_idx"])
                xw_left += kept_vals.tolist()
                xw_removed += removed_vals.tolist()
                max_cycle_epochs = max(max_cycle_epochs,
                                       max(info["cycle_epochs"]))
                # no re-selection event means the first set was kept
                first = sets[0] if sets else info["kept_idx"]
                for counts, kept in ((sign_first, first),
                                     (sign_final, info["kept_idx"])):
                    pos, neg = excluded_clean_noise_signs(noise, z, kept)
                    counts["pos"] += pos
                    counts["neg"] += neg
        lr_rows.append(row)

    grid = pd.concat(grid_rows, ignore_index=True)
    grid.to_csv(os.path.join(OUT_DIR, "extra_optimizer_grid.csv"), index=False)
    lrs = pd.DataFrame(lr_rows)

    agg = (grid.groupby(["Optimizer", "Batch Type", "Init Type"],
                        observed=True)["Weight Error"]
           .agg(mean="mean", std=lambda v: v.std(ddof=0)).reset_index())
    lr_cols = [f"readmit_lr{lr}" for lr in LEARNING_RATES]
    spread = (lrs[lr_cols].max(axis=1) - lrs[lr_cols].min(axis=1)).max()

    # per-seed results of the confirmatory test (run_confirmatory.py first)
    conf_path = os.path.join(OUT_DIR, "per_seed.csv")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(
            f"{conf_path} not found; run experiments/run_confirmatory.py first")
    conf = pd.read_csv(conf_path)
    if list(conf["seed"]) != list(SEEDS):
        raise ValueError(f"{conf_path} does not cover seeds "
                         f"{SEEDS.start}-{SEEDS.stop - 1}; rerun "
                         f"experiments/run_confirmatory.py")
    clean_removed = (N - conf["n_outliers"]) - conf["readmit_clean_kept"]
    vs_oracle = (conf["readmit"] - conf["Oracle"]).to_numpy()
    p_sign = sign_test_p(vs_oracle)
    ci = bootstrap_ci(vs_oracle, 10_000, np.random.default_rng(0))
    capped = []
    for s in conf.loc[~conf["readmit_stopped"], "seed"]:
        X, y, z, w1, _ = generate_mixture_data(N=N, D=4, p=0.9, seed=int(s))
        sets = []
        _, _, w_hist, info = ours_v2(
            X, y, w1, learning_rate=0.1, prune_rule="readmit", eval_X=X,
            eval_y=y, eval_mask=(z == 1),
            **{**CONFIG, "num_cycles": 20},
            on_prune=lambda ep, rem, kept: sets.append(kept.copy()))
        capped.append({
            "seed": int(s),
            "stopped_at_cycle_with_cap_20": info["stopped_at_cycle"],
            "inlier_set_sizes": [len(a) for a in sets],
            "changes_between_cycles": set_changes(sets),
            "weight_error_cap_5": float(conf.loc[conf.seed == s,
                                                 "readmit"].iloc[0]),
            "weight_error_cap_20": float(w_hist[-1]),
        })
    summary = {
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "capped_runs_with_20_cycles": capped,
        "max_epochs_in_one_cycle_lr0.1": int(max_cycle_epochs),
        "excluded_clean_noise_sign_after_first_cycle": sign_first,
        "excluded_clean_noise_sign_final": sign_final,
        "outlier_xw1_left": {"n": len(xw_left),
                             "mean": float(np.mean(xw_left)) if xw_left else None,
                             "max": float(np.max(xw_left)) if xw_left else None},
        "outlier_xw1_removed": {"n": len(xw_removed),
                                "mean": float(np.mean(xw_removed)),
                                "min": float(np.min(xw_removed)),
                                "fraction_below_max_left": (float(np.mean(
                                    np.array(xw_removed) < np.max(xw_left)))
                                    if xw_left else None)},
        "ours_better_than_oracle": int(np.sum(conf["readmit"] < conf["Oracle"])),
        "ours_clean_removed_mean": float(clean_removed.mean()),
        "ours_minus_oracle": {
            "mean": float(vs_oracle.mean()),
            "ci95_bootstrap": [float(ci[0]), float(ci[1])],
            "sign_test_p": float(p_sign),
        },
        "optimizer_grid_weight_error": [
            {"optimizer": r.Optimizer, "batch": r["Batch Type"],
             "init": r["Init Type"], "mean": float(r["mean"]),
             "std": float(r["std"])} for _, r in agg.iterrows()],
        "readmit_by_lr": {
            c: {"mean": float(lrs[c].mean()), "std": float(lrs[c].std(ddof=0))}
            for c in lr_cols},
        "readmit_max_lr_spread": float(spread),
    }
    with open(os.path.join(OUT_DIR, "extra_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(agg.to_string(index=False))
    for c, v in summary["readmit_by_lr"].items():
        print(f"{c}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"max spread across learning rates: {spread:.2e}")
    print(f"\nArtifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
