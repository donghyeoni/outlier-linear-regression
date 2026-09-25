"""E1: optimizer grid on clean data against the closed form.

Writes ``results/exp1/``: ``per_run.csv`` (one row per dataset and grid
combination), ``closed_form.csv``, ``summary.csv``, ``tables.md`` and
``curves.png`` (mean weight error per iteration, full batch, zero init).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import (EXP1_SEEDS, GRID_ITERS, GRID_LRS, f4, md_table, mean_sd,
                    out_dir, write_tables)
from outlier_regression.data import generate_linear_data
from outlier_regression.optimizers import OPTIMIZERS
from outlier_regression.plots import plot_curves
from outlier_regression.regression import closed_form_solution, weight_error
from outlier_regression.train import run_grid

KEYS = ["optimizer", "batch", "init", "lr"]


def main():
    out = out_dir("exp1")
    runs, cf_rows, curves = [], [], {}
    for seed in EXP1_SEEDS:
        X, y, w_true = generate_linear_data(N=1000, D=4, seed=seed)
        w_cf = closed_form_solution(X, y)
        cf_rows.append({"seed": seed,
                        "weight_error": weight_error(w_cf, w_true)})
        df, hist = run_grid(X, y, w_true, w_cf, learning_rates=GRID_LRS,
                            num_iters=GRID_ITERS, keep_histories=True)
        df.insert(0, "seed", seed)
        runs.append(df)
        for (opt, batch, init, lr), h in hist.items():
            if batch == "full" and init == "zero":
                curves.setdefault((opt, lr), []).append(h)
        print(f"seed {seed} done")

    per_run = pd.concat(runs, ignore_index=True)
    per_run.to_csv(os.path.join(out, "per_run.csv"), index=False)
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(os.path.join(out, "closed_form.csv"), index=False)

    summary = (per_run.groupby(KEYS, sort=False)
               .agg(we_mean=("weight_error", "mean"),
                    we_sd=("weight_error", lambda v: v.std(ddof=0)),
                    dist_max=("dist_to_star", "max"))
               .reset_index())
    summary["reaches_cf"] = summary["dist_max"] < 5e-5
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)

    t_cf = md_table(["method", "weight error (mean ± SD)"],
                    [["closed form", mean_sd(cf["weight_error"])]])
    t_count = md_table(
        ["optimizer"] + [f"lr {lr}" for lr in GRID_LRS],
        [[opt] + [f"{int(summary[(summary.optimizer == opt) & (summary.lr == lr)].reaches_cf.sum())} / 9"
                  for lr in GRID_LRS] for opt in OPTIMIZERS])
    t_full = md_table(
        ["optimizer", "batch", "init", "lr", "weight error (mean ± SD)",
         "max ‖ŵ − ŵ_cf‖", "reaches closed form"],
        [[r.optimizer, r.batch, r.init, r.lr,
          f"{f4(r.we_mean)} ± {f4(r.we_sd)}", f4(r.dist_max),
          "yes" if r.reaches_cf else "no"] for r in summary.itertuples()])
    fz = summary[(summary.batch == "full") & (summary.init == "zero")]
    t_fz = md_table(
        ["optimizer"] + [f"lr {lr}" for lr in GRID_LRS],
        [[opt] + [f4(fz[(fz.optimizer == opt) & (fz.lr == lr)].we_mean.iloc[0])
                  for lr in GRID_LRS] for opt in OPTIMIZERS])
    write_tables(os.path.join(out, "tables.md"), [
        ("E1-a closed form (20 datasets)", t_cf),
        ("E1-b combinations reaching the closed form "
         "(max ‖ŵ − ŵ_cf‖ over the 20 datasets is 0.0000 at 4 decimals), of 9 "
         "batch × init combinations", t_count),
        ("E1-c full batch, zero init: mean weight error", t_fz),
        ("E1-d all 72 combinations", t_full),
    ])

    cf_mean = float(cf["weight_error"].mean())
    panels = []
    for lr in GRID_LRS:
        c = {opt: np.mean(curves[(opt, lr)], axis=0) for opt in OPTIMIZERS}
        c["closed form"] = np.full(GRID_ITERS, cf_mean)
        panels.append((f"lr {lr}", c))
    plot_curves(panels, save_path=os.path.join(out, "curves.png"),
                ylabel="mean weight error")
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read()[:3000])


if __name__ == "__main__":
    main()
