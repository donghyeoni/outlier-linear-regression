"""Oracle, Naive and the optimizer grid on mixture data.

Used by D0 (development seeds) and T1 (final-evaluation seeds).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import GRID_ITERS, GRID_LRS, f4, md_table, mean_sd, min_max
from outlier_regression.data import generate_mixture_data
from outlier_regression.regression import closed_form_solution, weight_error
from outlier_regression.train import run_grid

KEYS = ["optimizer", "batch", "init", "lr"]


def load(seed):
    X, y, z, w1, _ = generate_mixture_data(N=1000, D=4, p=0.9, seed=seed)
    return X, y, z, w1


def run(seeds, out):
    """Write ``baselines.csv`` (Oracle / Naive per seed) and ``grid.csv``
    (one row per seed and grid combination) to ``out``; return both."""
    base, grids = [], []
    for seed in seeds:
        X, y, z, w1 = load(seed)
        w_or = closed_form_solution(X[z == 1], y[z == 1])
        w_na = closed_form_solution(X, y)
        base.append({"seed": seed, "n_outliers": int(np.sum(z == 2)),
                     "oracle": weight_error(w_or, w1),
                     "naive": weight_error(w_na, w1)})
        df, _ = run_grid(X, y, w1, w_na, learning_rates=GRID_LRS,
                         num_iters=GRID_ITERS)
        df.insert(0, "seed", seed)
        grids.append(df)
        print(f"seed {seed} done")
    base = pd.DataFrame(base)
    grid = pd.concat(grids, ignore_index=True)
    base.to_csv(os.path.join(out, "baselines.csv"), index=False)
    grid.to_csv(os.path.join(out, "grid.csv"), index=False)
    return base, grid


def grid_summary(grid):
    s = (grid.groupby(KEYS, sort=False)
         .agg(we_mean=("weight_error", "mean"),
              we_sd=("weight_error", lambda v: v.std(ddof=0)),
              we_min=("weight_error", "min"),
              we_max=("weight_error", "max"),
              dist_max=("dist_to_star", "max"))
         .reset_index())
    s["reaches_naive"] = s["dist_max"] < 5e-5
    return s


def tables(base, grid, tag):
    """Markdown tables: Oracle / Naive, grid overview, all combinations."""
    s = grid_summary(grid)
    best = s.loc[s.we_mean.idxmin()]
    t_base = md_table(
        ["method", "weight error (mean ± SD)", "min–max"],
        [["Oracle", mean_sd(base.oracle), min_max(base.oracle)],
         ["Naive", mean_sd(base.naive), min_max(base.naive)]])
    t_over = md_table(
        ["quantity", "value"],
        [["outliers per dataset (min–max)",
          f"{base.n_outliers.min()}–{base.n_outliers.max()}"],
         ["combinations reaching Naive (max ‖ŵ − ŵ_naive‖ is 0.0000 at 4 decimals)",
          f"{int(s.reaches_naive.sum())} / {len(s)}"],
         ["smallest mean weight error over the grid",
          f"{f4(best.we_mean)} ({best.optimizer}, {best.batch}, "
          f"{best.init}, lr {best.lr})"],
         ["datasets where that combination is below Naive",
          f"{int(np.sum(grid[(grid.optimizer == best.optimizer) & (grid.batch == best.batch) & (grid.init == best.init) & (grid.lr == best.lr)].weight_error.values < base.naive.values))} / {len(base)}"]])
    t_full = md_table(
        ["optimizer", "batch", "init", "lr", "weight error (mean ± SD)",
         "min–max", "max ‖ŵ − ŵ_naive‖"],
        [[r.optimizer, r.batch, r.init, r.lr,
          f"{f4(r.we_mean)} ± {f4(r.we_sd)}",
          f"{f4(r.we_min)}–{f4(r.we_max)}", f4(r.dist_max)]
         for r in s.itertuples()])
    return [(f"{tag}-a Oracle and Naive", t_base),
            (f"{tag}-b optimizer grid overview", t_over),
            (f"{tag}-c optimizer grid, all 72 combinations", t_full)], s
