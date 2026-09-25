"""D2: per-cycle convergence for the ratio pruning of ``ours_v1``.

Writes ``results/dev2/per_run.csv``, ``summary.csv`` and ``tables.md``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import DEV_SEEDS, f4, md_table, out_dir, write_tables
from mixture_baselines import load
from outlier_regression.outlier_removal import ours_v2
from outlier_regression.regression import closed_form_solution, weight_error

LRS = (0.01, 0.1, 0.5)
TOLS = (None, 1e-3, 1e-4, 1e-5, 1e-6)
CAP = 20000


def tol_label(tol):
    return "fixed 200" if tol is None else f"{tol:.0e}".replace("e-0", "e-")


def main():
    out = out_dir("dev2")
    rows = []
    for seed in DEV_SEEDS:
        X, y, z, w1 = load(seed)
        for lr in LRS:
            for tol in TOLS:
                w, hist, info = ours_v2(X, y, w1, prune_rule="ratio",
                                        learning_rate=lr, max_cycles=5,
                                        cycle_iters=200, converge_tol=tol,
                                        max_cycle_iters=CAP)
                kept = info["kept_idx"]
                rows.append({
                    "seed": seed, "lr": lr, "tol": tol_label(tol),
                    "weight_error": hist[-1],
                    "cf_on_kept": weight_error(
                        closed_form_solution(X[kept], y[kept]), w1),
                    "total_iters": int(sum(info["cycle_iters"])),
                    "max_cycle_iters": int(max(info["cycle_iters"])),
                    "capped_cycles": int(sum(n == CAP for n in info["cycle_iters"])
                                         if tol is not None else 0),
                    "outliers_kept": int(np.sum(z[kept] == 2)),
                })
        print(f"seed {seed} done")
    df = pd.DataFrame(rows)
    df["gap"] = (df.weight_error - df.cf_on_kept).abs()
    df.to_csv(os.path.join(out, "per_run.csv"), index=False)
    s = (df.groupby(["lr", "tol"], sort=False)
         .agg(we_mean=("weight_error", "mean"),
              we_sd=("weight_error", lambda v: v.std(ddof=0)),
              cf_mean=("cf_on_kept", "mean"), gap_max=("gap", "max"),
              iters_mean=("total_iters", "mean"),
              cycle_max=("max_cycle_iters", "max"),
              capped=("capped_cycles", "sum"),
              out_kept=("outliers_kept", "sum"))
         .reset_index())
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    t = md_table(
        ["lr", "cycle length", "weight error (mean ± SD)",
         "closed form on kept set (mean)", "max |trained − closed form|",
         "total iterations (mean)", "longest cycle", "capped cycles",
         "outliers kept (total)"],
        [[r.lr, r.tol, f"{f4(r.we_mean)} ± {f4(r.we_sd)}", f4(r.cf_mean),
          f4(r.gap_max), f"{r.iters_mean:.1f}", int(r.cycle_max),
          int(r.capped), int(r.out_kept)] for r in s.itertuples()])
    write_tables(os.path.join(out, "tables.md"),
                 [("D2-a ratio pruning, 5 cycles, 20 datasets "
                   "(cycle length: fixed 200 iterations or until ‖∇‖ < tol)",
                   t)])
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
