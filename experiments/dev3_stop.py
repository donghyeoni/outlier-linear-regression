"""D3: ratio pruning with the MAD stopping rule.

Writes ``results/dev3/per_run.csv``, ``summary.csv`` and ``tables.md``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import DEV_SEEDS, f4, md_table, out_dir, write_tables
from mixture_baselines import load
from outlier_regression.diagnostics import selection_summary
from outlier_regression.outlier_removal import ours_v2

KS = (None, 2.0, 2.5, 3.0, 3.5, 4.0)
TOL, LR = 1e-5, 0.1


def main():
    out = out_dir("dev3")
    rows = []
    for seed in DEV_SEEDS:
        X, y, z, w1 = load(seed)
        for k in KS:
            w, hist, info = ours_v2(X, y, w1, prune_rule="ratio",
                                    learning_rate=LR, max_cycles=5,
                                    converge_tol=TOL, stop_k=k)
            rows.append({"seed": seed, "k": "none" if k is None else k,
                         "weight_error": hist[-1],
                         "prunes": len(info["removed"]),
                         "stopped": bool(info["settled"]),
                         **selection_summary(X, y, z, w1, info["kept_idx"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_run.csv"), index=False)
    s = (df.groupby("k", sort=False)
         .agg(we_mean=("weight_error", "mean"),
              we_sd=("weight_error", lambda v: v.std(ddof=0)),
              prunes=("prunes", "mean"), stopped=("stopped", "sum"),
              clean_kept=("clean_kept", "mean"),
              out_kept=("outliers_kept", "sum"),
              pos=("clean_out_pos", "sum"), neg=("clean_out_neg", "sum"))
         .reset_index())
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    t = md_table(
        ["k", "weight error (mean ± SD)", "prunes (mean)",
         "stopped by the rule", "clean kept (mean)", "outliers kept (total)",
         "clean left out, noise + / − (total)"],
        [[r.k, f"{f4(r.we_mean)} ± {f4(r.we_sd)}", f"{r.prunes:.2f}",
          f"{int(r.stopped)} / 20" if r.k != "none" else "–",
          f"{r.clean_kept:.2f}", int(r.out_kept), f"{int(r.pos)} / {int(r.neg)}"]
         for r in s.itertuples()])
    write_tables(os.path.join(out, "tables.md"),
                 [("D3-a ratio pruning with stopping rule k (tol 1e-5, "
                   "lr 0.1, at most 5 cycles, 20 datasets)", t)])
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
