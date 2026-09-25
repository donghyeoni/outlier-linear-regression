"""D5: the chosen method (readmit, k = 3.5) with a 50-cycle cap at three
learning rates. Writes ``results/dev5/per_run.csv`` and ``tables.md``."""

from __future__ import annotations

import os

import pandas as pd

from common import DEV_SEEDS, f4, md_table, out_dir, write_tables
from mixture_baselines import load
from outlier_regression.outlier_removal import ours_v2

LRS = (0.01, 0.1, 0.5)


def main():
    out = out_dir("dev5")
    rows = []
    for seed in DEV_SEEDS:
        X, y, z, w1 = load(seed)
        for lr in LRS:
            w, hist, info = ours_v2(X, y, w1, prune_rule="readmit",
                                    stop_k=3.5, converge_tol=1e-5,
                                    max_cycles=50, learning_rate=lr)
            rows.append({"seed": seed, "lr": lr, "weight_error": hist[-1],
                         "cycles": len(info["cycle_iters"]),
                         "settled": bool(info["settled"]),
                         "longest_cycle": int(max(info["cycle_iters"]))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_run.csv"), index=False)
    s = (df.groupby("lr").agg(we_mean=("weight_error", "mean"),
                              we_sd=("weight_error", lambda v: v.std(ddof=0)),
                              cycles_max=("cycles", "max"),
                              settled=("settled", "sum"),
                              longest=("longest_cycle", "max")).reset_index())
    t = md_table(
        ["lr", "weight error (mean ± SD)", "cycles (max)",
         "stopped by the rule", "longest cycle (iterations)"],
        [[r.lr, f"{f4(r.we_mean)} ± {f4(r.we_sd)}", int(r.cycles_max),
          f"{int(r.settled)} / 20", int(r.longest)] for r in s.itertuples()])
    write_tables(os.path.join(out, "tables.md"),
                 [("D5-a readmit, k = 3.5, tol 1e-5, at most 50 cycles, "
                   "20 datasets", t)])
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
