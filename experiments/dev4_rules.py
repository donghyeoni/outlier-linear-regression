"""D4: ratio-stop vs threshold vs readmit pruning.

Writes ``results/dev4/per_run.csv``, ``summary.csv`` and ``tables.md``.
"""

from __future__ import annotations

import os

import pandas as pd

from common import DEV_SEEDS, f4, md_table, out_dir, write_tables
from mixture_baselines import load
from outlier_regression.diagnostics import selection_summary
from outlier_regression.outlier_removal import ours_v2

TOL, LR = 1e-5, 0.1
KS = (2.0, 2.5, 3.0, 3.5, 4.0)
CONFIGS = ([("ratio-stop", 4.0, 5)]
           + [(rule, k, 20) for rule in ("threshold", "readmit") for k in KS])


def main():
    out = out_dir("dev4")
    rows = []
    for seed in DEV_SEEDS:
        X, y, z, w1 = load(seed)
        for name, k, cycles in CONFIGS:
            rule = "ratio" if name == "ratio-stop" else name
            w, hist, info = ours_v2(X, y, w1, prune_rule=rule,
                                    learning_rate=LR, max_cycles=cycles,
                                    converge_tol=TOL, stop_k=k)
            rows.append({"seed": seed, "rule": name, "k": k,
                         "weight_error": hist[-1],
                         "cycles": len(info["cycle_iters"]),
                         "settled": bool(info["settled"]),
                         **selection_summary(X, y, z, w1, info["kept_idx"])})
        print(f"seed {seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_run.csv"), index=False)
    s = (df.groupby(["rule", "k"], sort=False)
         .agg(we_mean=("weight_error", "mean"),
              we_sd=("weight_error", lambda v: v.std(ddof=0)),
              cycles=("cycles", "mean"), cycles_max=("cycles", "max"),
              settled=("settled", "sum"),
              clean_kept=("clean_kept", "mean"),
              out_kept=("outliers_kept", "sum"),
              pos=("clean_out_pos", "sum"), neg=("clean_out_neg", "sum"))
         .reset_index())
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    t = md_table(
        ["rule", "k", "weight error (mean ± SD)", "cycles (mean)",
         "cycles (max)", "stopped by the rule", "clean kept (mean)",
         "outliers kept (total)", "clean left out, noise + / − (total)"],
        [[r.rule, r.k, f"{f4(r.we_mean)} ± {f4(r.we_sd)}", f"{r.cycles:.2f}",
          int(r.cycles_max), f"{int(r.settled)} / 20", f"{r.clean_kept:.2f}",
          int(r.out_kept), f"{int(r.pos)} / {int(r.neg)}"]
         for r in s.itertuples()])
    write_tables(os.path.join(out, "tables.md"),
                 [("D4-a pruning rules (tol 1e-5, lr 0.1, 20 datasets; "
                   "ratio-stop at most 5 cycles, the others at most 20)", t)])
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
