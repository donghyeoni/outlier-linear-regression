"""T1: final evaluation on seeds 100-199, as planned in the log.

Writes ``results/t1/``: ``baselines.csv``, ``grid.csv``, ``grid_summary.csv``,
``per_seed.csv``, ``tests.json``, ``tables.md`` and ``strip.png``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import mixture_baselines
from common import (TEST_SEEDS, f4, fp, md_table, mean_sd, min_max, out_dir,
                    save_json, write_tables)
from mixture_baselines import load
from outlier_regression.diagnostics import selection_summary
from outlier_regression.outlier_removal import ours, ours_v1, ours_v2
from outlier_regression.plots import plot_strip
from outlier_regression.stats import bootstrap_ci, sign_flip_p

BASELINES = {
    "ours_v1": lambda X, y, w1: ours_v1(X, y, w1),
    "ratio-stop": lambda X, y, w1: ours_v2(
        X, y, w1, prune_rule="ratio", stop_k=4.0, converge_tol=1e-5,
        max_cycles=5, learning_rate=0.1),
    "threshold": lambda X, y, w1: ours_v2(
        X, y, w1, prune_rule="threshold", stop_k=4.0, converge_tol=1e-5,
        max_cycles=20, learning_rate=0.1),
}
TESTED = ("naive", "ours_v1", "ratio-stop", "threshold")
LABEL = {"oracle": "Oracle", "naive": "Naive", "ours_v1": "ours_v1",
         "ratio-stop": "ratio-stop", "threshold": "threshold", "ours": "ours"}


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, run = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def main():
    out = out_dir("t1")
    base, grid = mixture_baselines.run(TEST_SEEDS, out)
    grid_sections, gsum = mixture_baselines.tables(base, grid, "T1-grid")
    gsum.to_csv(os.path.join(out, "grid_summary.csv"), index=False)

    rows = []
    for seed in TEST_SEEDS:
        X, y, z, w1 = load(seed)
        row = {"seed": seed}
        for name, fn in BASELINES.items():
            row[name] = fn(X, y, w1)[1][-1]
        _, hist, info = ours(X, y, w1)
        row.update({"ours": hist[-1], "cycles": len(info["cycle_iters"]),
                    "settled": bool(info["settled"]),
                    "longest_cycle": int(max(info["cycle_iters"])),
                    **selection_summary(X, y, z, w1, info["kept_idx"])})
        for lr in (0.01, 0.5):
            row[f"ours_lr{lr}"] = ours(X, y, w1, learning_rate=lr)[1][-1]
        rows.append(row)
        print(f"seed {seed} done")
    df = base.merge(pd.DataFrame(rows), on="seed")
    df.to_csv(os.path.join(out, "per_seed.csv"), index=False)
    n = len(df)

    tests = {}
    for b in TESTED:
        d = (df.ours - df[b]).values
        tests[b] = {"mean": float(d.mean()),
                    "ci95": bootstrap_ci(d, 10000, np.random.default_rng(1)),
                    "n_below": int(np.sum(d < 0)),
                    "p": sign_flip_p(d, 100000, np.random.default_rng(0))}
    adj = holm(np.array([tests[b]["p"] for b in TESTED]))
    for b, a in zip(TESTED, adj):
        tests[b]["p_holm"] = float(a)
        tests[b]["reject"] = bool(a < 0.05)
    d_or = (df.ours - df.oracle).values
    tests["oracle (descriptive)"] = {
        "mean": float(d_or.mean()),
        "ci95": bootstrap_ci(d_or, 10000, np.random.default_rng(1)),
        "n_below": int(np.sum(d_or < 0))}
    save_json(tests, os.path.join(out, "tests.json"))

    best = gsum.loc[gsum.we_mean.idxmin()]
    bm = grid[(grid.optimizer == best.optimizer) & (grid.batch == best.batch)
              & (grid.init == best.init) & (grid.lr == best.lr)]
    df = df.merge(bm[["seed", "weight_error"]].rename(
        columns={"weight_error": "best_grid"}), on="seed")
    grid_label = (f"optimizer grid, best mean ({best.optimizer}, "
                  f"{best.batch}, {best.init}, lr {best.lr})")
    t_main = md_table(
        ["method", "weight error (mean ± SD)", "median", "min–max"],
        [[LABEL.get(c, grid_label), mean_sd(df[c]), f4(df[c].median()),
          min_max(df[c])]
         for c in ("oracle", "naive", "best_grid", "ours_v1", "ratio-stop",
                   "threshold", "ours")])
    t_tests = md_table(
        ["B", "mean of ours − B", "95% CI", "ours < B", "p", "p (Holm)",
         "H0 rejected"],
        [[LABEL[b], f4(tests[b]["mean"]),
          f"[{f4(tests[b]['ci95'][0])}, {f4(tests[b]['ci95'][1])}]",
          f"{tests[b]['n_below']} / {n}", fp(tests[b]["p"]),
          fp(tests[b]["p_holm"]), "yes" if tests[b]["reject"] else "no"]
         for b in TESTED])
    o = tests["oracle (descriptive)"]
    t_or = md_table(
        ["quantity", "value"],
        [["mean of ours − Oracle", f4(o["mean"])],
         ["95% CI", f"[{f4(o['ci95'][0])}, {f4(o['ci95'][1])}]"],
         ["datasets with ours < Oracle", f"{o['n_below']} / {n}"]])
    t_lr = md_table(
        ["lr", "weight error of ours (mean ± SD)"],
        [[lr, mean_sd(df[c])] for lr, c in
         ((0.01, "ours_lr0.01"), (0.1, "ours"), (0.5, "ours_lr0.5"))])
    clean = int((df.n_outliers.rsub(1000)).sum())
    t_diag = md_table(
        ["quantity", "value"],
        [["cycles used (min–max)", f"{df.cycles.min()}–{df.cycles.max()}"],
         ["runs stopped by the rule", f"{int(df.settled.sum())} / {n}"],
         ["longest cycle (iterations)", int(df.longest_cycle.max())],
         ["clean samples kept (total)", f"{int(df.clean_kept.sum())} of {clean}"],
         ["clean left out, noise + / − (total)",
          f"{int(df.clean_out_pos.sum())} / {int(df.clean_out_neg.sum())}"],
         ["outliers kept (total)",
          f"{int(df.outliers_kept.sum())} of {int(df.n_outliers.sum())}"],
         ["datasets with at least one outlier kept",
          f"{int(np.sum(df.outliers_kept > 0))} / {n}"]])
    write_tables(os.path.join(out, "tables.md"), [
        ("T1-a weight error, 100 datasets", t_main),
        ("T1-b pre-registered tests", t_tests),
        ("T1-c ours − Oracle (descriptive)", t_or),
        ("T1-d ours by learning rate (descriptive)", t_lr),
        ("T1-e selection by ours (descriptive)", t_diag),
    ] + grid_sections)

    plot_strip([("Oracle", df.oracle, "#4C78A8"),
                ("Naive", df.naive, "#E45756"),
                ("best grid", df.best_grid, "#B279A2"),
                ("ours_v1", df.ours_v1, "#9D755D"),
                ("ratio-stop", df["ratio-stop"], "#F58518"),
                ("threshold", df.threshold, "#EECA3B"),
                ("ours", df.ours, "#54A24B")],
               save_path=os.path.join(out, "strip.png"),
               title="Weight error on the 100 evaluation datasets")
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read()[:4000])


if __name__ == "__main__":
    main()
