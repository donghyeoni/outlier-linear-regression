"""D1: the starting method ``ours_v1`` on the development seeds.

Records, per dataset, the weight error, what each prune removed, the
gradient norm at the end of each cycle, and the closed form on the final
kept set. Writes ``results/dev1/per_seed.csv`` and ``tables.md``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import (DEV_SEEDS, f4, md_table, mean_sd, min_max, out_dir,
                    write_tables)
from mixture_baselines import load
from outlier_regression.outlier_removal import ours_v1
from outlier_regression.regression import closed_form_solution, weight_error


def main():
    out = out_dir("dev1")
    base = pd.read_csv(os.path.join(out_dir("dev0"), "baselines.csv"))
    rows = []
    for seed in DEV_SEEDS:
        X, y, z, w1 = load(seed)
        w, hist, info = ours_v1(X, y, w1)
        kept = info["kept_idx"]
        row = {"seed": seed, "ours_v1": hist[-1],
               "cf_on_kept": weight_error(
                   closed_form_solution(X[kept], y[kept]), w1),
               "kept": len(kept), "outliers_kept": int(np.sum(z[kept] == 2))}
        left = int(np.sum(z == 2))
        for i, rem in enumerate(info["removed"], 1):
            row[f"prune{i}_outliers"] = int(np.sum(z[rem] == 2))
            row[f"prune{i}_clean"] = int(np.sum(z[rem] == 1))
            left -= row[f"prune{i}_outliers"]
            row[f"prune{i}_outliers_left"] = left
        for i, g in enumerate(info["grad_norm"], 1):
            row[f"cycle{i}_grad_norm"] = g
        rows.append(row)
    df = pd.DataFrame(rows).merge(base, on="seed")
    df.to_csv(os.path.join(out, "per_seed.csv"), index=False)

    n = len(df)
    t_main = md_table(
        ["method", "weight error (mean ± SD)", "min–max"],
        [[m, mean_sd(df[c]), min_max(df[c])] for m, c in
         [("Oracle", "oracle"), ("Naive", "naive"), ("ours_v1", "ours_v1")]])
    t_cmp = md_table(
        ["comparison", "datasets"],
        [["ours_v1 < Naive", f"{int(np.sum(df.ours_v1 < df.naive))} / {n}"],
         ["ours_v1 < Oracle", f"{int(np.sum(df.ours_v1 < df.oracle))} / {n}"]])
    t_prune = md_table(
        ["prune (after cycle)", "outliers removed", "clean removed",
         "outliers left"],
        [[i, int(df[f"prune{i}_outliers"].sum()),
          int(df[f"prune{i}_clean"].sum()),
          int(df[f"prune{i}_outliers_left"].sum())] for i in range(1, 5)])
    t_grad = md_table(
        ["cycle", "‖∇‖ at the end (mean)", "min–max"],
        [[i, f4(df[f"cycle{i}_grad_norm"].mean()),
          min_max(df[f"cycle{i}_grad_norm"])] for i in range(1, 6)])
    t_kept = md_table(
        ["quantity", "value"],
        [["samples kept after the last prune", f"{df.kept.min()}–{df.kept.max()}"],
         ["outliers kept, total over datasets", int(df.outliers_kept.sum())],
         ["ours_v1 (trained)", mean_sd(df.ours_v1)],
         ["closed form on the final kept set", mean_sd(df.cf_on_kept)]])
    write_tables(os.path.join(out, "tables.md"), [
        ("D1-a weight error", t_main),
        ("D1-b per-dataset comparison", t_cmp),
        (f"D1-c prunes, totals over the {n} datasets "
         f"({int(df.n_outliers.sum())} outliers in total)", t_prune),
        ("D1-d gradient norm at the end of each cycle", t_grad),
        ("D1-e final kept set", t_kept),
    ])
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
