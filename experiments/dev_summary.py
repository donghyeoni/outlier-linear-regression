"""Development summary: each step of D1-D4 on the development seeds.

Reads the per-run files of D0-D4 and writes ``results/dev_summary/``:
``per_seed.csv``, ``tables.md`` and ``strip.png``.
"""

from __future__ import annotations

import os

import pandas as pd

from common import RESULTS, f4, md_table, mean_sd, out_dir, write_tables
from outlier_regression.plots import plot_strip

STEPS = [
    ("Oracle", "reference", "dev0/baselines.csv", "oracle", None),
    ("Naive", "reference", "dev0/baselines.csv", "naive", None),
    ("ours_v1", "D1", "dev1/per_seed.csv", "ours_v1", None),
    ("+ per-cycle convergence", "D2", "dev2/per_run.csv", "weight_error",
     lambda d: d[(d.lr == 0.1) & (d.tol == "1e-5")]),
    ("+ stopping rule (ratio-stop)", "D3", "dev3/per_run.csv", "weight_error",
     lambda d: d[d.k == "4.0"]),
    ("threshold, k = 4", "D4", "dev4/per_run.csv", "weight_error",
     lambda d: d[(d.rule == "threshold") & (d.k == 4.0)]),
    ("readmit, k = 3.5 (ours)", "D4", "dev4/per_run.csv", "weight_error",
     lambda d: d[(d.rule == "readmit") & (d.k == 3.5)]),
]
COLORS = ["#4C78A8", "#E45756", "#9D755D", "#72B7B2", "#F58518", "#EECA3B",
          "#54A24B"]
SHORT = ["Oracle", "Naive", "ours_v1", "+conv.", "ratio-stop", "threshold",
         "ours"]


def main():
    out = out_dir("dev_summary")
    cols = {}
    for name, _, path, col, sel in STEPS:
        d = pd.read_csv(os.path.join(RESULTS, path), dtype={"k": str}
                        if "dev3" in path else None)
        if sel is not None:
            d = sel(d)
        cols[name] = d.sort_values("seed")[col].to_numpy()
        assert len(cols[name]) == 20, name
    pd.DataFrame(cols).to_csv(os.path.join(out, "per_seed.csv"), index=False)
    t = md_table(["step", "section", "weight error (mean ± SD)", "median"],
                 [[name, sec, mean_sd(cols[name]), f4(pd.Series(cols[name]).median())]
                  for name, sec, *_ in STEPS])
    write_tables(os.path.join(out, "tables.md"),
                 [("S-a development steps, seeds 0–19", t)])
    plot_strip([(s, cols[name], c) for s, (name, *_), c
                in zip(SHORT, STEPS, COLORS)],
               save_path=os.path.join(out, "strip.png"),
               title="Development steps on the 20 development datasets")
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
