"""D0: Oracle, Naive and the optimizer grid on the development seeds.

Writes ``results/dev0/``: ``baselines.csv``, ``grid.csv``,
``grid_summary.csv`` and ``tables.md``.
"""

from __future__ import annotations

import os

import mixture_baselines
from common import DEV_SEEDS, out_dir, write_tables


def main():
    out = out_dir("dev0")
    base, grid = mixture_baselines.run(DEV_SEEDS, out)
    sections, summary = mixture_baselines.tables(base, grid, "D0")
    summary.to_csv(os.path.join(out, "grid_summary.csv"), index=False)
    write_tables(os.path.join(out, "tables.md"), sections)


if __name__ == "__main__":
    main()
