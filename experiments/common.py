"""Shared settings and table helpers for the experiment scripts."""

from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")

EXP1_SEEDS = tuple(range(0, 20))
DEV_SEEDS = tuple(range(0, 20))
TEST_SEEDS = tuple(range(100, 200))
GRID_LRS = (0.01, 0.1)
GRID_ITERS = 1000


def out_dir(name):
    path = os.path.join(RESULTS, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def f4(x):
    """A value rounded to 4 decimals; ``-0.0000`` is written ``0.0000``."""
    s = f"{x:.4f}"
    return "0.0000" if s == "-0.0000" else s


def fp(p):
    """A p-value to 4 decimals, or ``< 0.0001`` if it rounds to 0."""
    return "< 0.0001" if round(p, 4) == 0 else f"{p:.4f}"


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return f"{f4(v.mean())} ± {f4(v.std())}"


def min_max(v):
    v = np.asarray(v, dtype=float)
    return f"{f4(v.min())}–{f4(v.max())}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_tables(path, sections):
    """Write ``[(title, table_markdown), ...]`` to a ``tables.md`` file."""
    with open(path, "w", encoding="utf-8") as f:
        for title, table in sections:
            f.write(f"### {title}\n\n{table}\n\n")
