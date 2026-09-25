"""Draw the 1-D concept figure used in the README.

A toy version of the outlier study with a single feature: 90% of the points
follow ``y = w x`` and 10% follow ``y = -w x``. Fitting a line to all points
pulls it away from the true line. Not an experiment result, only an
illustration of the setup.

Writes ``docs/images/concept.png`` and the fitted slopes to
``docs/images/concept_values.json``.

Usage
-----
    python experiments/plot_concept.py
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from common import REPO_ROOT

OUT_DIR = os.path.join(REPO_ROOT, "docs", "images")

# Korean-capable fonts, applied only while this figure is drawn so the other
# result plots keep matplotlib's defaults.
FONT_RC = {
    "font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic",
                    "DejaVu Sans"],
    "axes.unicode_minus": False,
}
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with plt.rc_context(FONT_RC):
        path = draw()
    print(f"Saved {path}")


def draw():
    rng = np.random.default_rng(0)
    n, w = 200, 1.0
    x = rng.random(n)
    outlier = rng.random(n) >= 0.9
    y = np.where(outlier, -w, w) * x + 0.1 * rng.standard_normal(n)

    w_naive = (x @ y) / (x @ x)
    w_clean = (x[~outlier] @ y[~outlier]) / (x[~outlier] @ x[~outlier])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x[~outlier], y[~outlier], s=14, color="#4C78A8",
               label="정상 집단 (90%)")
    ax.scatter(x[outlier], y[outlier], s=22, color="#E45756", marker="x",
               label="이상치 집단 (10%, $w_2 = -w_1$)")
    xs = np.array([0, 1])
    ax.plot(xs, w_clean * xs, color="#4C78A8", lw=2,
            label=f"정상 집단 적합 (기울기 {w_clean:.2f})")
    ax.plot(xs, w_naive * xs, color="#E45756", lw=2, ls="--",
            label=f"전체 데이터 적합 (기울기 {w_naive:.2f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("이상치 혼입에 따른 회귀선 편향 (1차원 예시)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "concept.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    with open(os.path.join(OUT_DIR, "concept_values.json"), "w") as f:
        json.dump({"slope_true": w, "slope_clean_fit": float(w_clean),
                   "slope_all_fit": float(w_naive),
                   "n_points": n, "n_outliers": int(outlier.sum())}, f,
                  indent=2)
    return path


if __name__ == "__main__":
    main()
