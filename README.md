# Outliers & Gradient-Based Optimization for Linear Regression

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

A pure-NumPy study of how hand-coded gradient-based optimizers behave on
synthetic linear regression, and of how a small outlier population distorts the
fit. Everything is implemented from scratch (no scikit-learn, no autograd) so
the update rules are fully visible.

## Overview

The project has two parts, mirrored by two experiment scripts:

- **Baseline (`experiments/run_baseline.py`)** — On a clean synthetic dataset,
  four hand-coded optimizers (**Gradient Descent, AdaGrad, RMSProp, Adam**) are
  benchmarked across **3 weight initializations** (`random`, `zero`, `sparse`)
  and **3 batch schemes** (`full`, `mini-batch`, `SGD`), and compared against
  the closed-form normal-equation solution.

- **Outlier study (`experiments/run_outlier.py`)** — The data is a
  two-population mixture: 90% of samples follow weights `w1` (the clean
  population, `z == 1`) and 10% follow `w2 = -w1` (the outlier population,
  `z == 2`). We measure how contamination degrades a naive least-squares fit,
  run the same optimizer grid, and evaluate a custom method (**"ours"**) that
  periodically removes the highest-residual points and re-initializes Adam to
  recover the clean-population weights.

## Dataset

**Fully synthetic — nothing to download.** All data is generated in-code with a
fixed seed (`seed=0`), so every run is reproducible.

- **Clean data** (`generate_linear_data`): `X ~ Uniform[0,1)^{N x D}`,
  `y = X @ w_true + 0.1 * N(0, 1)`, with `N=1000`, `D=4`.
- **Mixture data** (`generate_mixture_data`): same `X`, but each sample follows
  `w1` with probability `p=0.9` and `w2 = -w1` otherwise; the population label
  is returned as `z`.

## Project structure

```
outlier-linear-regression/
├── src/
│   ├── data.py             # synthetic data generators (clean & mixture)
│   ├── optimizers.py       # GD / AdaGrad / RMSProp / Adam update steps
│   ├── regression.py       # closed-form normal-equation solver
│   ├── train.py            # unified trainer + experiment-grid runner
│   ├── outlier_removal.py  # the custom "ours" method
│   └── plots.py            # convergence-plot helpers
├── experiments/
│   ├── run_all.py          # regenerate every results/ artifact in one command
│   ├── run_baseline.py     # baseline optimizer benchmark (clean data)
│   └── run_outlier.py      # outlier study + "ours"
├── results/                # committed artifacts (figures + metrics), see RESULTS.md
│   ├── baseline/           # closed_form.json, optimizer_grid.csv, *.png
│   └── outlier/            # summary.json, optimizer_grid.csv, lr_sweep.png
├── docs/
│   └── Analyzing the Impact of Outliers ... .pdf   # project report
├── requirements.txt
├── RESULTS.md              # rendered results with figures and tables
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <your-fork-url>
cd outlier-linear-regression
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python experiments/run_all.py         # regenerate everything under results/
# or individually:
python experiments/run_baseline.py    # clean-data optimizer benchmark
python experiments/run_outlier.py     # outlier study + "ours"
```

Each script prints its result tables to stdout and writes figures + metrics
(`.png`, `.csv`, `.json`) to `results/` (created automatically). These
artifacts are committed to the repository — see **[RESULTS.md](RESULTS.md)** for
the rendered figures and summary tables.

You can also import the modules directly:

```python
from src.data import generate_mixture_data
from src.outlier_removal import ours

X, y, z, w1, w2 = generate_mixture_data(N=1000, D=4, p=0.9, seed=0)
w, est_hist, w_hist = ours(X, y, w1, eval_X=X, eval_y=y, eval_mask=(z == 1))
```

## Notes

- **One unified trainer.** The two notebooks each contained a near-duplicate
  training loop. They are merged into a single `src.train.train` /
  `run_experiment`. The only real difference — how per-epoch metrics are
  recorded — is exposed via a `record` mode: `"batch"` (notebook 1: pre-update
  batch MSE vs `w_true`) and `"eval"` (notebook 2: post-update MSE on the
  `z == 1` subset vs `w1`).
- **Fixed bug (aliasing).** In notebook 1 the optimizer accumulators were
  initialized as `m = v = grad_squared = running_avg = np.zeros_like(w)`, which
  made them all reference the *same* array. This is harmless for a single-
  optimizer run but incorrect in general; `src.optimizers.init_state` gives each
  accumulator its own array.
- **Closed form.** `closed_form_solution` supports both the explicit normal
  equation (`method="inv"`, as in notebook 1) and the pseudo-inverse
  (`method="pinv"`, as in notebook 2).
- **Reproducibility.** Numbers reproduce the notebooks. The one place output can
  differ by a small amount is the clean-only pseudo-inverse fit (notebook 2,
  Part 1): `np.linalg.pinv` on the ~908x4 clean subset is mildly sensitive to
  the installed NumPy/LAPACK version. This is a library difference, not an
  algorithm change — running the notebook's own code in this environment
  produces the same values these scripts do.
- **Preserved quirks.** The Adam timestep in `ours` resets to 0 at every
  re-initialization (bias correction restarts each cycle), matching the
  original method.
