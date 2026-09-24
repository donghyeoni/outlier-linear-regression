# Experiment Log

Record of what was tried, what was observed and what was decided. The
[README](../README.md) shows only the final method and its results; the
development process lives here.

- Experiments are numbered **E0–E8** and listed **oldest first**. E0 is the
  starting point; E8 is the final evaluation reported in the README.
- The software environment is listed at the end.
- Unless stated otherwise: `N = 1000`, `D = 4`, `X ~ U[0,1)^{1000×4}`,
  noise `ε ~ 0.1·N(0, 1)`, clean ratio `p = 0.9`, outliers follow
  `w2 = -w1`, weight initialization `N(0, 1)` with `seed = 0`.
- **Metric.** Weight error `‖ŵ − w_ref‖₂`, with `w_ref = w_true` in
  experiment 1 and `w_ref = w1` (clean population) in experiment 2. Residuals
  are `r = Xŵ − y`. Standard deviations are population standard deviations.
- **Epoch.** One parameter update. Full batch uses all data, mini-batch 32
  samples, SGD one sample.
- **Seed.** A dataset seed determines `X`, the true weights, the noise and
  which samples are outliers.

**Names used in this log.**

| name | meaning |
| --- | --- |
| `ours` v1 | function `ours_v1`: fixed 200-epoch cycles, removal of the top 10% of residuals, re-initialization (E0–E2) |
| ratio | `ours_v2(prune_rule="ratio")` with stopping rule and per-cycle convergence (E3) |
| threshold | `ours_v2(prune_rule="threshold")` (E4) |
| readmit | `ours_v2(prune_rule="readmit")` (E5–E6); the final method |
| `ours` | function `ours`: the final method with its settings as defaults, i.e. `ours_v2(prune_rule="readmit", stop_k=3, converge_tol=1e-5, num_cycles=5, learning_rate=0.1)`; used in the README and in E8 |

---

## Current status

**Final method:** function `ours` = `ours_v2(..., prune_rule="readmit", stop_k=3,
converge_tol=1e-5, num_cycles=5, learning_rate=0.1)`.
Each cycle trains Adam until convergence. The inlier set is then recomputed
from all samples (`|r| ≤ 3·σ_MAD`), and training stops when the set no
longer changes. Adopted after the pre-registered test in E6 and evaluated on
untouched datasets in E8.

| method (E8, seeds 141–240, n = 100) | weight error |
| --- | --- |
| Oracle | 0.0188 ± 0.0082 |
| Naive | 0.2478 ± 0.0674 |
| `ours` (readmit) | 0.0188 ± 0.0078 |

## Summary of experiments

| # | question | result | decision |
| --- | --- | --- | --- |
| E0 | Starting point | Experiment 1 (optimizers on clean data) and `ours` v1 on seed 0: 0.14654 vs Naive 0.13124 | Baseline for everything below |
| E1 | Why is `ours` v1 worse than Naive? | At lr 0.01, pruning never stops and the last cycle does not converge | Superseded in part by E2 |
| E2 | Does this hold at other learning rates? | At lr 0.1 / 0.5 `ours` reaches 0.030 (Oracle 0.029). lr 0.01 fails because cycles do not converge | Two limitations identified (① convergence, ② no stopping rule) |
| E3 | Fix ① and ② | Per-cycle convergence removes the lr dependence; the stopping rule removes over-pruning. Held-out (seeds 1–20): 0.0224 | Ratio-based `ours_v2` becomes the final method |
| E4 | Threshold instead of fixed 10% removal? | Worse on held-out data (0.0262 vs 0.0224): one-sided removal of clean samples | Rejected |
| E5 | Re-admit wrongly removed samples? | 0.0190 vs ratio 0.0218 on seeds 21–40, not significant (p = 0.15) | Candidate; confirmatory test planned |
| E6 | Pre-registered test: readmit vs ratio | n = 100, mean diff −0.0025, p = 0.00012 | Readmit adopted |
| E7 | Descriptive values on the E6 seeds | Optimizer grid, lr, diagnostics on seeds 41–140 | Superseded for reporting by E8 (these seeds were used for the E6 choice) |
| E8 | Final evaluation on untouched seeds | Seeds 141–240: `ours` 0.0188 vs Oracle 0.0188, Naive 0.2478 | Reported in the README |

## Use of data seeds

Each seed generates a different dataset. Seeds were used in this order.
Seeds 0–140 were used for development and selection; the final evaluation
(E8) uses seeds 141–240, which were not used for any decision.

| seeds | used in | purpose |
| --- | --- | --- |
| 0 | E0–E5 | development |
| 1–20 | E3, E4 (secondary in E5) | held-out evaluation of the ratio method; selection between ratio and threshold |
| 21–40 | E5 | selection between ratio and readmit |
| 41–140 | E6, E7 | selection between ratio and readmit (pre-registered test); descriptive values |
| 141–240 | E8 | final evaluation only; no decision depends on it |

---

## E0 — Starting point: experiment 1 and `ours` v1

Experiment 1 and the first version of the method, `ours` v1 (function
`ours_v1`: fixed 200-epoch cycles, removal of the top 10% of residuals,
re-initialization), on the development dataset (`run_baseline.py`,
`run_outlier.py`, seed 0).

### Experiment 1 — optimizers on clean data (seed 0, lr 0.1, 1000 epochs, mini-batch size 32)

`results/baseline/closed_form.json`, `results/baseline/optimizer_grid.csv`

Closed-form solution (explicit normal equation): weight error 0.0236
(0.02362), MSE 0.0103 (0.01028). True weights 0.2926, 0.5665, 0.1374,
0.3497; estimated 0.3032, 0.5620, 0.1179, 0.3563.

Full grid (estimation error = pre-update batch MSE, as recorded by the trainer):

| optimizer | batch | init | estimation error | weight error |
| --- | --- | --- | --- | --- |
| GD | SGD | random | 0.01008 | 0.09510 |
| GD | SGD | sparse | 0.00478 | 0.11321 |
| GD | SGD | zero | 0.00307 | 0.09685 |
| GD | full | random | 0.01028 | 0.02362 |
| GD | full | sparse | 0.01028 | 0.02362 |
| GD | full | zero | 0.01028 | 0.02362 |
| GD | mini-batch | random | 0.01168 | 0.02331 |
| GD | mini-batch | sparse | 0.01168 | 0.02331 |
| GD | mini-batch | zero | 0.01168 | 0.02331 |
| AdaGrad | SGD | random | 0.00056 | 0.50181 |
| AdaGrad | SGD | sparse | 0.01513 | 0.20485 |
| AdaGrad | SGD | zero | 0.00353 | 0.03876 |
| AdaGrad | full | random | 0.01239 | 0.17724 |
| AdaGrad | full | sparse | 0.01028 | 0.02376 |
| AdaGrad | full | zero | 0.01028 | 0.02362 |
| AdaGrad | mini-batch | random | 0.01284 | 0.20284 |
| AdaGrad | mini-batch | sparse | 0.01166 | 0.02380 |
| AdaGrad | mini-batch | zero | 0.01165 | 0.02357 |
| RMSProp | SGD | random | 0.00039 | 0.27911 |
| RMSProp | SGD | sparse | 0.00711 | 0.37162 |
| RMSProp | SGD | zero | 0.00020 | 0.19650 |
| RMSProp | full | random | 0.02108 | 0.09931 |
| RMSProp | full | sparse | 0.02108 | 0.09931 |
| RMSProp | full | zero | 0.02108 | 0.10608 |
| RMSProp | mini-batch | random | 0.03122 | 0.11616 |
| RMSProp | mini-batch | sparse | 0.03122 | 0.11616 |
| RMSProp | mini-batch | zero | 0.03122 | 0.11616 |
| Adam | SGD | random | 0.00811 | 0.28725 |
| Adam | SGD | sparse | 0.00078 | 0.27639 |
| Adam | SGD | zero | 0.00619 | 0.41186 |
| Adam | full | random | 0.01028 | 0.02362 |
| Adam | full | sparse | 0.01028 | 0.02362 |
| Adam | full | zero | 0.01028 | 0.02362 |
| Adam | mini-batch | random | 0.01308 | 0.02653 |
| Adam | mini-batch | sparse | 0.01261 | 0.04543 |
| Adam | mini-batch | zero | 0.01316 | 0.03097 |

At `init=zero`, `batch=full`, GD / AdaGrad / Adam reach the closed-form
solution (0.01028 / 0.02362). RMSProp does not converge at lr 0.1
(0.02108 / 0.10608): over the last 100 epochs its weight error oscillates
between 0.0993 and 0.1061, while GD, AdaGrad and Adam stay at 0.02362
(`results/baseline/convergence_tail.json`). These are the experiment-1
values shown in the README.

### Experiment 2 — `ours` v1 on the mixture (seed 0)

`results/outlier/summary.json`, `results/outlier/optimizer_grid.csv`

| fit | MSE (clean) | weight error |
| --- | --- | --- |
| Oracle (clean population only) | 0.01015 | 0.02921 |
| Naive (all data) | 0.02742 | 0.13124 |
| `ours` v1 (lr 0.01, 1000 epochs) | 0.01188 | 0.14654 |

Optimizer grid on the mixture (lr 0.01, estimation error = post-update MSE on
the clean population, weight error against `w1`):

| optimizer | batch | init | estimation error | weight error |
| --- | --- | --- | --- | --- |
| GD | SGD | random | 0.05609 | 0.30922 |
| GD | SGD | sparse | 0.08416 | 0.36304 |
| GD | SGD | zero | 0.06833 | 0.25095 |
| GD | full | random | 0.03430 | 0.33341 |
| GD | full | sparse | 0.03653 | 0.36008 |
| GD | full | zero | 0.02776 | 0.14056 |
| GD | mini-batch | random | 0.03629 | 0.33462 |
| GD | mini-batch | sparse | 0.03821 | 0.35687 |
| GD | mini-batch | zero | 0.03053 | 0.15082 |
| AdaGrad | SGD | random | 1.62310 | 1.93514 |
| AdaGrad | SGD | sparse | 0.28890 | 1.80541 |
| AdaGrad | SGD | zero | 0.07126 | 0.36445 |
| AdaGrad | full | random | 1.10776 | 1.79883 |
| AdaGrad | full | sparse | 0.24806 | 1.65306 |
| AdaGrad | full | zero | 0.03364 | 0.26553 |
| AdaGrad | mini-batch | random | 1.12972 | 1.80490 |
| AdaGrad | mini-batch | sparse | 0.25603 | 1.67449 |
| AdaGrad | mini-batch | zero | 0.03666 | 0.28144 |
| RMSProp | SGD | random | 0.03473 | 0.20857 |
| RMSProp | SGD | sparse | 0.05687 | 0.24330 |
| RMSProp | SGD | zero | 0.04312 | 0.20295 |
| RMSProp | full | random | 0.03024 | 0.14117 |
| RMSProp | full | sparse | 0.02480 | 0.12132 |
| RMSProp | full | zero | 0.03024 | 0.14117 |
| RMSProp | mini-batch | random | 0.02987 | 0.14563 |
| RMSProp | mini-batch | sparse | 0.02987 | 0.14563 |
| RMSProp | mini-batch | zero | 0.02987 | 0.14563 |
| Adam | SGD | random | 0.08727 | 0.44600 |
| Adam | SGD | sparse | 0.10538 | 0.34384 |
| Adam | SGD | zero | 0.11706 | 0.32225 |
| Adam | full | random | 0.02776 | 0.16410 |
| Adam | full | sparse | 0.02742 | 0.13123 |
| Adam | full | zero | 0.02742 | 0.13124 |
| Adam | mini-batch | random | 0.03307 | 0.18017 |
| Adam | mini-batch | sparse | 0.04819 | 0.19291 |
| Adam | mini-batch | zero | 0.04903 | 0.19597 |

The learning-rate sweep of `ours` v1 (lr 0.01 / 0.1 / 0.5) is
plotted in `results/outlier/lr_sweep.png`; its final values are in E2.

### Concept figure (README figure 1)

`docs/images/concept_values.json`. 1-D toy data, 200 points, of which 20
are outliers; true slope 1 (1.00). Least-squares slope on the clean points 1.01 (1.0085), on all points 0.84 (0.8409).

## E1 — Why `ours` v1 has a worse weight error than Naive

> **Superseded in part by E2.** This analysis used lr = 0.01 only. At
> lr 0.1 / 0.5, `ours` works and the stopping rule is a secondary issue.

**Observation.** `ours` reaches a clean-population MSE of 0.0119 (oracle
0.0102), but its weight error is 0.1465, worse than the naive fit (0.131).

**Diagnostic** (`experiments/diagnose_pruning.py`). There are 92 true outliers
out of 1000 samples.

| Prune at epoch | Removed (outliers) | Kept (outliers) |
| --- | --- | --- |
| 200 | 100 (72) | 900 (20) |
| 400 | 90 (20) | 810 (0) |
| 600 | 81 (0) | 729 (0) |
| 800 | 73 (0) | 656 (0) |

Final weight error:

| num_epochs | trained `ours` | closed form on kept set |
| --- | --- | --- |
| 1000 | 0.1465 | 0.0847 |
| 3000 | 0.3997 | 0.1209 (227 samples left) |

**Findings.**

1. The outlier detection itself works: all outliers are removed by epoch 400.
2. There is no stopping rule. Every cycle removes another 10% of the *current*
   set, so after epoch 400 only clean samples are removed. These are the
   clean samples with the largest residuals, so the kept set becomes biased.
   Closed form on the kept set gives 0.085 against the oracle's 0.029.
3. The last cycle is too short to converge. After the final prune the weights
   are re-initialized randomly and trained for only about 200 Adam steps at
   lr = 0.01. The trained error of 0.1465 against the kept-set closed form of
   0.085 is this gap.
4. Longer training makes it worse (3000 epochs: 227 samples, error 0.40).

## E2 — Learning rate, not the stopping rule, is the main issue

Re-checked E1 with the learning rates from the sweep. `diagnose_pruning.py`
runs lr 0.01 / 0.1 / 0.5 and records the weight error of the model that makes
each pruning decision.

| Prune at epoch | lr 0.01: w_err before prune | outliers removed | lr 0.1: w_err before prune | outliers removed |
| --- | --- | --- | --- | --- |
| 200 | 1.355 | 72 | 0.131 | 89 |
| 400 | 0.385 | 20 | 0.030 | 3 |
| 600 | 0.158 | 0 | 0.032 | 0 |
| 800 | 0.103 | 0 | 0.034 | 0 |
| final (1000) | 0.147 | – | 0.043 | – |

lr = 0.5 matches lr = 0.1 to three decimals (final 0.0433).

**Findings.**

1. At lr 0.1 / 0.5, every 200-epoch cycle converges. The trained final weights
   equal the closed form on the kept set (0.0434 vs 0.0433). The first prune
   is made from the converged naive fit (0.131) and catches 89 of 92
   outliers.
2. At lr 0.01, the cycles never converge (1.355 at epoch 200). The first prune
   is made by an unfit model and catches only 72. The kept set it ends with
   is also worse: the closed form on it gives 0.085, against 0.043 for the
   lr 0.1 kept set of the same size.
3. `ours` works: at lr 0.1 its best point (epoch 400, 0.030) matches the
   oracle (0.029), and the final 0.043 is a third of the naive fit's 0.131.
4. The missing stopping rule is real but secondary. It costs 0.030 → 0.043
   at lr 0.1.
5. So the E1 conclusion ("no stopping rule is the core defect") was mostly an
   artifact of lr = 0.01. The headline `ours` number in `summary.json`
   (lr 0.01, 0.1465) is the worst of the three learning rates.

**Limitations carried into E3.**

- **①** At lr 0.01, a 200-epoch cycle does not converge, so pruning decisions
  are made by an unfit model.
- **②** There is no stopping rule, so clean samples keep being removed after
  all outliers are gone.

**MSE vs weight error** (`experiments/analyze_mse_vs_weight.py` →
`results/outlier/mse_vs_weight_error.json`). For lr 0.01, the error `w - w1`
is almost entirely orthogonal to the all-ones direction (0.1464 of 0.1465;
the along-ones component is −0.0068). Because `X ~ U[0,1)`, the feature
covariance has one large eigenvalue (~1.08, along the ones direction) and
three small ones (~0.08–0.09). The resulting excess MSE, `(w - w_oracle)' S (w - w_oracle)`,
is 0.001722. This exactly equals the observed MSE gap between `ours` and the
oracle (0.011876 − 0.010154). So a near-oracle MSE can hide a large weight
error, and weight error should stay the primary metric.

## E3 — Stopping rule and per-cycle convergence

Tested the two fixes for ① and ② one at a time, then combined them. The
implementation is `ours_v2` in `src/outlier_regression/outlier_removal.py`.
With no stopping rule, ratio pruning and fixed 5 × 200-epoch cycles, it
reproduces `ours` v1 bit-for-bit at the same learning rate
(weights and histories identical at lr 0.01 / 0.1 / 0.5). The default
learning rates of the two functions differ (0.01 vs 0.1).

**Stopping rule (for ②).** Before each prune, compute the MAD scale of the
residuals, `σ_MAD = 1.4826 · median(|r − median(r)|)`. If no sample has
`|r| > k·σ_MAD`, stop pruning. The rule uses residuals only, never the labels
`z`. In fixed-length mode, the remaining epoch budget is spent training the
current weights without re-initialization.

**Per-cycle convergence (for ①).** Each cycle trains until `‖∇‖ < tol`
(capped at 20000 epochs) instead of a fixed 200 epochs.

### Step 1 — stopping rule only (seed 0, fixed 200-epoch cycles)

`experiments/run_ablation.py` → `results/ablation/step1_stop_rule.json`

| lr | k | final w_err | outliers removed per prune | kept (outliers left) |
| --- | --- | --- | --- | --- |
| 0.1 | none | 0.0434 | 89, 3, 0, 0 | 656 (0) |
| 0.1 | 2.5 / 3 / 3.5 | 0.0317 | 89, 3 | 810 (0) |
| 0.1 | 4 / 5 | 0.0301 | 89 | 900 (3) |
| 0.01 | none | 0.1465 | 72, 20, 0, 0 | 656 (0) |
| 0.01 | 3 | 0.0982 | 72, 20 | 810 (0) |
| 0.01 | 5 | 0.1641 | – (stops before any prune) | 1000 (92) |

lr 0.5 matches lr 0.1. With lr 0.1 / 0.5, the stopping rule brings the final
error from 0.043 to 0.030–0.032, the oracle level. At lr 0.01 it helps
(0.147 → 0.098) but does not fix it, as expected: limitation ① remains.

### Step 2 — per-cycle convergence only (seed 0, no stopping rule)

`results/ablation/step2_converge.json`

| lr | tol | final w_err | epochs per cycle | outliers removed per prune |
| --- | --- | --- | --- | --- |
| 0.01 | 1e-5 | 0.0433 | 2072, 636, 443, 203, 524 | 89, 3, 0, 0 |
| 0.1 | 1e-5 | 0.0434 | 233, 171, 188, 181, 193 | 89, 3, 0, 0 |
| 0.5 | 1e-5 | 0.0433 | 197, 197, 188, 172, 206 | 89, 3, 0, 0 |

When every cycle converges, lr 0.01 makes exactly the same pruning decisions
as lr 0.1 and ends at the same error. So limitation ① was entirely a
convergence problem. Tolerances 1e-4 to 1e-6 give the same result; 1e-3
stops slightly early (0.041–0.044). Chose `tol = 1e-5`.

### Step 3 — combined method, held-out evaluation

Combined = per-cycle convergence (`tol = 1e-5`) + stopping rule, with ratio
pruning (10% per cycle). `k = 3` is the conventional 3σ cut. The step-1
table on seed 0 had already shown the same result for k = 2.5–3.5, so this
choice is not sensitive on the development data. Seed 0 was used for all development,
so the method is evaluated on 20 fresh datasets, seeds 1–20 (82–119 outliers
each).

`experiments/run_final.py` → `results/final/`

| method | weight error, mean ± std (min – max) |
| --- | --- |
| Oracle | 0.0195 ± 0.0094 (0.0030 – 0.0350) |
| Naive | 0.2476 ± 0.0965 (0.0987 – 0.4832) |
| `ours` v1, lr 0.01 | 0.4628 ± 0.3508 (0.0658 – 1.1549) |
| `ours` v1, lr 0.1 | 0.0274 ± 0.0125 (0.0085 – 0.0588) |
| **combined, lr 0.01 / 0.1 / 0.5** | **0.0224 ± 0.0102 (0.0058 – 0.0503)** |

(The `ours` v1 rows are the `v1_lr*` columns of
`results/final/per_seed.csv`.)

- The combined method beats Naive on 20/20 datasets, is 91% below Naive on
  average, and is 0.0029 above Oracle on average.
- The learning rate no longer matters: across the 20 datasets, the
  largest difference between learning rates is 9e-5.
- k is not sensitive: k = 2.5 gives identical results, and k = 4 gives
  0.0228–0.0229 (it sometimes stops after one prune with up to 3 outliers
  left).
- On every held-out dataset, the method pruned exactly twice (100 + 90
  samples) and kept 810. Only 1 outlier survived across all 20 datasets.
- Seed 0 (development): 0.0317 at all three learning rates (Oracle 0.0292).
- `ours` v1 at lr 0.01 is far worse on held-out data (0.46) than on
  seed 0 (0.15), confirming that the seed-0 result was not representative.

**Decision.** The ratio-based `ours_v2` became the final method.

**Remaining limitation.** Pruning is still a fixed 10% per cycle. The second
prune removes about 90 samples to catch the last few outliers (on seed 0:
3 outliers, 87 clean). → E4.

## E4 — Threshold-based pruning: negative result

Replace the fixed 10% removal with threshold removal, `|r| > k·σ_MAD`
(`prune_rule="threshold"` in `ours_v2`). This uses the same threshold as the
stopping rule, so pruning ends by itself once no sample exceeds it.
Everything else is as in E3 (per-cycle convergence, `tol = 1e-5`), including
`k = 3`.

`experiments/run_threshold.py` → `results/ablation/step3_threshold.csv`,
`step3_threshold_summary.json`

**Development (seed 0).** Threshold looked better: it kept 896 samples (1
outlier) against ratio's 810, and ended at 0.0305 against 0.0317. It removed
89 + 2 outliers and only 11 + 2 clean samples. The cycle cap (5 / 10 / 20)
never binds. k = 2.5 gave 0.0243, below the oracle (0.0292). With a single
dataset this is likely chance, so it was not used to pick k.

**Held-out (seeds 1–20, lr 0.1).**

| rule | k | weight error | clean kept (of ~899) | outliers left (total) | removed clean, noise + / − |
| --- | --- | --- | --- | --- | --- |
| ratio (E3) | 3 | 0.0224 ± 0.0102 | 810.0 | 1 | 932 / 852 |
| threshold | 2.5 | 0.0389 ± 0.0210 | 804.4 | 1 | 1672 / 223 |
| threshold | 3 | 0.0262 ± 0.0112 | 866.8 | 5 | 612 / 35 |
| threshold | 4 | 0.0211 ± 0.0089 | 898.0 | 24 | 24 / 0 |

Threshold (k = 3) beats ratio on only 7/20 datasets and is worse on average,
even though it keeps 57 more clean samples. The development result did not
generalize.

**Cause: one-sided removal.** "noise + / −" counts the sign of the *true*
noise of the clean samples that were removed. The first prune is decided by
the naive fit, which is pulled toward the outliers. Clean residuals
under that fit are skewed, so the clean samples above the threshold are
almost all on one side (positive noise). Examples:

- seed 12: 78 of 79 removed clean samples had positive noise, and the mean
  noise of the kept clean set shifted to −0.012 (0.0357 vs ratio 0.0168).
- seed 3: 85 of 89 positive (0.0432 vs ratio 0.0216).

Threshold then removes very few samples, so this bias is never undone. Ratio
removes about 90 clean samples in its second prune. That prune is made from
a nearly unbiased fit, so it cuts both tails (e.g. 46 / 39 on seed 3), which
happens to cancel most of the first-prune bias. Smaller k removes more
samples one-sidedly and is worse. k = 4 removes almost no clean samples
(24 / 0) and does slightly better than ratio on average (0.0211), but leaves
24 outliers in the data. Its std is also similar, so the difference is not
clearly meaningful.

**Decision.** Keep ratio pruning.

**Next.** The bias comes from removals made by the contaminated first fit
being permanent. → E5: recompute the inlier set from all samples after each
refit (the idea of the C-step in least trimmed squares). Seeds 1–20 have now
been used to compare pruning rules, so E5 is judged on fresh seeds.

## E5 — Re-admission pruning

`prune_rule="readmit"` in `ours_v2`. After each converged cycle, the inlier
set is recomputed from *all* samples: sample `i` is an inlier if
`|r_i| ≤ 3·σ_MAD`, where `σ_MAD` is estimated from the residuals of the
current inliers. Samples removed by an earlier, biased fit can therefore
return. Training stops when the inlier set no longer changes. The fixed 10%
ratio is no longer used, so `k = 3` is the only pruning parameter. Settings
were fixed before evaluation: `k = 3`, `tol = 1e-5`, at most 5 cycles (same
as E3).

`experiments/run_readmit.py` → `results/ablation/step4_readmit.csv`,
`step4_readmit_summary.json`

**Development (seed 0,** `seed0_trace` **in the summary).** The inlier set
is 900 (3 outliers) after the first cycle, 906 (1 outlier) after the
second, then fixed. Only 3 clean samples are removed (noise + / −: 1 / 2), so the one-sided bias of threshold
pruning is gone. Weight error is 0.0313 (ratio 0.0317, oracle 0.0292).

**Held-out.** Seeds 1–20 were already used in E4, so the primary evaluation
uses fresh seeds 21–40.

| method | seeds 21–40 (primary) | seeds 1–20 (secondary) |
| --- | --- | --- |
| Oracle | 0.0191 ± 0.0080 | 0.0195 ± 0.0094 |
| Naive | 0.2297 ± 0.0807 | 0.2476 ± 0.0965 |
| ratio (E3) | 0.0218 ± 0.0093 | 0.0224 ± 0.0102 |
| readmit | 0.0190 ± 0.0078 | 0.0201 ± 0.0094 |

These are the lr 0.1 values. lr 0.01 / 0.5 are identical to four decimals.

| readmit (seeds 21–40) | value |
| --- | --- |
| clean samples kept (of ~900) | 900.5 on average (ratio: 809.9) |
| removed clean, noise + / − | 23 / 26 (ratio: 1006 / 856) |
| outliers left, total over 20 datasets | 12 (ratio: 3) |

**Paired comparison, readmit − ratio (lr 0.1)**
(`paired_tests_lr0.1` in `step4_readmit_summary.json`).

| seeds | mean diff | readmit better | sign test p | sign-flip p |
| --- | --- | --- | --- | --- |
| 21–40 | −0.0027 | 12 / 20 | 0.50 | 0.15 |
| 1–20 | −0.0023 | 14 / 20 | 0.12 | 0.17 |
| 1–40 | −0.0025 | 26 / 40 | 0.08 | 0.04 |

- Readmit matches the oracle on average (−0.0001 vs oracle on seeds 21–40),
  keeps about 90 more clean samples, and removes clean samples symmetrically.
- The improvement over ratio has the same direction on both seed sets but
  is small. It is not significant on the fresh set alone (p = 0.15). The
  pooled p = 0.04 includes seeds 1–20, which were already used for an
  earlier model choice.
- It leaves more outliers in the data (12 vs 3 over 20 datasets). These are
  outliers whose residuals are within 3σ, i.e. samples with small `x·w1`.
  They are close to the regression surface. Their effect on the fit is
  measured in E8.
- Two runs (seeds 17 and 23) reached the 5-cycle cap before the set stopped
  changing. This is not oscillation: the set changes by 1–2 samples per
  cycle. With a 20-cycle cap both stop at cycle 5 with the same weight error
  (`capped_runs_with_20_cycles` in the summary).

**Decision.** Readmit is a candidate (small, consistent improvement, one
fewer parameter, more data kept), but the evidence is weak. → E6: a
pre-registered test with enough samples.

## E6 — Pre-registered confirmatory test: readmit vs ratio

### Plan (written before the run, unchanged)

- **Hypothesis.** Re-admission pruning (`prune_rule="readmit"`) gives a
  different final weight error than ratio pruning (the current final method).
  The expected direction is lower.
- **Primary metric.** Final weight error `‖ŵ − w1‖₂`, learning rate 0.1.
- **Test.** Paired two-sided sign-flip permutation test on the per-dataset
  differences (readmit − ratio), with 100000 random sign flips
  (`numpy.random.default_rng(0)`). Secondary: Wilcoxon signed-rank test.
- **α.** 0.05. This is the only confirmatory comparison.
- **Data.** 100 fresh datasets, seeds 41–140, never used before.
- **Settings (unchanged).** Both methods: `converge_tol = 1e-5`,
  `stop_k = 3`, at most 5 cycles, `seed = 0` for weight initialization.
  Ratio uses `outlier_ratio = 0.1`.
- **Sample size.** From seeds 21–40: mean difference −0.0027, SD of the
  differences about 0.0081. For 80% power at α = 0.05, n ≈ 70 is needed,
  so n = 100 was chosen.
- **Decision rule.** If p < 0.05 and the mean difference is negative,
  readmit becomes the final method. Otherwise ratio stays.
- **Also reported.** The mean difference with a 95% bootstrap CI (10000
  resamples, `default_rng(0)`), the win count, and Oracle / Naive on the
  same seeds.

### Result

Ran exactly as planned. `experiments/run_confirmatory.py` →
`results/confirmatory/`

| method (seeds 41–140, n = 100, lr 0.1) | weight error |
| --- | --- |
| Oracle | 0.0172 ± 0.0072 |
| Naive | 0.2476 ± 0.0693 |
| ratio | 0.0199 ± 0.0086 |
| readmit | 0.0174 ± 0.0077 |

- readmit − ratio: mean −0.00253, 95% bootstrap CI [−0.00376, −0.00133].
  Readmit is better on 65 / 100 datasets. The SD of the differences is
  0.0062, close to the planning estimate (0.0081).
- Primary, sign-flip permutation: **p = 0.00012**. Secondary, Wilcoxon
  signed-rank: z = −3.72, p = 0.00020.
- **Decision (per the pre-registered rule): adopt readmit as the final
  method.**
- Descriptive, not tested:
  - readmit − Oracle is +0.00018 on average.
  - Readmit keeps 896.8 of ~899.6 clean samples on average (ratio: 812.6).
  - It leaves 43 outliers in total, in 28 of 100 datasets (ratio: 10).
  - 11 / 100 runs reached the 5-cycle cap before the inlier set stopped
    changing.

## E7 — Descriptive values on the E6 seeds

Descriptive values of readmit on seeds 41–140 (n = 100), which were not part
of the pre-registered test. Because these seeds were used for the E6 choice,
the README reports the final method on untouched seeds instead (E8).

`results/confirmatory/summary.json`, `per_seed.csv`,
`extra_summary.json`, `extra_optimizer_grid.csv`
(`experiments/run_confirmatory.py`, `experiments/run_eval_extra.py`)

### Main table

| method | weight error (mean ± std) | min – max |
| --- | --- | --- |
| Oracle | 0.0172 ± 0.0072 | 0.0042 – 0.0355 |
| Naive | 0.2476 ± 0.0693 | 0.0620 – 0.4117 |
| optimizer grid, Adam (`full`, `zero`) | 0.2476 ± 0.0693 | 0.0620 – 0.4117 |
| optimizer grid, best: RMSProp (`SGD`, `zero`) | 0.1866 ± 0.0744 | 0.0323 – 0.3987 |
| `ours` (readmit, lr 0.1) | 0.0174 ± 0.0077 | 0.0022 – 0.0351 |
| ratio (E6) | 0.0199 ± 0.0086 | 0.0033 – 0.0491 |

### Optimizer grid on the evaluation set (lr 0.01, 1000 epochs, mini-batch size 32)

| optimizer | batch | init | weight error (mean ± std) | min – max |
| --- | --- | --- | --- | --- |
| GD | SGD | random | 0.3942 ± 0.1203 | 0.1664 – 0.8776 |
| GD | SGD | sparse | 0.4668 ± 0.1343 | 0.1759 – 0.9486 |
| GD | SGD | zero | 0.2938 ± 0.0996 | 0.0628 – 0.6992 |
| GD | full | random | 0.3798 ± 0.0903 | 0.1846 – 0.5680 |
| GD | full | sparse | 0.4582 ± 0.1058 | 0.2275 – 0.6992 |
| GD | full | zero | 0.2771 ± 0.0696 | 0.0680 – 0.4657 |
| GD | mini-batch | random | 0.3790 ± 0.0908 | 0.2021 – 0.5778 |
| GD | mini-batch | sparse | 0.4572 ± 0.1075 | 0.2336 – 0.7098 |
| GD | mini-batch | zero | 0.2749 ± 0.0721 | 0.0771 – 0.4909 |
| AdaGrad | SGD | random | 1.7193 ± 0.2791 | 0.9938 – 2.2614 |
| AdaGrad | SGD | sparse | 1.8480 ± 0.2847 | 1.1772 – 2.3682 |
| AdaGrad | SGD | zero | 0.6719 ± 0.1971 | 0.1479 – 1.0687 |
| AdaGrad | full | random | 1.6089 ± 0.2834 | 0.8814 – 2.1663 |
| AdaGrad | full | sparse | 1.5079 ± 0.3304 | 0.9268 – 2.0476 |
| AdaGrad | full | zero | 0.5237 ± 0.1628 | 0.0698 – 0.8498 |
| AdaGrad | mini-batch | random | 1.6130 ± 0.2834 | 0.8834 – 2.1685 |
| AdaGrad | mini-batch | sparse | 1.6101 ± 0.2985 | 1.0839 – 2.1203 |
| AdaGrad | mini-batch | zero | 0.5346 ± 0.1644 | 0.0850 – 0.8574 |
| RMSProp | SGD | random | 0.2649 ± 0.1235 | 0.0572 – 0.6228 |
| RMSProp | SGD | sparse | 0.3058 ± 0.1549 | 0.0374 – 0.8161 |
| RMSProp | SGD | zero | 0.1866 ± 0.0744 | 0.0323 – 0.3987 |
| RMSProp | full | random | 0.2479 ± 0.0694 | 0.0719 – 0.4195 |
| RMSProp | full | sparse | 0.2476 ± 0.0703 | 0.0523 – 0.4195 |
| RMSProp | full | zero | 0.2484 ± 0.0696 | 0.0523 – 0.4081 |
| RMSProp | mini-batch | random | 0.2440 ± 0.0781 | 0.0741 – 0.4837 |
| RMSProp | mini-batch | sparse | 0.2440 ± 0.0781 | 0.0741 – 0.4841 |
| RMSProp | mini-batch | zero | 0.2440 ± 0.0781 | 0.0741 – 0.4836 |
| Adam | SGD | random | 0.5021 ± 0.1466 | 0.1970 – 0.8899 |
| Adam | SGD | sparse | 0.4567 ± 0.1778 | 0.1037 – 1.0515 |
| Adam | SGD | zero | 0.2918 ± 0.1027 | 0.0321 – 0.6066 |
| Adam | full | random | 0.2662 ± 0.0611 | 0.1435 – 0.4147 |
| Adam | full | sparse | 0.2476 ± 0.0693 | 0.0619 – 0.4117 |
| Adam | full | zero | 0.2476 ± 0.0693 | 0.0620 – 0.4117 |
| Adam | mini-batch | random | 0.2704 ± 0.0681 | 0.1525 – 0.4737 |
| Adam | mini-batch | sparse | 0.2472 ± 0.0789 | 0.0673 – 0.4958 |
| Adam | mini-batch | zero | 0.2468 ± 0.0794 | 0.0518 – 0.4917 |

The smallest mean weight error over the 36 configurations is
0.1866 (RMSProp, `SGD`, `zero`). Full-batch configurations that
converge reach the naive solution (Adam, `full`, `zero`: 0.2476).

### `ours` by learning rate

| lr | weight error (mean ± std) |
| --- | --- |
| 0.01 | 0.0174 ± 0.0077 |
| 0.1 | 0.0174 ± 0.0077 |
| 0.5 | 0.0174 ± 0.0077 |

The largest per-dataset difference between learning rates is
9.16e-05 (`9.2×10⁻⁵`).

### `ours` vs Oracle per dataset

`ours` beats Oracle on 48 of 100 datasets, although Oracle is better on
average (mean difference 0.00018). The two are least-squares fits on
different sample sets:

- Oracle uses every clean sample.
- `ours` excludes the clean samples with `|r| > 3·σ_MAD` (2.88 on average)
  and keeps the outliers with small residuals (43 in total).

When the excluded clean samples happen to carry large noise, the `ours`
fit is closer to `w1`. With Gaussian noise, using all clean samples is
better on average. Oracle is therefore an upper bound *on average*, not on
every dataset.

The difference is not distinguishable from zero: two-sided sign test
p = 0.76 (48 / 100), mean difference 0.00018 with 95% bootstrap CI
[−0.00031, +0.00070] (10000 resamples, `default_rng(0)`; `ours_minus_oracle`
in `extra_summary.json`). `ours` does not systematically beat Oracle, which
would have pointed to label leakage or a bug.

### Runs that hit the 5-cycle cap

11 of 100 runs reached the 5-cycle cap before the inlier set was confirmed
unchanged. Re-run with a 20-cycle cap (`capped_runs_with_20_cycles` in
`extra_summary.json`):

| seed | stops with 20-cycle cap | inlier set sizes | changes between cycles | weight error, cap 5 | weight error, cap 20 |
| --- | --- | --- | --- | --- | --- |
| 60 | yes (after cycle 5) | 864, 913, 914, 913 | 53, 3, 1 | 0.0169 | 0.0169 |
| 71 | yes (after cycle 5) | 850, 894, 898, 899 | 48, 4, 1 | 0.0183 | 0.0183 |
| 72 | yes (after cycle 5) | 812, 894, 897, 898 | 86, 3, 1 | 0.0161 | 0.0161 |
| 92 | yes (after cycle 5) | 881, 895, 898, 899 | 30, 3, 1 | 0.0351 | 0.0351 |
| 93 | yes (after cycle 5) | 844, 901, 908, 909 | 63, 7, 1 | 0.0059 | 0.0059 |
| 98 | no (oscillates) | 852, 895, 899, 898, 899, … (alternating) | 49, 4, 1, 1, 1, … | 0.0153 | 0.0155 |
| 108 | yes (after cycle 5) | 796, 873, 884, 887 | 89, 11, 3 | 0.0154 | 0.0154 |
| 110 | yes (after cycle 5) | 841, 887, 890, 891 | 56, 3, 1 | 0.0295 | 0.0295 |
| 120 | yes (after cycle 5) | 859, 895, 896, 895 | 42, 3, 1 | 0.0291 | 0.0291 |
| 139 | yes (after cycle 5) | 844, 880, 883, 882 | 38, 3, 1 | 0.0198 | 0.0198 |
| 140 | yes (after cycle 5) | 830, 882, 885, 886 | 58, 3, 1 | 0.0131 | 0.0131 |

- 10 of 11 settle: the set after cycle 5 no longer changes, so the 5-cycle
  result is identical (the cap only skipped the final check).
- `seed=98` does not settle. One sample alternates between in and out, so the
  inlier set oscillates between 898 and 899 samples and the weight error
  between 0.0153 and 0.0155. No rule against oscillation is applied.
- This corrects E5, where "not oscillation" was checked on seeds 17 and 23
  only.

### Outliers kept by `ours`, and cycle length

`outlier_xw1_left`, `outlier_xw1_removed` and `max_epochs_in_one_cycle_lr0.1`
in `extra_summary.json` (lr 0.1):

| | n | `x·w1` |
| --- | --- | --- |
| outliers kept (not removed) | 43 | mean 0.144, max 0.263 |
| outliers removed | 9993 | mean 0.989, min 0.074 |

Kept outliers have small `x·w1`, so their residual under the clean fit
(about `2·x·w1`) is within `3·σ_MAD`. They lie close to the regression
surface. Only 1.8% of the removed outliers have `x·w1` below the largest
kept value (0.263) (`fraction_below_max_left`).

The longest cycle needed 651 epochs, so the per-cycle cap of 20000 epochs
was never reached.

### Removal of the first-fit bias

`excluded_clean_noise_sign_after_first_cycle` and
`excluded_clean_noise_sign_final` in `extra_summary.json` (lr 0.1, summed over
the 100 datasets). The counts are clean samples excluded from the inlier
set, split by the sign of their true noise:

| | noise > 0 | noise < 0 |
| --- | --- | --- |
| after the first cycle | 3398 | 56 |
| final inlier set | 128 | 160 |

The first selection is made from the contaminated fit and is one-sided, as
in E4. After re-selection, the exclusions are roughly balanced. The final
total of 288 clean samples is 2.88 per dataset.

### Derived values

| value | computation |
| --- | --- |
| 14.4× | Naive / Oracle mean weight error: 0.2476 / 0.0172 = 14.41 |
| 10.9× | best optimizer-grid configuration / Oracle: 0.1866 / 0.0172 = 10.86 (RMSProp, `SGD`, `zero`) |
| 93.0% | reduction vs Naive: 1 − 0.0174 / 0.2476 = 92.99% |
| 0.00018 | mean of `ours` − Oracle: 0.00018 |
| 48 / 100 | datasets where `ours` < Oracle: 48 (`ours_better_than_oracle` in `extra_summary.json`) |
| 2.88 | clean samples excluded by `ours`, mean: 2.88 (`ours_clean_removed_mean`) |
| 100 / 100 | datasets where `ours` < Naive: 100 |
| 896.8 of 899.6 (99.7%) | clean samples kept by `ours`: 896.76 of 899.64 (99.68%) |
| 28, 43 | datasets with outliers left, total outliers left: 28, 43 |
| 11 | runs that hit the 5-cycle cap: 11 |
| 80–125 | outliers per dataset: 80–125 |
| 0.017 | median weight error of Oracle and readmit: 0.0171, 0.0166 |
| 0.243 | median weight error of Naive: 0.2434 |

## E8 — Final evaluation on untouched seeds

### Plan (written before the run)

Seeds 41–140 were used in E6 to choose between two candidates, so numbers
measured on them are not an unbiased estimate of the chosen method. E8
evaluates the final method once on 100 new datasets that were never used
for any decision.

- **Data.** Seeds 141–240 (100 datasets), not used before.
- **Method.** `ours` = `ours_v2(prune_rule="readmit", stop_k=3,
  converge_tol=1e-5, num_cycles=5, learning_rate=0.1, seed=0)`, exactly as
  adopted in E6. Also run at lr 0.01 and 0.5 to report the learning-rate
  dependence. No setting is changed after this plan.
- **Comparisons.** Oracle and Naive (closed form, `pinv`). The optimizer
  grid of experiment 2 (36 configurations, lr 0.01, 1000 epochs).
- **Reported.** Weight error mean ± std (population std) and min–max for
  each method. `ours` vs Naive: count of datasets where `ours` is lower.
  `ours` vs Oracle: mean difference, 95% bootstrap CI (10000 resamples,
  `default_rng(0)`), and a two-sided sign-flip test (100000 flips,
  `default_rng(0)`), the same test as E6.
- **Descriptive diagnostics.** Clean samples kept, outliers left, and `x·w1`
  of kept vs removed outliers. The effect of the kept outliers, measured by
  refitting the final inlier set without them. The final `σ_MAD` against
  the true noise standard deviation (0.1). Noise signs of the excluded
  clean samples after the first cycle and at the end. Runs that reach the
  5-cycle cap, re-run with a 20-cycle cap. The longest cycle in epochs.
- **No decision** depends on E8. It only reports the final method.

### Result

Ran as planned. `experiments/run_evaluation.py` → `results/evaluation/`
(`summary.json`, `per_seed.csv`, `optimizer_grid.csv`,
`weight_error_by_seed.png`). Outliers per dataset: 81–123.
These are the values reported in the README.

**Main table (README table 3).**

| method | weight error (mean ± std) | min – max |
| --- | --- | --- |
| Oracle | 0.0188 ± 0.0082 | 0.0046 – 0.0415 |
| Naive | 0.2478 ± 0.0674 | 0.1218 – 0.4573 |
| optimizer grid, Adam (`full`, `zero`) | 0.2478 ± 0.0674 | 0.1218 – 0.4573 |
| `ours` | 0.0188 ± 0.0078 | 0.0053 – 0.0414 |

Medians shown in README figure 5: Oracle 0.0181 (0.018), Naive
0.2407 (0.241), `ours` 0.0179 (0.018).

**`ours` vs Naive and Oracle.**

- `ours` < Naive on 100 / 100 datasets. Reduction of the mean:
  1 − 0.0188 / 0.2478 = 92.4%. Naive / Oracle = 0.2478 / 0.0188 = 13.2×.
- `ours` − Oracle: mean +0.00003 (+0.00003), 95% bootstrap CI
  [−0.00044, +0.00049], two-sided sign-flip p = 0.895.
  `ours` is lower on 49 / 100. Not significant. This does not show that
  the two are equal; it shows no detectable difference at n = 100.

**Learning rate.**

| lr | weight error (mean ± std) |
| --- | --- |
| 0.01 | 0.0188 ± 0.0079 |
| 0.1 | 0.0188 ± 0.0078 |
| 0.5 | 0.0188 ± 0.0079 |

Largest per-dataset difference between learning rates: 0.0015
(0.0015).

**Optimizer grid (lr 0.01, 1000 epochs, mini-batch size 32).** Every optimizer
minimizes the same convex loss (MSE on all data), whose unique minimizer is
the Naive solution. Adam (`full`, `zero`) matches Naive within
0.00055 (0.00055) on every dataset. A configuration whose value differs
from Naive has not reached the minimum within 1000 epochs. The smallest mean
over the 36 configurations is 0.1882 (RMSProp, `SGD`, `zero`).

| optimizer | batch | init | weight error (mean ± std) | min – max |
| --- | --- | --- | --- | --- |
| GD | SGD | random | 0.3962 ± 0.1065 | 0.1590 – 0.6275 |
| GD | SGD | sparse | 0.4727 ± 0.1083 | 0.1551 – 0.7582 |
| GD | SGD | zero | 0.2949 ± 0.0977 | 0.0740 – 0.5753 |
| GD | full | random | 0.3829 ± 0.0909 | 0.1444 – 0.5899 |
| GD | full | sparse | 0.4622 ± 0.0962 | 0.2115 – 0.6692 |
| GD | full | zero | 0.2776 ± 0.0633 | 0.1572 – 0.4595 |
| GD | mini-batch | random | 0.3839 ± 0.0903 | 0.1432 – 0.5931 |
| GD | mini-batch | sparse | 0.4629 ± 0.0960 | 0.2144 – 0.6571 |
| GD | mini-batch | zero | 0.2776 ± 0.0635 | 0.1598 – 0.4679 |
| AdaGrad | SGD | random | 1.7294 ± 0.2893 | 1.0630 – 2.3522 |
| AdaGrad | SGD | sparse | 1.8590 ± 0.2927 | 1.1658 – 2.4382 |
| AdaGrad | SGD | zero | 0.6836 ± 0.1888 | 0.2574 – 1.1297 |
| AdaGrad | full | random | 1.6220 ± 0.2943 | 0.9086 – 2.2565 |
| AdaGrad | full | sparse | 1.5147 ± 0.3308 | 0.9070 – 2.0713 |
| AdaGrad | full | zero | 0.5282 ± 0.1532 | 0.2014 – 0.8976 |
| AdaGrad | mini-batch | random | 1.6262 ± 0.2936 | 0.9231 – 2.2623 |
| AdaGrad | mini-batch | sparse | 1.6187 ± 0.2976 | 1.0509 – 2.1189 |
| AdaGrad | mini-batch | zero | 0.5395 ± 0.1544 | 0.2023 – 0.9147 |
| RMSProp | SGD | random | 0.2738 ± 0.1170 | 0.0619 – 0.6010 |
| RMSProp | SGD | sparse | 0.3115 ± 0.1337 | 0.0426 – 0.7269 |
| RMSProp | SGD | zero | 0.1882 ± 0.0738 | 0.0404 – 0.4124 |
| RMSProp | full | random | 0.2480 ± 0.0691 | 0.1289 – 0.4640 |
| RMSProp | full | sparse | 0.2471 ± 0.0664 | 0.1289 – 0.4507 |
| RMSProp | full | zero | 0.2481 ± 0.0684 | 0.1152 – 0.4507 |
| RMSProp | mini-batch | random | 0.2478 ± 0.0705 | 0.0761 – 0.4750 |
| RMSProp | mini-batch | sparse | 0.2478 ± 0.0705 | 0.0761 – 0.4746 |
| RMSProp | mini-batch | zero | 0.2478 ± 0.0705 | 0.0761 – 0.4742 |
| Adam | SGD | random | 0.5061 ± 0.1420 | 0.2184 – 0.8375 |
| Adam | SGD | sparse | 0.4671 ± 0.1500 | 0.0963 – 0.9177 |
| Adam | SGD | zero | 0.2946 ± 0.0986 | 0.0666 – 0.6233 |
| Adam | full | random | 0.2664 ± 0.0615 | 0.1492 – 0.4647 |
| Adam | full | sparse | 0.2478 ± 0.0674 | 0.1217 – 0.4573 |
| Adam | full | zero | 0.2478 ± 0.0674 | 0.1218 – 0.4573 |
| Adam | mini-batch | random | 0.2719 ± 0.0629 | 0.1478 – 0.4746 |
| Adam | mini-batch | sparse | 0.2478 ± 0.0723 | 0.0959 – 0.4562 |
| Adam | mini-batch | zero | 0.2477 ± 0.0721 | 0.0897 – 0.4567 |

**Samples kept.** Clean samples kept: 896.25 of 898.8 on average
(99.72%, 99.7%). Clean samples excluded: 2.55 per dataset.

**Bias of the first selection.** Clean samples excluded from the inlier
set, by the sign of their true noise, summed over the 100 datasets:

| | noise > 0 | noise < 0 |
| --- | --- | --- |
| after the first cycle | 3482 | 33 |
| final inlier set | 129 | 126 |

**`σ_MAD`.** Final-cycle value: mean 0.0994, range 0.0888–0.1129; true
noise standard deviation 0.1.

**Outliers kept.** 35 outliers in 24 datasets. `x·w1` of kept outliers:
mean 0.155, max 0.225; of the 10085 removed outliers: mean 1.004, min
0.089. Refitting the final inlier set without the kept outliers changes the
weight error by at most 0.0030 (mean 0.00010).

**5-cycle cap.** 11 / 100 runs reached the cap before the inlier set was
confirmed unchanged. Re-run with a 20-cycle cap:

| seed | cycles until the set is fixed | inlier set sizes | changes between cycles | weight error, cap 5 | weight error, cap 20 |
| --- | --- | --- | --- | --- | --- |
| 143 | 5 | 868, 892, 895, 897 | 34, 3, 2 | 0.0189 | 0.0189 |
| 169 | 6 | 863, 900, 902, 901, 900 | 43, 2, 1, 1 | 0.0108 | 0.0094 |
| 190 | 5 | 863, 910, 913, 914 | 57, 3, 1 | 0.0128 | 0.0128 |
| 193 | 5 | 887, 892, 891, 890 | 9, 1, 1 | 0.0310 | 0.0310 |
| 197 | 5 | 829, 883, 887, 888 | 60, 4, 1 | 0.0186 | 0.0186 |
| 204 | 5 | 881, 902, 903, 902 | 29, 1, 1 | 0.0148 | 0.0148 |
| 210 | 5 | 861, 894, 899, 900 | 39, 5, 1 | 0.0182 | 0.0182 |
| 227 | 5 | 845, 901, 903, 904 | 58, 2, 1 | 0.0239 | 0.0239 |
| 232 | 7 | 829, 890, 895, 897, 896, 897 | 67, 5, 2, 1, 1 | 0.0350 | 0.0355 |
| 233 | 5 | 899, 904, 905, 906 | 13, 1, 1 | 0.0188 | 0.0188 |
| 239 | 5 | 817, 897, 904, 903 | 80, 7, 1 | 0.0244 | 0.0244 |

9 of 11 give the same result: the set after cycle 5 no longer changes (the
cap only skipped the final check). `seed=169` and `seed=232` are fixed only
after the 6th and 7th cycle, and their weight error changes (0.0108 → 0.0094,
0.0350 → 0.0355). No run oscillated.

**Cycle length.** The longest cycle needed 242 epochs (cap 20000).

**Settings of the final method (README table 1).** `k = 3` (conventional 3σ
cut), convergence tolerance 1e-5 (E3: 1e-4 to 1e-6 give the same result on
seed 0), at most 5 cycles (as in `ours` v1, 5 × 200 epochs), at most 20000
epochs per cycle, lr 0.1, initial weights `N(0, 1)` with seed 0 in every
cycle, Adam β1 = 0.9, β2 = 0.999, ε = 1e-8.

---

## Open items

- **Outlier ratio and type.** All experiments use 10% outliers following
  `w2 = -w1`. Higher ratios, outliers close to `w1`, leverage points and
  heavy-tailed noise are untested. These are listed as future work in the
  README.
- **Baselines.** Robust regression baselines (Huber loss, RANSAC) have not
  been compared.
- **Cycle cap and oscillation.** The 5-cycle cap is reached in 11 / 100 runs
  on both seed sets. On seeds 41–140 (E7), 10 settle with an identical result
  and `seed=98` oscillates. On seeds 141–240 (E8), 9 settle with an identical
  result, and 2 (`seed=169`, `seed=232`) need 6 and 7 cycles, which changes
  their weight error. There is no rule against oscillation, and a larger cap
  was not part of the tested configuration.

---

## Environment

Every result in `results/` was produced with Python 3.12.10, NumPy 2.5.3,
pandas 3.0.6 and matplotlib 3.11.2, on Windows 11, CPU only.
`python experiments/run_all.py` regenerates all of them from fixed seeds.
The clean-only `pinv` fit (Oracle) is mildly sensitive to the NumPy / LAPACK
version, so other versions can differ in the last digits.

Measured, not saved: one fixed-cycle run (5 × 200 epochs, the `ours` v1
algorithm) takes about 26 ms, about 26 µs per epoch. `run_eval_extra.py` takes about
2 minutes, mostly for the 36-configuration optimizer grid on 100 datasets.
A GPU was not used: the per-epoch work (a few 1000×4 matrix products) is
smaller than GPU launch overhead, and epochs are sequential.
