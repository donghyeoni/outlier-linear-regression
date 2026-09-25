# Experiment Log

Every run of the project, in the order it was made: the plan, the setting,
the full result tables, and the decision taken. The [README](../README.md)
summarises the final results; all of its numbers appear here.

## Conventions

- **Data.** `N = 1000`, `D = 4`, `X ~ U[0,1)^{1000×4}`, noise `ε ~ 0.1·N(0, 1)`.
  True weights are drawn from `U[0,1)^4` for each dataset.
  - Clean data (experiment 1): `y = X·w_true + ε`.
  - Mixture data (experiment 2): each sample follows `w1` (clean) with
    probability 0.9 and `w2 = -w1` (outlier) otherwise; label `z` = 1 / 2.
- **Dataset seed.** One seed fixes `X`, the true weights, the noise and the
  labels (`src/outlier_regression/data.py`). For the same seed, the clean and
  the mixture dataset share `X`, the true weights (`w_true = w1`) and the
  noise.
- **Training seed.** Every training run seeds the global NumPy RNG with 0
  before its first weight initialisation.
- **Iteration.** One parameter update. Full batch uses all samples,
  mini-batch 32 samples, SGD one sample.
- **Initialisations.** `random`: `N(0, 1)`; `zero`: all zeros; `sparse`:
  `N(0, 1)` with each entry set to 0 with probability 0.8.
- **Metric.** Weight error `‖ŵ − w_ref‖₂`, with `w_ref = w_true`
  (experiment 1) or `w1` (experiment 2). Residual `r = Xŵ − y`.
- **Closed form.** `pinv(X)·y`. Oracle: closed form on the clean samples
  (`z = 1`). Naive: closed form on all samples.
- **Precision.** Measured values are rounded to 4 decimals, and every
  comparison in the text is made on the rounded values. A p-value that
  rounds to 0.0000 is written `< 0.0001`. Per-dataset counts such as
  "A < B on 18 / 20 datasets" compare unrounded values. Standard deviations
  are population SDs (`ddof = 0`). Settings such as `tol = 1e-5` are exact.
- **Result tables.** Every result table (titled with its id and file) is
  written by the script named in its section to a `tables.md` file and
  copied here unchanged.

## Seeds

| use | dataset seeds | number |
| --- | --- | --- |
| experiment 1 (clean data) | 0–19 | 20 |
| development of `ours` (mixture data) | 0–19 | 20 |
| final evaluation (mixture data) | 100–199 | 100 |

The final-evaluation seeds are used only in T1, after every setting has been
fixed.

## Plan (written before any run)

1. **E1.** Optimizer grid on clean data: 4 optimizers (GD, AdaGrad, RMSProp,
   Adam) × 3 batch types × 3 initialisations × learning rate {0.01, 0.1},
   1000 iterations, against the closed form. Purpose: check the optimizer
   implementations.
2. **D0.** The same grid, Oracle and Naive on the development mixture data.
   Purpose: measure the effect of the outliers and whether any optimizer
   setting avoids it.
3. **D1.** `ours_v1`, the starting method: 5 cycles × 200 iterations of
   full-batch Adam (lr 0.01), re-initialised each cycle, removing the 10% of
   kept samples with the largest `|r|` after cycles 1–4. Diagnose where its
   error comes from.
4. **D2–D5.** Change one component at a time, as the D1 diagnosis suggests,
   and choose each setting on the development seeds by the rules below.
5. **T1.** Freeze the final method `ours`, write the test plan, then evaluate
   once on seeds 100–199.

**Selection rules for development** (fixed now):

- A setting is chosen by the mean weight error over the 20 development
  datasets. Settings whose means differ by less than 0.0001 are tied.
  *(Changed after D2 to "equal at 4 decimals"; see "Changes made after a
  run", item 2.)*
- Ties are broken as stated in each section before its run.

## E1 — Optimizer grid on clean data

**Setting.** `experiments/exp1_clean.py` → `results/exp1/`. Seeds 0–19, clean
data. 72 combinations: 4 optimizers × 3 batch types × 3 initialisations ×
learning rate {0.01, 0.1}, 1000 iterations each. Adam: `β1 = 0.9`,
`β2 = 0.999`, `ε = 1e-8`; RMSProp decay 0.9, `ε = 1e-8`; AdaGrad `ε = 1e-8`.
`ŵ_cf` is the closed form on the same dataset.

**E1-a closed form (20 datasets)** (`results/exp1/tables.md`)

| method | weight error (mean ± SD) |
| --- | --- |
| closed form | 0.0168 ± 0.0088 |

**E1-b combinations reaching the closed form (max ‖ŵ − ŵ_cf‖ over the 20 datasets is 0.0000 at 4 decimals), of 9 batch × init combinations** (`results/exp1/tables.md`)

| optimizer | lr 0.01 | lr 0.1 |
| --- | --- | --- |
| GD | 0 / 9 | 3 / 9 |
| AdaGrad | 0 / 9 | 0 / 9 |
| RMSProp | 0 / 9 | 0 / 9 |
| Adam | 1 / 9 | 2 / 9 |

**E1-c full batch, zero init: mean weight error** (`results/exp1/tables.md`)

| optimizer | lr 0.01 | lr 0.1 |
| --- | --- | --- |
| GD | 0.0838 | 0.0168 |
| AdaGrad | 0.4319 | 0.0171 |
| RMSProp | 0.0208 | 0.1006 |
| Adam | 0.0169 | 0.0168 |

Figure `results/exp1/curves.png`: mean weight error per iteration over the 20
datasets, full batch, zero initialisation.

**E1-d all 72 combinations** (`results/exp1/tables.md`)

| optimizer | batch | init | lr | weight error (mean ± SD) | max ‖ŵ − ŵ_cf‖ | reaches closed form |
| --- | --- | --- | --- | --- | --- | --- |
| GD | full | random | 0.01 | 0.2797 ± 0.0507 | 0.3767 | no |
| GD | full | zero | 0.01 | 0.0838 ± 0.0298 | 0.1328 | no |
| GD | full | sparse | 0.01 | 0.3682 ± 0.0594 | 0.4587 | no |
| GD | mini-batch | random | 0.01 | 0.2798 ± 0.0496 | 0.3666 | no |
| GD | mini-batch | zero | 0.01 | 0.0843 ± 0.0296 | 0.1353 | no |
| GD | mini-batch | sparse | 0.01 | 0.3697 ± 0.0602 | 0.4660 | no |
| GD | SGD | random | 0.01 | 0.2853 ± 0.0630 | 0.4169 | no |
| GD | SGD | zero | 0.01 | 0.0865 ± 0.0324 | 0.1594 | no |
| GD | SGD | sparse | 0.01 | 0.3716 ± 0.0665 | 0.4813 | no |
| AdaGrad | full | random | 0.01 | 1.6477 ± 0.2869 | 2.1496 | no |
| AdaGrad | full | zero | 0.01 | 0.4319 ± 0.1663 | 0.6699 | no |
| AdaGrad | full | sparse | 0.01 | 1.4022 ± 0.3555 | 1.8979 | no |
| AdaGrad | mini-batch | random | 0.01 | 1.6519 ± 0.2887 | 2.1541 | no |
| AdaGrad | mini-batch | zero | 0.01 | 0.4345 ± 0.1674 | 0.6737 | no |
| AdaGrad | mini-batch | sparse | 0.01 | 1.4637 ± 0.3392 | 1.9437 | no |
| AdaGrad | SGD | random | 0.01 | 1.7535 ± 0.2972 | 2.2627 | no |
| AdaGrad | SGD | zero | 0.01 | 0.4891 ± 0.1882 | 0.7590 | no |
| AdaGrad | SGD | sparse | 0.01 | 1.7551 ± 0.2850 | 2.1794 | no |
| RMSProp | full | random | 0.01 | 0.0199 ± 0.0075 | 0.0100 | no |
| RMSProp | full | zero | 0.01 | 0.0208 ± 0.0074 | 0.0100 | no |
| RMSProp | full | sparse | 0.01 | 0.0198 ± 0.0081 | 0.0100 | no |
| RMSProp | mini-batch | random | 0.01 | 0.0299 ± 0.0125 | 0.0537 | no |
| RMSProp | mini-batch | zero | 0.01 | 0.0299 ± 0.0125 | 0.0537 | no |
| RMSProp | mini-batch | sparse | 0.01 | 0.0299 ± 0.0125 | 0.0537 | no |
| RMSProp | SGD | random | 0.01 | 0.0569 ± 0.0226 | 0.1154 | no |
| RMSProp | SGD | zero | 0.01 | 0.0601 ± 0.0197 | 0.0961 | no |
| RMSProp | SGD | sparse | 0.01 | 0.0529 ± 0.0214 | 0.1032 | no |
| Adam | full | random | 0.01 | 0.0564 ± 0.0336 | 0.1327 | no |
| Adam | full | zero | 0.01 | 0.0169 ± 0.0087 | 0.0024 | no |
| Adam | full | sparse | 0.01 | 0.0168 ± 0.0088 | 0.0000 | yes |
| Adam | mini-batch | random | 0.01 | 0.0620 ± 0.0367 | 0.1520 | no |
| Adam | mini-batch | zero | 0.01 | 0.0204 ± 0.0077 | 0.0179 | no |
| Adam | mini-batch | sparse | 0.01 | 0.0228 ± 0.0085 | 0.0315 | no |
| Adam | SGD | random | 0.01 | 0.2047 ± 0.1099 | 0.4306 | no |
| Adam | SGD | zero | 0.01 | 0.0479 ± 0.0210 | 0.0908 | no |
| Adam | SGD | sparse | 0.01 | 0.0756 ± 0.0397 | 0.1708 | no |
| GD | full | random | 0.1 | 0.0168 ± 0.0088 | 0.0000 | yes |
| GD | full | zero | 0.1 | 0.0168 ± 0.0088 | 0.0000 | yes |
| GD | full | sparse | 0.1 | 0.0168 ± 0.0088 | 0.0000 | yes |
| GD | mini-batch | random | 0.1 | 0.0206 ± 0.0073 | 0.0202 | no |
| GD | mini-batch | zero | 0.1 | 0.0206 ± 0.0073 | 0.0202 | no |
| GD | mini-batch | sparse | 0.1 | 0.0206 ± 0.0073 | 0.0202 | no |
| GD | SGD | random | 0.1 | 0.0682 ± 0.0305 | 0.1338 | no |
| GD | SGD | zero | 0.1 | 0.0723 ± 0.0248 | 0.1230 | no |
| GD | SGD | sparse | 0.1 | 0.0652 ± 0.0271 | 0.1143 | no |
| AdaGrad | full | random | 0.1 | 0.1119 ± 0.0693 | 0.2421 | no |
| AdaGrad | full | zero | 0.1 | 0.0171 ± 0.0086 | 0.0066 | no |
| AdaGrad | full | sparse | 0.1 | 0.0170 ± 0.0085 | 0.0030 | no |
| AdaGrad | mini-batch | random | 0.1 | 0.1240 ± 0.0777 | 0.2900 | no |
| AdaGrad | mini-batch | zero | 0.1 | 0.0194 ± 0.0077 | 0.0177 | no |
| AdaGrad | mini-batch | sparse | 0.1 | 0.0203 ± 0.0074 | 0.0171 | no |
| AdaGrad | SGD | random | 0.1 | 0.3967 ± 0.1931 | 0.7495 | no |
| AdaGrad | SGD | zero | 0.1 | 0.0367 ± 0.0128 | 0.0520 | no |
| AdaGrad | SGD | sparse | 0.1 | 0.2586 ± 0.1448 | 0.5528 | no |
| RMSProp | full | random | 0.1 | 0.1010 ± 0.0038 | 0.1000 | no |
| RMSProp | full | zero | 0.1 | 0.1006 ± 0.0034 | 0.1000 | no |
| RMSProp | full | sparse | 0.1 | 0.1023 ± 0.0033 | 0.1000 | no |
| RMSProp | mini-batch | random | 0.1 | 0.1054 ± 0.0474 | 0.1952 | no |
| RMSProp | mini-batch | zero | 0.1 | 0.1054 ± 0.0474 | 0.1952 | no |
| RMSProp | mini-batch | sparse | 0.1 | 0.1054 ± 0.0474 | 0.1952 | no |
| RMSProp | SGD | random | 0.1 | 0.2242 ± 0.0619 | 0.3170 | no |
| RMSProp | SGD | zero | 0.1 | 0.2056 ± 0.0692 | 0.3308 | no |
| RMSProp | SGD | sparse | 0.1 | 0.2475 ± 0.0645 | 0.3548 | no |
| Adam | full | random | 0.1 | 0.0168 ± 0.0088 | 0.0000 | yes |
| Adam | full | zero | 0.1 | 0.0168 ± 0.0088 | 0.0000 | yes |
| Adam | full | sparse | 0.1 | 0.0168 ± 0.0088 | 0.0001 | no |
| Adam | mini-batch | random | 0.1 | 0.0387 ± 0.0165 | 0.0732 | no |
| Adam | mini-batch | zero | 0.1 | 0.0510 ± 0.0221 | 0.1043 | no |
| Adam | mini-batch | sparse | 0.1 | 0.0568 ± 0.0221 | 0.1019 | no |
| Adam | SGD | random | 0.1 | 0.1879 ± 0.0964 | 0.5042 | no |
| Adam | SGD | zero | 0.1 | 0.2040 ± 0.0865 | 0.4009 | no |
| Adam | SGD | sparse | 0.1 | 0.1777 ± 0.1113 | 0.5935 | no |

**Observations.**

- The 6 combinations that reach the closed form are all full batch: GD at
  lr 0.1 (all three initialisations), Adam at lr 0.1 (`random`, `zero`) and
  Adam at lr 0.01 (`sparse`).
- No mini-batch or SGD combination reaches the closed form within 1000
  iterations.
- RMSProp with full batch ends at a maximum distance of 0.0100 (lr 0.01) and
  0.1000 (lr 0.1) from the closed form for every initialisation.

**Decision.** The optimizer implementations are used unchanged in D0 and T1.

## D0 — Outliers: Oracle, Naive and the optimizer grid

**Setting.** `experiments/dev0_baselines.py` → `results/dev0/`. Development
seeds 0–19, mixture data. The grid of E1 is trained on all samples;
`ŵ_naive` is the Naive solution of the same dataset. Weight errors are
against `w1`.

**D0-a Oracle and Naive** (`results/dev0/tables.md`)

| method | weight error (mean ± SD) | min–max |
| --- | --- | --- |
| Oracle | 0.0194 ± 0.0093 | 0.0030–0.0350 |
| Naive | 0.2399 ± 0.0993 | 0.0987–0.4832 |

**D0-b optimizer grid overview** (`results/dev0/tables.md`)

| quantity | value |
| --- | --- |
| outliers per dataset (min–max) | 82–119 |
| combinations reaching Naive (max ‖ŵ − ŵ_naive‖ is 0.0000 at 4 decimals) | 6 / 72 |
| smallest mean weight error over the grid | 0.1905 (RMSProp, SGD, zero, lr 0.01) |
| datasets where that combination is below Naive | 15 / 20 |

**D0-c optimizer grid, all 72 combinations** (`results/dev0/tables.md`)

| optimizer | batch | init | lr | weight error (mean ± SD) | min–max | max ‖ŵ − ŵ_naive‖ |
| --- | --- | --- | --- | --- | --- | --- |
| GD | full | random | 0.01 | 0.3742 ± 0.0757 | 0.2657–0.5219 | 0.3550 |
| GD | full | zero | 0.01 | 0.2579 ± 0.0985 | 0.1049–0.4932 | 0.1259 |
| GD | full | sparse | 0.01 | 0.4395 ± 0.0874 | 0.2217–0.5569 | 0.4820 |
| GD | mini-batch | random | 0.01 | 0.3706 ± 0.0789 | 0.2545–0.5430 | 0.3660 |
| GD | mini-batch | zero | 0.01 | 0.2535 ± 0.0977 | 0.1035–0.4812 | 0.1446 |
| GD | mini-batch | sparse | 0.01 | 0.4375 ± 0.0900 | 0.2049–0.5529 | 0.5079 |
| GD | SGD | random | 0.01 | 0.4220 ± 0.1172 | 0.2771–0.6343 | 0.4463 |
| GD | SGD | zero | 0.01 | 0.2915 ± 0.1314 | 0.0740–0.6158 | 0.2565 |
| GD | SGD | sparse | 0.01 | 0.4710 ± 0.1169 | 0.2132–0.6647 | 0.5221 |
| AdaGrad | full | random | 0.01 | 1.6441 ± 0.2877 | 1.0451–2.1470 | 2.1295 |
| AdaGrad | full | zero | 0.01 | 0.4606 ± 0.1701 | 0.1415–0.7128 | 0.5968 |
| AdaGrad | full | sparse | 0.01 | 1.5463 ± 0.3321 | 0.9777–2.0299 | 1.9691 |
| AdaGrad | mini-batch | random | 0.01 | 1.6482 ± 0.2896 | 1.0439–2.1513 | 2.1341 |
| AdaGrad | mini-batch | zero | 0.01 | 0.4720 ± 0.1722 | 0.1504–0.7264 | 0.6056 |
| AdaGrad | mini-batch | sparse | 0.01 | 1.6307 ± 0.2921 | 1.0765–2.0661 | 2.0035 |
| AdaGrad | SGD | random | 0.01 | 1.7605 ± 0.2883 | 1.1455–2.2659 | 2.2616 |
| AdaGrad | SGD | zero | 0.01 | 0.5983 ± 0.2231 | 0.1819–0.9461 | 0.7008 |
| AdaGrad | SGD | sparse | 0.01 | 1.8388 ± 0.2616 | 1.2928–2.2222 | 2.2950 |
| RMSProp | full | random | 0.01 | 0.2405 ± 0.1029 | 0.0896–0.4887 | 0.0100 |
| RMSProp | full | zero | 0.01 | 0.2381 ± 0.0994 | 0.0896–0.4778 | 0.0100 |
| RMSProp | full | sparse | 0.01 | 0.2422 ± 0.1007 | 0.1008–0.4887 | 0.0100 |
| RMSProp | mini-batch | random | 0.01 | 0.2312 ± 0.1026 | 0.0877–0.4812 | 0.1036 |
| RMSProp | mini-batch | zero | 0.01 | 0.2313 ± 0.1027 | 0.0877–0.4812 | 0.1036 |
| RMSProp | mini-batch | sparse | 0.01 | 0.2313 ± 0.1026 | 0.0877–0.4812 | 0.1036 |
| RMSProp | SGD | random | 0.01 | 0.2761 ± 0.1288 | 0.0726–0.5391 | 0.4821 |
| RMSProp | SGD | zero | 0.01 | 0.1905 ± 0.0940 | 0.0523–0.3982 | 0.2315 |
| RMSProp | SGD | sparse | 0.01 | 0.2943 ± 0.1297 | 0.0960–0.5603 | 0.5780 |
| Adam | full | random | 0.01 | 0.2601 ± 0.0853 | 0.1400–0.4755 | 0.1677 |
| Adam | full | zero | 0.01 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0001 |
| Adam | full | sparse | 0.01 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0003 |
| Adam | mini-batch | random | 0.01 | 0.2555 ± 0.0809 | 0.1406–0.4550 | 0.1724 |
| Adam | mini-batch | zero | 0.01 | 0.2338 ± 0.0959 | 0.0616–0.4684 | 0.0962 |
| Adam | mini-batch | sparse | 0.01 | 0.2346 ± 0.0960 | 0.0639–0.4686 | 0.1026 |
| Adam | SGD | random | 0.01 | 0.5188 ± 0.1252 | 0.2682–0.7581 | 0.6318 |
| Adam | SGD | zero | 0.01 | 0.2829 ± 0.1265 | 0.0474–0.5560 | 0.2324 |
| Adam | SGD | sparse | 0.01 | 0.4462 ± 0.1456 | 0.1847–0.7459 | 0.6065 |
| GD | full | random | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0000 |
| GD | full | zero | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0000 |
| GD | full | sparse | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0000 |
| GD | mini-batch | random | 0.1 | 0.2359 ± 0.1134 | 0.0857–0.5104 | 0.1373 |
| GD | mini-batch | zero | 0.1 | 0.2359 ± 0.1134 | 0.0857–0.5104 | 0.1373 |
| GD | mini-batch | sparse | 0.1 | 0.2359 ± 0.1134 | 0.0857–0.5104 | 0.1373 |
| GD | SGD | random | 0.1 | 0.5098 ± 0.2733 | 0.1385–1.2380 | 0.7889 |
| GD | SGD | zero | 0.1 | 0.5125 ± 0.3742 | 0.1606–1.5574 | 1.0910 |
| GD | SGD | sparse | 0.1 | 0.4561 ± 0.2518 | 0.1898–1.0962 | 0.7732 |
| AdaGrad | full | random | 0.1 | 0.3017 ± 0.0724 | 0.2087–0.4704 | 0.2950 |
| AdaGrad | full | zero | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0004 |
| AdaGrad | full | sparse | 0.1 | 0.2401 ± 0.0992 | 0.0986–0.4832 | 0.0067 |
| AdaGrad | mini-batch | random | 0.1 | 0.3165 ± 0.0745 | 0.2119–0.4685 | 0.3428 |
| AdaGrad | mini-batch | zero | 0.1 | 0.2327 ± 0.0999 | 0.0832–0.4728 | 0.0506 |
| AdaGrad | mini-batch | sparse | 0.1 | 0.2326 ± 0.0978 | 0.0831–0.4735 | 0.0861 |
| AdaGrad | SGD | random | 0.1 | 0.7453 ± 0.1603 | 0.4021–1.0198 | 0.9625 |
| AdaGrad | SGD | zero | 0.1 | 0.3038 ± 0.1343 | 0.0666–0.6115 | 0.3299 |
| AdaGrad | SGD | sparse | 0.1 | 0.7283 ± 0.1735 | 0.3656–1.0289 | 1.1389 |
| RMSProp | full | random | 0.1 | 0.2551 ± 0.1183 | 0.0383–0.5115 | 0.1000 |
| RMSProp | full | zero | 0.1 | 0.2666 ± 0.1244 | 0.0643–0.5441 | 0.1000 |
| RMSProp | full | sparse | 0.1 | 0.2017 ± 0.1286 | 0.0342–0.5115 | 0.1000 |
| RMSProp | mini-batch | random | 0.1 | 0.2769 ± 0.1205 | 0.1143–0.6432 | 0.2772 |
| RMSProp | mini-batch | zero | 0.1 | 0.2769 ± 0.1205 | 0.1143–0.6432 | 0.2772 |
| RMSProp | mini-batch | sparse | 0.1 | 0.2769 ± 0.1205 | 0.1143–0.6432 | 0.2772 |
| RMSProp | SGD | random | 0.1 | 0.4591 ± 0.2118 | 0.0528–0.9606 | 0.7401 |
| RMSProp | SGD | zero | 0.1 | 0.4222 ± 0.2402 | 0.1474–1.0504 | 0.5974 |
| RMSProp | SGD | sparse | 0.1 | 0.4236 ± 0.1945 | 0.1885–0.9159 | 0.6705 |
| Adam | full | random | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0000 |
| Adam | full | zero | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0000 |
| Adam | full | sparse | 0.1 | 0.2399 ± 0.0993 | 0.0987–0.4832 | 0.0000 |
| Adam | mini-batch | random | 0.1 | 0.2900 ± 0.1606 | 0.1126–0.6919 | 0.3420 |
| Adam | mini-batch | zero | 0.1 | 0.2965 ± 0.1610 | 0.1384–0.7140 | 0.3428 |
| Adam | mini-batch | sparse | 0.1 | 0.2968 ± 0.1617 | 0.1389–0.7169 | 0.3439 |
| Adam | SGD | random | 0.1 | 0.5601 ± 0.2489 | 0.2263–1.1699 | 0.7601 |
| Adam | SGD | zero | 0.1 | 0.4952 ± 0.2670 | 0.1203–1.0923 | 0.8944 |
| Adam | SGD | sparse | 0.1 | 0.5540 ± 0.2493 | 0.2336–1.0875 | 0.8163 |

**Observations.**

- The mean weight error is 0.2399 for Naive and 0.0194 for Oracle.
- The 6 combinations that reach Naive are GD and Adam, full batch, lr 0.1,
  all three initialisations. Every optimizer minimises the MSE on all
  samples, whose minimiser is the Naive solution.
- The smallest grid mean is 0.1905 (RMSProp, SGD, `zero`, lr 0.01).

**Decision.** Develop a sample-selection method, starting from `ours_v1`.

## D1 — The starting method `ours_v1`

**Setting.** `experiments/dev1_v1.py` → `results/dev1/`. `ours_v1`
(`src/outlier_regression/outlier_removal.py`): 5 cycles of 200 full-batch
Adam iterations at lr 0.01. Each cycle starts from new `N(0, 1)` weights and
a fresh Adam state. After cycles 1–4, the 10% of kept samples with the largest
`|r|` are removed permanently. The closed form on the final kept set separates
the effect of the selection from that of the training.

**D1-a weight error** (`results/dev1/tables.md`)

| method | weight error (mean ± SD) | min–max |
| --- | --- | --- |
| Oracle | 0.0194 ± 0.0093 | 0.0030–0.0350 |
| Naive | 0.2399 ± 0.0993 | 0.0987–0.4832 |
| ours_v1 | 0.4664 ± 0.3472 | 0.0658–1.1549 |

**D1-b per-dataset comparison** (`results/dev1/tables.md`)

| comparison | datasets |
| --- | --- |
| ours_v1 < Naive | 2 / 20 |
| ours_v1 < Oracle | 0 / 20 |

**D1-c prunes, totals over the 20 datasets (2016 outliers in total)** (`results/dev1/tables.md`)

| prune (after cycle) | outliers removed | clean removed | outliers left |
| --- | --- | --- | --- |
| 1 | 1688 | 312 | 328 |
| 2 | 323 | 1477 | 5 |
| 3 | 5 | 1615 | 0 |
| 4 | 0 | 1460 | 0 |

**D1-d gradient norm at the end of each cycle** (`results/dev1/tables.md`)

| cycle | ‖∇‖ at the end (mean) | min–max |
| --- | --- | --- |
| 1 | 0.2335 | 0.1361–0.3324 |
| 2 | 0.0604 | 0.0107–0.1286 |
| 3 | 0.0178 | 0.0000–0.0503 |
| 4 | 0.0029 | 0.0000–0.0209 |
| 5 | 0.0748 | 0.0145–0.1604 |

**D1-e final kept set** (`results/dev1/tables.md`)

| quantity | value |
| --- | --- |
| samples kept after the last prune | 656–656 |
| outliers kept, total over datasets | 0 |
| ours_v1 (trained) | 0.4664 ± 0.3472 |
| closed form on the final kept set | 0.0678 ± 0.0158 |

**Observations.**

1. **Training does not converge.** On the same kept sets, the closed form
   reaches 0.0678 and the trained weights 0.4664. The mean gradient norm at
   the end of cycle 5 is 0.0748.
2. **Pruning continues after the outliers are gone.** Over the 20 datasets,
   5 outliers are left after prune 2 and none after prune 3. Prunes 3 and 4
   remove 3075 clean samples (1615 + 1460). Every dataset ends with 656
   samples, and the closed form on them (0.0678) is worse than Oracle
   (0.0194).

**Decision.** Address observation 1 first (D2), then observation 2 (D3–D4).

## D2 — Per-cycle convergence (D1, observation 1)

**Plan (written before the run).** `ours_v2` with the pruning of `ours_v1`
(ratio 10%, 5 cycles, no stopping rule), but each cycle trains until
`‖∇‖ < tol` (at most 20000 iterations). Grid: `tol` ∈ {1e-3, 1e-4, 1e-5,
1e-6} × lr ∈ {0.01, 0.1, 0.5}, plus fixed 200-iteration cycles at each lr for
reference. Choice:

- `tol`: the largest value whose mean weight error is tied (difference below
  0.0001) with that of `tol = 1e-6` at every learning rate.
- lr: among the learning rates whose mean at that `tol` is tied with the
  smallest, the one with the fewest mean total iterations.

**Setting.** `experiments/dev2_converge.py` → `results/dev2/`. Seeds 0–19.

**D2-a ratio pruning, 5 cycles, 20 datasets (cycle length: fixed 200 iterations or until ‖∇‖ < tol)** (`results/dev2/tables.md`)

| lr | cycle length | weight error (mean ± SD) | closed form on kept set (mean) | max |trained − closed form| | total iterations (mean) | longest cycle | capped cycles | outliers kept (total) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | fixed 200 | 0.4664 ± 0.3472 | 0.0678 | 1.0774 | 1000.0 | 200 | 0 | 0 |
| 0.01 | 1e-3 | 0.0298 ± 0.0126 | 0.0299 | 0.0053 | 2869.4 | 1669 | 0 | 0 |
| 0.01 | 1e-4 | 0.0290 ± 0.0126 | 0.0290 | 0.0006 | 3521.2 | 2013 | 0 | 0 |
| 0.01 | 1e-5 | 0.0289 ± 0.0126 | 0.0289 | 0.0001 | 4078.1 | 2297 | 0 | 0 |
| 0.01 | 1e-6 | 0.0289 ± 0.0126 | 0.0289 | 0.0000 | 4566.8 | 2539 | 0 | 0 |
| 0.1 | fixed 200 | 0.0289 ± 0.0126 | 0.0289 | 0.0001 | 1000.0 | 200 | 0 | 0 |
| 0.1 | 1e-3 | 0.0301 ± 0.0129 | 0.0293 | 0.0050 | 518.0 | 159 | 0 | 0 |
| 0.1 | 1e-4 | 0.0290 ± 0.0125 | 0.0289 | 0.0004 | 746.8 | 200 | 0 | 0 |
| 0.1 | 1e-5 | 0.0289 ± 0.0126 | 0.0289 | 0.0000 | 967.4 | 249 | 0 | 0 |
| 0.1 | 1e-6 | 0.0289 ± 0.0126 | 0.0289 | 0.0000 | 1189.2 | 296 | 0 | 0 |
| 0.5 | fixed 200 | 0.0291 ± 0.0125 | 0.0291 | 0.0000 | 1000.0 | 200 | 0 | 0 |
| 0.5 | 1e-3 | 0.0293 ± 0.0134 | 0.0294 | 0.0043 | 528.6 | 133 | 0 | 0 |
| 0.5 | 1e-4 | 0.0291 ± 0.0131 | 0.0292 | 0.0003 | 761.6 | 176 | 0 | 0 |
| 0.5 | 1e-5 | 0.0289 ± 0.0126 | 0.0289 | 0.0001 | 980.5 | 303 | 0 | 0 |
| 0.5 | 1e-6 | 0.0289 ± 0.0126 | 0.0289 | 0.0000 | 1240.7 | 1017 | 0 | 0 |

**Observations.**

- At `tol = 1e-5` and `1e-6`, the mean weight error is 0.0289 at all three
  learning rates, and in every run the trained weight error differs from
  that of the closed form on the kept set by at most 0.0001.
- Fixed 200-iteration cycles at lr 0.01 are `ours_v1` (0.4664, as in D1).
- No cycle reached the 20000-iteration cap.
- With converged cycles, the remaining error comes from the kept set: the
  closed form on it is 0.0289, against 0.0194 for Oracle (D0).

**Decision** (by the rule above, with ties as "equal at 4 decimals"; the
first version gives the same choice, see item 2 of "Changes made after a
run"). `tol = 1e-4` and `1e-3` are not tied with
`1e-6` at lr 0.01 (0.0290 and 0.0298 against 0.0289); `tol = 1e-5` is tied at
all three learning rates. At `tol = 1e-5` the three learning rates are tied
(0.0289), and lr 0.1 needs the fewest iterations (967.4 on average). **Chosen:
`tol = 1e-5`, lr 0.1.**

## D3 — Stopping rule for ratio pruning (D1, observation 2)

**Plan (written before the run).** Ratio pruning with `tol = 1e-5`, lr 0.1,
at most 5 cycles, plus a stopping rule: after each cycle, compute
`σ_MAD = 1.4826 · median(|r − median(r)|)` on the kept samples; if no kept
sample has `|r| > k·σ_MAD`, stop. `k` ∈ {2, 2.5, 3, 3.5, 4}; no rule as the
reference. Choice: the smallest mean weight error; ties are broken by the `k`
closest to 3.

**Setting.** `experiments/dev3_stop.py` → `results/dev3/`. Seeds 0–19.
"Clean left out" counts clean samples outside the final kept set, split by
the sign of their true noise `ε = y − x·w1`.

**D3-a ratio pruning with stopping rule k (tol 1e-5, lr 0.1, at most 5 cycles, 20 datasets)** (`results/dev3/tables.md`)

| k | weight error (mean ± SD) | prunes (mean) | stopped by the rule | clean kept (mean) | outliers kept (total) | clean left out, noise + / − (total) |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.0289 ± 0.0126 | 4.00 | – | 656.00 | 0 | 2455 / 2409 |
| 2.0 | 0.0236 ± 0.0106 | 2.15 | 20 / 20 | 797.85 | 0 | 1062 / 965 |
| 2.5 | 0.0230 ± 0.0103 | 2.00 | 20 / 20 | 810.00 | 0 | 937 / 847 |
| 3.0 | 0.0230 ± 0.0103 | 2.00 | 20 / 20 | 810.00 | 0 | 937 / 847 |
| 3.5 | 0.0230 ± 0.0103 | 2.00 | 20 / 20 | 810.00 | 0 | 937 / 847 |
| 4.0 | 0.0229 ± 0.0102 | 1.80 | 20 / 20 | 827.75 | 5 | 777 / 652 |

**Observations.**

- With any `k`, the rule stopped pruning within 5 cycles on all 20 datasets,
  after 1.80–2.15 prunes on average instead of 4.
- `k` = 2.5, 3 and 3.5 give the same means (0.0230, 810 clean samples kept,
  no outlier kept). `k = 4` keeps more clean samples (827.75) and 5 outliers,
  with a mean of 0.0229.
- Every setting keeps at most 827.75 clean samples on average, while the
  datasets have 881–918 clean samples: each prune removes a fixed 10% of the
  kept samples.

**Decision (by the rule above).** **Chosen: `k = 4`** (0.0229, the smallest
mean). This setting is called **ratio-stop** below.

## D4 — Pruning rule (D1, observation 2)

**Plan (written before the run).** Ratio pruning removes a fixed 10% per
prune, including clean samples. Two rules use the `k·σ_MAD` cut itself,
with `tol = 1e-5`, lr 0.1:

- **threshold:** remove the kept samples with `|r| > k·σ_MAD`; stop when none
  are left. Removed samples do not return.
- **readmit:** after each cycle, select from **all** samples those with
  `|r| ≤ k·σ_MAD` (`σ_MAD` of the kept samples' residuals); samples removed
  earlier can return. Stop when the selected set equals the current one.

`k` ∈ {2, 2.5, 3, 3.5, 4}, at most 20 cycles, compared with ratio-stop from D3.
Choice: the smallest mean weight error over all rules and `k`; ties are broken
by the `k` closest to 3, then by the fewest mean cycles. The cycle cap is
chosen in D5.

**Setting.** `experiments/dev4_rules.py` → `results/dev4/`. Seeds 0–19.

**D4-a pruning rules (tol 1e-5, lr 0.1, 20 datasets; ratio-stop at most 5 cycles, the others at most 20)** (`results/dev4/tables.md`)

| rule | k | weight error (mean ± SD) | cycles (mean) | cycles (max) | stopped by the rule | clean kept (mean) | outliers kept (total) | clean left out, noise + / − (total) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ratio-stop | 4.0 | 0.0229 ± 0.0102 | 2.80 | 3 | 20 / 20 | 827.75 | 5 | 777 / 652 |
| threshold | 2.0 | 0.0598 ± 0.0301 | 6.15 | 11 | 20 / 20 | 656.50 | 0 | 3737 / 1117 |
| threshold | 2.5 | 0.0384 ± 0.0212 | 3.60 | 5 | 20 / 20 | 804.30 | 0 | 1676 / 222 |
| threshold | 3.0 | 0.0265 ± 0.0112 | 3.00 | 4 | 20 / 20 | 866.75 | 4 | 613 / 36 |
| threshold | 3.5 | 0.0219 ± 0.0089 | 2.70 | 3 | 20 / 20 | 891.40 | 13 | 150 / 6 |
| threshold | 4.0 | 0.0210 ± 0.0088 | 2.65 | 3 | 20 / 20 | 897.95 | 24 | 25 / 0 |
| readmit | 2.0 | 0.0233 ± 0.0117 | 8.25 | 17 | 20 / 20 | 839.20 | 0 | 600 / 600 |
| readmit | 2.5 | 0.0206 ± 0.0098 | 5.60 | 20 | 19 / 20 | 885.20 | 1 | 150 / 130 |
| readmit | 3.0 | 0.0201 ± 0.0094 | 3.65 | 5 | 20 / 20 | 896.50 | 4 | 26 / 28 |
| readmit | 3.5 | 0.0199 ± 0.0096 | 3.05 | 4 | 20 / 20 | 898.75 | 13 | 3 / 6 |
| readmit | 4.0 | 0.0207 ± 0.0094 | 2.80 | 3 | 20 / 20 | 899.15 | 24 | 1 / 0 |

**Observations.**

- **threshold** leaves out more clean samples with positive noise than with
  negative noise at every `k` (for example 613 / 36 at `k = 3`); removed
  samples do not return.
- At `k` ≤ 3.5, **readmit** leaves out clean samples of both signs (for
  example 26 / 28 at `k = 3`). At every `k`, it keeps more clean samples on
  average than threshold.
- readmit with `k = 2.5` did not stop within 20 cycles on 1 of the 20
  datasets; every other run stopped by its rule.
- Within readmit, a larger `k` keeps more clean samples and more outliers
  (0, 1, 4, 13 and 24 outliers for `k` = 2 to 4): at `k = 3.5`, readmit
  keeps 898.75 clean samples on average and 13 outliers in total.

**Decision (by the rule above).** **Chosen: readmit, `k = 3.5`** (0.0199, the
smallest mean).

## D5 — Final settings

**Plan (written before the run).** Fix the cycle cap at 50, as a safety limit
far above the cycles used in D4 (at most 5 for readmit with `k` ≥ 3), and run
the chosen method at lr 0.01, 0.1 and 0.5 to check that the choice of lr from
D2 still holds. No setting is chosen from this run unless lr 0.1 is not tied
with the smallest mean.

**Setting.** `experiments/dev5_final.py` → `results/dev5/`. Seeds 0–19.

**D5-a readmit, k = 3.5, tol 1e-5, at most 50 cycles, 20 datasets** (`results/dev5/tables.md`)

| lr | weight error (mean ± SD) | cycles (max) | stopped by the rule | longest cycle (iterations) |
| --- | --- | --- | --- | --- |
| 0.01 | 0.0199 ± 0.0095 | 4 | 20 / 20 | 2297 |
| 0.1 | 0.0199 ± 0.0096 | 4 | 20 / 20 | 249 |
| 0.5 | 0.0199 ± 0.0095 | 4 | 20 / 20 | 432 |

**Observations.** The mean is 0.0199 at all three learning rates; every run
stopped by the rule within 4 cycles.

**Decision.** lr 0.1 is kept. The final method is fixed (next section).

## Final method `ours`

`ours` in `src/outlier_regression/outlier_removal.py`, i.e.
`ours_v2(prune_rule="readmit", stop_k=3.5, converge_tol=1e-5, max_cycles=50,
max_cycle_iters=20000, learning_rate=0.1)`:

1. Start with all samples kept.
2. Initialise the weights from `N(0, 1)` and a fresh Adam state
   (`β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`); train full-batch Adam at lr 0.1 on
   the kept samples until `‖∇‖ < 1e-5` (at most 20000 iterations).
3. Compute `σ_MAD` of the kept samples' residuals and select, from all
   samples, those with `|r| ≤ 3.5·σ_MAD`.
4. If the selection equals the kept set, stop. Otherwise it becomes the kept
   set; go to 2 (at most 50 cycles in total).

| setting | value | chosen in |
| --- | --- | --- |
| convergence tolerance | `1e-5` | D2 |
| learning rate | 0.1 | D2, checked in D5 |
| pruning rule | readmit | D4 |
| `k` | 3.5 | D4 |
| cycle cap | 50 | D5 (safety limit) |
| iteration cap per cycle | 20000 | fixed before D2 |

## T1 plan (written before the run)

- **Data.** Seeds 100–199 (100 datasets), not used before.
- **Methods.** Oracle, Naive, the 72-combination optimizer grid of E1,
  `ours_v1` (D1), ratio-stop (D3: ratio 10%, `k = 4`, at most 5 cycles),
  threshold with `k = 4` (the best threshold setting in D4, at most 20
  cycles), and `ours`. Every setting is as in development.
- **Hypotheses.** Per-dataset differences `d = ours − B` for
  B ∈ {Naive, `ours_v1`, ratio-stop, threshold}. H0: mean `d` = 0.
- **Tests.** Two-sided sign-flip permutation test of mean `d` (100000 random
  sign vectors, `numpy.random.default_rng(0)`, p = (count + 1) / 100001).
  Holm correction over the four hypotheses at α = 0.05. Reported with each:
  mean `d`, 95% percentile bootstrap CI (10000 resamples,
  `default_rng(1)`), and the number of datasets with `d < 0`.
- **Descriptive (no test).** `ours − Oracle` with the same CI; the grid
  summary as in D0; `ours` at lr 0.01 and 0.5; cycles used, runs stopped by
  the rule, clean samples kept and outliers kept.
- The results are reported whatever they are; no setting is changed after the
  run.

## T1 — Final evaluation (seeds 100–199)

**Setting.** `experiments/t1_final.py` → `results/t1/`, run as planned (see
item 3 of "Changes made after a run").
`tests.json` holds the unrounded test values.

**T1-a weight error, 100 datasets** (`results/t1/tables.md`)

| method | weight error (mean ± SD) | median | min–max |
| --- | --- | --- | --- |
| Oracle | 0.0186 ± 0.0081 | 0.0173 | 0.0046–0.0389 |
| Naive | 0.2441 ± 0.0679 | 0.2299 | 0.0850–0.4573 |
| optimizer grid, best mean (RMSProp, SGD, zero, lr 0.01) | 0.1791 ± 0.0721 | 0.1692 | 0.0413–0.3941 |
| ours_v1 | 0.5461 ± 0.3498 | 0.4967 | 0.0144–1.5263 |
| ratio-stop | 0.0215 ± 0.0095 | 0.0198 | 0.0042–0.0493 |
| threshold | 0.0191 ± 0.0081 | 0.0184 | 0.0049–0.0367 |
| ours | 0.0187 ± 0.0083 | 0.0172 | 0.0048–0.0386 |

**T1-b pre-registered tests** (`results/t1/tables.md`)

| B | mean of ours − B | 95% CI | ours < B | p | p (Holm) | H0 rejected |
| --- | --- | --- | --- | --- | --- | --- |
| Naive | -0.2254 | [-0.2392, -0.2121] | 100 / 100 | < 0.0001 | < 0.0001 | yes |
| ours_v1 | -0.5274 | [-0.5987, -0.4610] | 100 / 100 | < 0.0001 | < 0.0001 | yes |
| ratio-stop | -0.0028 | [-0.0041, -0.0016] | 61 / 100 | < 0.0001 | 0.0001 | yes |
| threshold | -0.0005 | [-0.0008, -0.0001] | 53 / 100 | 0.0129 | 0.0129 | yes |

**T1-c ours − Oracle (descriptive)** (`results/t1/tables.md`)

| quantity | value |
| --- | --- |
| mean of ours − Oracle | 0.0001 |
| 95% CI | [-0.0002, 0.0004] |
| datasets with ours < Oracle | 49 / 100 |

**T1-d ours by learning rate (descriptive)** (`results/t1/tables.md`)

| lr | weight error of ours (mean ± SD) |
| --- | --- |
| 0.01 | 0.0187 ± 0.0083 |
| 0.1 | 0.0187 ± 0.0083 |
| 0.5 | 0.0187 ± 0.0083 |

**T1-e selection by ours (descriptive)** (`results/t1/tables.md`)

| quantity | value |
| --- | --- |
| cycles used (min–max) | 2–4 |
| runs stopped by the rule | 100 / 100 |
| longest cycle (iterations) | 290 |
| clean samples kept (total) | 89826 of 89876 |
| clean left out, noise + / − (total) | 28 / 22 |
| outliers kept (total) | 55 of 10124 |
| datasets with at least one outlier kept | 33 / 100 |

Figure `results/t1/strip.png`: per-dataset weight errors; the bars and
numbers are the medians of T1-a.

**T1-grid-a Oracle and Naive** (`results/t1/tables.md`)

| method | weight error (mean ± SD) | min–max |
| --- | --- | --- |
| Oracle | 0.0186 ± 0.0081 | 0.0046–0.0389 |
| Naive | 0.2441 ± 0.0679 | 0.0850–0.4573 |

**T1-grid-b optimizer grid overview** (`results/t1/tables.md`)

| quantity | value |
| --- | --- |
| outliers per dataset (min–max) | 81–125 |
| combinations reaching Naive (max ‖ŵ − ŵ_naive‖ is 0.0000 at 4 decimals) | 5 / 72 |
| smallest mean weight error over the grid | 0.1791 (RMSProp, SGD, zero, lr 0.01) |
| datasets where that combination is below Naive | 79 / 100 |

**T1-grid-c optimizer grid, all 72 combinations** (`results/t1/tables.md`)

| optimizer | batch | init | lr | weight error (mean ± SD) | min–max | max ‖ŵ − ŵ_naive‖ |
| --- | --- | --- | --- | --- | --- | --- |
| GD | full | random | 0.01 | 0.3850 ± 0.0883 | 0.1444–0.5899 | 0.3875 |
| GD | full | zero | 0.01 | 0.2736 ± 0.0665 | 0.1027–0.4595 | 0.1431 |
| GD | full | sparse | 0.01 | 0.4632 ± 0.0921 | 0.2115–0.6725 | 0.4843 |
| GD | mini-batch | random | 0.01 | 0.3835 ± 0.0887 | 0.1432–0.5931 | 0.3854 |
| GD | mini-batch | zero | 0.01 | 0.2709 ± 0.0673 | 0.1036–0.4679 | 0.1365 |
| GD | mini-batch | sparse | 0.01 | 0.4618 ± 0.0938 | 0.2144–0.6596 | 0.4798 |
| GD | SGD | random | 0.01 | 0.3965 ± 0.1095 | 0.1664–0.6275 | 0.5655 |
| GD | SGD | zero | 0.01 | 0.2835 ± 0.0933 | 0.0740–0.5686 | 0.3165 |
| GD | SGD | sparse | 0.01 | 0.4695 ± 0.1115 | 0.1551–0.7337 | 0.5656 |
| AdaGrad | full | random | 0.01 | 1.6377 ± 0.2998 | 0.9086–2.2565 | 2.2481 |
| AdaGrad | full | zero | 0.01 | 0.5283 ± 0.1617 | 0.1161–0.8976 | 0.6875 |
| AdaGrad | full | sparse | 0.01 | 1.5274 ± 0.3349 | 0.9070–2.0701 | 2.0425 |
| AdaGrad | mini-batch | random | 0.01 | 1.6422 ± 0.2995 | 0.9231–2.2623 | 2.2548 |
| AdaGrad | mini-batch | zero | 0.01 | 0.5386 ± 0.1633 | 0.1238–0.9147 | 0.6915 |
| AdaGrad | mini-batch | sparse | 0.01 | 1.6288 ± 0.2996 | 1.0664–2.1189 | 2.0524 |
| AdaGrad | SGD | random | 0.01 | 1.7451 ± 0.2947 | 1.0630–2.3522 | 2.3434 |
| AdaGrad | SGD | zero | 0.01 | 0.6769 ± 0.1981 | 0.1479–1.1215 | 0.7894 |
| AdaGrad | SGD | sparse | 0.01 | 1.8712 ± 0.2871 | 1.1735–2.4382 | 2.3735 |
| RMSProp | full | random | 0.01 | 0.2436 ± 0.0688 | 0.0764–0.4640 | 0.0100 |
| RMSProp | full | zero | 0.01 | 0.2442 ± 0.0683 | 0.0938–0.4507 | 0.0100 |
| RMSProp | full | sparse | 0.01 | 0.2437 ± 0.0675 | 0.0938–0.4507 | 0.0100 |
| RMSProp | mini-batch | random | 0.01 | 0.2419 ± 0.0729 | 0.0761–0.4750 | 0.1391 |
| RMSProp | mini-batch | zero | 0.01 | 0.2419 ± 0.0729 | 0.0761–0.4743 | 0.1388 |
| RMSProp | mini-batch | sparse | 0.01 | 0.2419 ± 0.0729 | 0.0761–0.4746 | 0.1386 |
| RMSProp | SGD | random | 0.01 | 0.2686 ± 0.1217 | 0.0572–0.6010 | 0.6599 |
| RMSProp | SGD | zero | 0.01 | 0.1791 ± 0.0721 | 0.0413–0.3941 | 0.3387 |
| RMSProp | SGD | sparse | 0.01 | 0.3024 ± 0.1416 | 0.0374–0.6948 | 0.6799 |
| Adam | full | random | 0.01 | 0.2639 ± 0.0601 | 0.1435–0.4647 | 0.1751 |
| Adam | full | zero | 0.01 | 0.2440 ± 0.0679 | 0.0850–0.4573 | 0.0004 |
| Adam | full | sparse | 0.01 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0005 |
| Adam | mini-batch | random | 0.01 | 0.2667 ± 0.0637 | 0.1478–0.4746 | 0.1860 |
| Adam | mini-batch | zero | 0.01 | 0.2405 ± 0.0750 | 0.0859–0.4567 | 0.1237 |
| Adam | mini-batch | sparse | 0.01 | 0.2409 ± 0.0749 | 0.0946–0.4562 | 0.1342 |
| Adam | SGD | random | 0.01 | 0.5076 ± 0.1433 | 0.2102–0.8375 | 0.8233 |
| Adam | SGD | zero | 0.01 | 0.2824 ± 0.0970 | 0.0321–0.5640 | 0.3145 |
| Adam | SGD | sparse | 0.01 | 0.4622 ± 0.1599 | 0.0963–0.8822 | 0.7615 |
| GD | full | random | 0.1 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0000 |
| GD | full | zero | 0.1 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0000 |
| GD | full | sparse | 0.1 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0000 |
| GD | mini-batch | random | 0.1 | 0.2436 ± 0.0788 | 0.0780–0.5868 | 0.2041 |
| GD | mini-batch | zero | 0.1 | 0.2436 ± 0.0788 | 0.0780–0.5868 | 0.2041 |
| GD | mini-batch | sparse | 0.1 | 0.2436 ± 0.0788 | 0.0780–0.5868 | 0.2041 |
| GD | SGD | random | 0.1 | 0.4540 ± 0.2737 | 0.0778–1.4418 | 1.2866 |
| GD | SGD | zero | 0.1 | 0.4381 ± 0.2540 | 0.0634–1.5420 | 1.1812 |
| GD | SGD | sparse | 0.1 | 0.4388 ± 0.2406 | 0.0567–1.3256 | 1.0530 |
| AdaGrad | full | random | 0.1 | 0.3009 ± 0.0619 | 0.1316–0.4754 | 0.3266 |
| AdaGrad | full | zero | 0.1 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0015 |
| AdaGrad | full | sparse | 0.1 | 0.2442 ± 0.0678 | 0.0858–0.4573 | 0.0102 |
| AdaGrad | mini-batch | random | 0.1 | 0.3211 ± 0.0724 | 0.1313–0.5164 | 0.3399 |
| AdaGrad | mini-batch | zero | 0.1 | 0.2414 ± 0.0693 | 0.0974–0.4646 | 0.0831 |
| AdaGrad | mini-batch | sparse | 0.1 | 0.2435 ± 0.0691 | 0.0980–0.4675 | 0.1144 |
| AdaGrad | SGD | random | 0.1 | 0.7388 ± 0.1956 | 0.2997–1.1599 | 1.1360 |
| AdaGrad | SGD | zero | 0.1 | 0.3110 ± 0.0916 | 0.0838–0.6490 | 0.3730 |
| AdaGrad | SGD | sparse | 0.1 | 0.7697 ± 0.2109 | 0.2664–1.3610 | 1.2561 |
| RMSProp | full | random | 0.1 | 0.2459 ± 0.0985 | 0.0506–0.4901 | 0.1000 |
| RMSProp | full | zero | 0.1 | 0.2788 ± 0.0922 | 0.0824–0.5290 | 0.1000 |
| RMSProp | full | sparse | 0.1 | 0.2039 ± 0.1125 | 0.0506–0.5290 | 0.1000 |
| RMSProp | mini-batch | random | 0.1 | 0.2731 ± 0.1150 | 0.0552–0.6141 | 0.3490 |
| RMSProp | mini-batch | zero | 0.1 | 0.2731 ± 0.1150 | 0.0552–0.6141 | 0.3490 |
| RMSProp | mini-batch | sparse | 0.1 | 0.2731 ± 0.1150 | 0.0552–0.6141 | 0.3490 |
| RMSProp | SGD | random | 0.1 | 0.3821 ± 0.2061 | 0.0697–1.1863 | 1.0304 |
| RMSProp | SGD | zero | 0.1 | 0.3743 ± 0.1872 | 0.0645–0.8489 | 0.6856 |
| RMSProp | SGD | sparse | 0.1 | 0.3692 ± 0.1852 | 0.0608–0.9887 | 0.7372 |
| Adam | full | random | 0.1 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0000 |
| Adam | full | zero | 0.1 | 0.2441 ± 0.0679 | 0.0850–0.4573 | 0.0000 |
| Adam | full | sparse | 0.1 | 0.2440 ± 0.0678 | 0.0850–0.4568 | 0.0101 |
| Adam | mini-batch | random | 0.1 | 0.2757 ± 0.1006 | 0.0478–0.5511 | 0.3202 |
| Adam | mini-batch | zero | 0.1 | 0.2819 ± 0.1004 | 0.1156–0.5560 | 0.3207 |
| Adam | mini-batch | sparse | 0.1 | 0.2824 ± 0.1010 | 0.1141–0.5581 | 0.3227 |
| Adam | SGD | random | 0.1 | 0.4702 ± 0.2412 | 0.0576–1.3274 | 1.1600 |
| Adam | SGD | zero | 0.1 | 0.4551 ± 0.2164 | 0.0828–1.1263 | 1.0217 |
| Adam | SGD | sparse | 0.1 | 0.4763 ± 0.2164 | 0.1178–1.0463 | 1.0023 |

**Results.**

- All four null hypotheses are rejected after the Holm correction:
  `ours` has a lower mean weight error than Naive, `ours_v1`, ratio-stop and
  threshold.
- Against threshold, the difference is small (mean −0.0005, CI
  [−0.0008, −0.0001]) and `ours` is lower on 53 of the 100 datasets.
- `ours` and Oracle: mean difference 0.0001, 95% CI [−0.0002, 0.0004];
  `ours` is lower on 49 of the 100 datasets. This comparison was not tested.
- The mean of `ours` is 0.0187 at lr 0.01, 0.1 and 0.5.
- Every run of `ours` stopped by its rule, after 2–4 cycles. Over the 100
  datasets it kept 55 of the 10124 outliers (in 33 datasets) and left out 50
  clean samples (28 with positive noise, 22 with negative).
- `ours_v1` is worse than Naive on average (0.5461 against 0.2441).
- The 5 grid combinations that reach Naive are GD (full batch, lr 0.1, all
  three initialisations) and Adam (full batch, lr 0.1, `random` and `zero`).
  The smallest grid mean is 0.1791.

## Development summary

`experiments/dev_summary.py` → `results/dev_summary/` collects one row per
development step from D0–D4 (seeds 0–19). Figure
`results/dev_summary/strip.png` shows the per-dataset values and the medians.

**S-a development steps, seeds 0–19** (`results/dev_summary/tables.md`)

| step | section | weight error (mean ± SD) | median |
| --- | --- | --- | --- |
| Oracle | reference | 0.0194 ± 0.0093 | 0.0171 |
| Naive | reference | 0.2399 ± 0.0993 | 0.2244 |
| ours_v1 | D1 | 0.4664 ± 0.3472 | 0.3093 |
| + per-cycle convergence | D2 | 0.0289 ± 0.0126 | 0.0273 |
| + stopping rule (ratio-stop) | D3 | 0.0229 ± 0.0102 | 0.0217 |
| threshold, k = 4 | D4 | 0.0210 ± 0.0088 | 0.0220 |
| readmit, k = 3.5 (ours) | D4 | 0.0199 ± 0.0096 | 0.0189 |

## Concept figure

`experiments/plot_concept.py` → `docs/images/concept.png` and
`concept_values.json`. An illustration, not an experiment: 200 points with one
feature, `x ~ U[0,1)`, `y = ±x + 0.1·N(0, 1)`, of which 20 follow the negative
slope (`numpy.random.default_rng(0)`). Least-squares slope through the origin:
1.0085 on the clean points and 0.8409 on all points, shown as 1.01 and 0.84.

## Changes made after a run

Every change of a definition or script made after seeing a result:

1. **"Reaches the closed form / Naive"** (E1, D0, T1). First run: maximum
   distance below 0.0001. Changed after the first E1 and D0 runs to "0.0000
   at 4 decimals", because one D0 combination (Adam, full batch, `zero`,
   lr 0.01) has a maximum distance shown as 0.0001 but was counted as
   reaching Naive. Effect: the D0 count went from 7 to 6; E1 did not change.
   No decision depends on this count.
2. **Tie between two means** (development rule). First version: difference
   below 0.0001. Changed after the D2 run to "equal at 4 decimals", so that
   ties can be read off the tables. Under the first version, D2 makes the
   same choice (`tol = 1e-4` differs from `1e-6` by 0.0002 at lr 0.5, and the
   three learning rates are tied at `tol = 1e-5`). D3–D5 were planned and run
   with the second version.
3. **T1 tables.** After the first T1 run, the script was run again twice to
   rename the grid table ids and to add the median column. `tests.json` did
   not change.

## Environment

Python 3.12.10, NumPy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, Windows 11,
CPU only. With the package installed (`pip install -e .`),
`python experiments/run_all.py` runs the scripts in the order E1, D0–D5,
development summary, T1, concept figure, and rewrites `results/` and
`docs/images/`.
