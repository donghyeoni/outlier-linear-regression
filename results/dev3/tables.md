### D3-a ratio pruning with stopping rule k (tol 1e-5, lr 0.1, at most 5 cycles, 20 datasets)

| k | weight error (mean ± SD) | prunes (mean) | stopped by the rule | clean kept (mean) | outliers kept (total) | clean left out, noise + / − (total) |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.0289 ± 0.0126 | 4.00 | – | 656.00 | 0 | 2455 / 2409 |
| 2.0 | 0.0236 ± 0.0106 | 2.15 | 20 / 20 | 797.85 | 0 | 1062 / 965 |
| 2.5 | 0.0230 ± 0.0103 | 2.00 | 20 / 20 | 810.00 | 0 | 937 / 847 |
| 3.0 | 0.0230 ± 0.0103 | 2.00 | 20 / 20 | 810.00 | 0 | 937 / 847 |
| 3.5 | 0.0230 ± 0.0103 | 2.00 | 20 / 20 | 810.00 | 0 | 937 / 847 |
| 4.0 | 0.0229 ± 0.0102 | 1.80 | 20 / 20 | 827.75 | 5 | 777 / 652 |

