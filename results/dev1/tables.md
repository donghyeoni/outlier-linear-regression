### D1-a weight error

| method | weight error (mean ± SD) | min–max |
| --- | --- | --- |
| Oracle | 0.0194 ± 0.0093 | 0.0030–0.0350 |
| Naive | 0.2399 ± 0.0993 | 0.0987–0.4832 |
| ours_v1 | 0.4664 ± 0.3472 | 0.0658–1.1549 |

### D1-b per-dataset comparison

| comparison | datasets |
| --- | --- |
| ours_v1 < Naive | 2 / 20 |
| ours_v1 < Oracle | 0 / 20 |

### D1-c prunes, totals over the 20 datasets (2016 outliers in total)

| prune (after cycle) | outliers removed | clean removed | outliers left |
| --- | --- | --- | --- |
| 1 | 1688 | 312 | 328 |
| 2 | 323 | 1477 | 5 |
| 3 | 5 | 1615 | 0 |
| 4 | 0 | 1460 | 0 |

### D1-d gradient norm at the end of each cycle

| cycle | ‖∇‖ at the end (mean) | min–max |
| --- | --- | --- |
| 1 | 0.2335 | 0.1361–0.3324 |
| 2 | 0.0604 | 0.0107–0.1286 |
| 3 | 0.0178 | 0.0000–0.0503 |
| 4 | 0.0029 | 0.0000–0.0209 |
| 5 | 0.0748 | 0.0145–0.1604 |

### D1-e final kept set

| quantity | value |
| --- | --- |
| samples kept after the last prune | 656–656 |
| outliers kept, total over datasets | 0 |
| ours_v1 (trained) | 0.4664 ± 0.3472 |
| closed form on the final kept set | 0.0678 ± 0.0158 |

