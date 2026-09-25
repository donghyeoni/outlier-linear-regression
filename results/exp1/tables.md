### E1-a closed form (20 datasets)

| method | weight error (mean ± SD) |
| --- | --- |
| closed form | 0.0168 ± 0.0088 |

### E1-b combinations reaching the closed form (max ‖ŵ − ŵ_cf‖ over the 20 datasets is 0.0000 at 4 decimals), of 9 batch × init combinations

| optimizer | lr 0.01 | lr 0.1 |
| --- | --- | --- |
| GD | 0 / 9 | 3 / 9 |
| AdaGrad | 0 / 9 | 0 / 9 |
| RMSProp | 0 / 9 | 0 / 9 |
| Adam | 1 / 9 | 2 / 9 |

### E1-c full batch, zero init: mean weight error

| optimizer | lr 0.01 | lr 0.1 |
| --- | --- | --- |
| GD | 0.0838 | 0.0168 |
| AdaGrad | 0.4319 | 0.0171 |
| RMSProp | 0.0208 | 0.1006 |
| Adam | 0.0169 | 0.0168 |

### E1-d all 72 combinations

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

