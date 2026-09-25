### D0-a Oracle and Naive

| method | weight error (mean ± SD) | min–max |
| --- | --- | --- |
| Oracle | 0.0194 ± 0.0093 | 0.0030–0.0350 |
| Naive | 0.2399 ± 0.0993 | 0.0987–0.4832 |

### D0-b optimizer grid overview

| quantity | value |
| --- | --- |
| outliers per dataset (min–max) | 82–119 |
| combinations reaching Naive (max ‖ŵ − ŵ_naive‖ is 0.0000 at 4 decimals) | 6 / 72 |
| smallest mean weight error over the grid | 0.1905 (RMSProp, SGD, zero, lr 0.01) |
| datasets where that combination is below Naive | 15 / 20 |

### D0-c optimizer grid, all 72 combinations

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

