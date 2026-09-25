### T1-a weight error, 100 datasets

| method | weight error (mean ± SD) | median | min–max |
| --- | --- | --- | --- |
| Oracle | 0.0186 ± 0.0081 | 0.0173 | 0.0046–0.0389 |
| Naive | 0.2441 ± 0.0679 | 0.2299 | 0.0850–0.4573 |
| optimizer grid, best mean (RMSProp, SGD, zero, lr 0.01) | 0.1791 ± 0.0721 | 0.1692 | 0.0413–0.3941 |
| ours_v1 | 0.5461 ± 0.3498 | 0.4967 | 0.0144–1.5263 |
| ratio-stop | 0.0215 ± 0.0095 | 0.0198 | 0.0042–0.0493 |
| threshold | 0.0191 ± 0.0081 | 0.0184 | 0.0049–0.0367 |
| ours | 0.0187 ± 0.0083 | 0.0172 | 0.0048–0.0386 |

### T1-b pre-registered tests

| B | mean of ours − B | 95% CI | ours < B | p | p (Holm) | H0 rejected |
| --- | --- | --- | --- | --- | --- | --- |
| Naive | -0.2254 | [-0.2392, -0.2121] | 100 / 100 | < 0.0001 | < 0.0001 | yes |
| ours_v1 | -0.5274 | [-0.5987, -0.4610] | 100 / 100 | < 0.0001 | < 0.0001 | yes |
| ratio-stop | -0.0028 | [-0.0041, -0.0016] | 61 / 100 | < 0.0001 | 0.0001 | yes |
| threshold | -0.0005 | [-0.0008, -0.0001] | 53 / 100 | 0.0129 | 0.0129 | yes |

### T1-c ours − Oracle (descriptive)

| quantity | value |
| --- | --- |
| mean of ours − Oracle | 0.0001 |
| 95% CI | [-0.0002, 0.0004] |
| datasets with ours < Oracle | 49 / 100 |

### T1-d ours by learning rate (descriptive)

| lr | weight error of ours (mean ± SD) |
| --- | --- |
| 0.01 | 0.0187 ± 0.0083 |
| 0.1 | 0.0187 ± 0.0083 |
| 0.5 | 0.0187 ± 0.0083 |

### T1-e selection by ours (descriptive)

| quantity | value |
| --- | --- |
| cycles used (min–max) | 2–4 |
| runs stopped by the rule | 100 / 100 |
| longest cycle (iterations) | 290 |
| clean samples kept (total) | 89826 of 89876 |
| clean left out, noise + / − (total) | 28 / 22 |
| outliers kept (total) | 55 of 10124 |
| datasets with at least one outlier kept | 33 / 100 |

### T1-grid-a Oracle and Naive

| method | weight error (mean ± SD) | min–max |
| --- | --- | --- |
| Oracle | 0.0186 ± 0.0081 | 0.0046–0.0389 |
| Naive | 0.2441 ± 0.0679 | 0.0850–0.4573 |

### T1-grid-b optimizer grid overview

| quantity | value |
| --- | --- |
| outliers per dataset (min–max) | 81–125 |
| combinations reaching Naive (max ‖ŵ − ŵ_naive‖ is 0.0000 at 4 decimals) | 5 / 72 |
| smallest mean weight error over the grid | 0.1791 (RMSProp, SGD, zero, lr 0.01) |
| datasets where that combination is below Naive | 79 / 100 |

### T1-grid-c optimizer grid, all 72 combinations

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

