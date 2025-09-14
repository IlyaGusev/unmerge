=== PIVOT TABLES FOR PHASE 4 REPORT ===


### Results by Decomposition Algorithm

| Algorithm | Reconstruction Error | Sparsity | Component Precision | Component Recall | Perfect Match |
|---|---|---|---|---|---|
| lasso | 0.7410 ± 0.2624 (n=216) | 0.94 ± 1.01 (n=216) | 0.5278 ± 0.4992 (n=216) | 0.3023 ± 0.3340 (n=216) | 0.4167 ± 0.4930 (n=216) |
| nnls | 0.4500 ± 0.2135 (n=216) | 6.10 ± 1.64 (n=216) | 0.4437 ± 0.4404 (n=216) | 0.6667 ± 0.4714 (n=216) | 0.3472 ± 0.4761 (n=216) |
| omp | 0.4494 ± 0.2137 (n=216) | 6.72 ± 1.60 (n=216) | 0.3886 ± 0.4035 (n=216) | 0.6667 ± 0.4714 (n=216) | 0.2500 ± 0.4330 (n=216) |
| elastic_net | 0.4545 ± 0.2120 (n=216) | 7.32 ± 1.01 (n=216) | 0.3194 ± 0.3236 (n=216) | 0.6667 ± 0.4714 (n=216) | 0.0556 ± 0.2291 (n=216) |
| ridge | 0.4494 ± 0.2138 (n=216) | 8.00 ± 0.00 (n=216) | 0.2569 ± 0.2411 (n=216) | 0.6667 ± 0.4714 (n=216) | 0.0000 ± 0.0000 (n=216) |
| dot_product | 0.4910 ± 0.1984 (n=216) | 7.60 ± 0.49 (n=216) | 0.2646 ± 0.2436 (n=216) | 0.6667 ± 0.4714 (n=216) | 0.0000 ± 0.0000 (n=216) |


### Results by Model Category

| Category | Reconstruction Error | Sparsity | Component Precision | Component Recall | Perfect Match |
|---|---|---|---|---|---|
| known | 0.2984 ± 0.1775 (n=432) | 5.62 ± 2.32 (n=432) | 0.7636 ± 0.2426 (n=432) | 0.9081 ± 0.2261 (n=432) | 0.3194 ± 0.4663 (n=432) |
| mixed | 0.4868 ± 0.1794 (n=432) | 6.15 ± 2.55 (n=432) | 0.3369 ± 0.3014 (n=432) | 0.9097 ± 0.2545 (n=432) | 0.0556 ± 0.2291 (n=432) |
| unknown | 0.7324 ± 0.1431 (n=432) | 6.56 ± 2.93 (n=432) | 0.0000 ± 0.0000 (n=432) | 0.0000 ± 0.0000 (n=432) | 0.1597 ± 0.3663 (n=432) |


### Results by Merging Method

| Merging Method | Reconstruction Error | Sparsity | Component Precision | Component Recall | Perfect Match |
|---|---|---|---|---|---|
| ties | 0.5099 ± 0.2178 (n=324) | 6.23 ± 2.52 (n=324) | 0.3877 ± 0.3863 (n=324) | 0.6437 ± 0.4654 (n=324) | 0.2037 ± 0.4028 (n=324) |
| linear | 0.4841 ± 0.2540 (n=324) | 6.07 ± 2.70 (n=324) | 0.3543 ± 0.3817 (n=324) | 0.5895 ± 0.4741 (n=324) | 0.1667 ± 0.3727 (n=324) |
| dare_linear | 0.6147 ± 0.1684 (n=324) | 6.08 ± 2.65 (n=324) | 0.3722 ± 0.3869 (n=324) | 0.6011 ± 0.4699 (n=324) | 0.1759 ± 0.3808 (n=324) |
| task_arithmetic | 0.4148 ± 0.2786 (n=324) | 6.06 ± 2.68 (n=324) | 0.3532 ± 0.3804 (n=324) | 0.5895 ± 0.4741 (n=324) | 0.1667 ± 0.3727 (n=324) |


### Algorithm x Category - Reconstruction Error
| Algorithm | Known | Mixed | Unknown |
|---|---|---|---|
| lasso | 0.5188 | 0.7109 | 0.9933 |
| nnls | 0.2426 | 0.4304 | 0.6769 |
| omp | 0.2416 | 0.4299 | 0.6768 |
| elastic_net | 0.2485 | 0.4349 | 0.6801 |
| ridge | 0.2415 | 0.4297 | 0.6768 |
| dot_product | 0.2973 | 0.4853 | 0.6903 |

### Algorithm x Category - Reconstruction Error
| Algorithm | Known | Mixed | Unknown |
|---|---|---|---|
| lasso | 0.5188 | 0.7109 | 0.9933 |
| nnls | 0.2426 | 0.4304 | 0.6769 |
| omp | 0.2416 | 0.4299 | 0.6768 |
| elastic_net | 0.2485 | 0.4349 | 0.6801 |
| ridge | 0.2415 | 0.4297 | 0.6768 |
| dot_product | 0.2973 | 0.4853 | 0.6903 |

### Algorithm x Merging method- Reconstruction Error
| Algorithm | DARE | TIES | Linear | Task Arithmetic |
|---|---|---|---|---|
| lasso | 0.8088 | 0.6094 | 0.7862 | 0.7595 |
| nnls | 0.5692 | 0.4800 | 0.4147 | 0.3360 |
| omp | 0.5690 | 0.4784 | 0.4145 | 0.3358 |
| elastic_net | 0.5731 | 0.4830 | 0.4200 | 0.3420 |
| ridge | 0.5690 | 0.4784 | 0.4144 | 0.3357 |
| dot_product | 0.5990 | 0.5301 | 0.4548 | 0.3800 |

### Algorithm x Merging method (known category only) - Reconstruction Error
| Algorithm | DARE | TIES | Linear | Task Arithmetic |
|---|---|---|---|---|
| lasso | 0.6445 | 0.3622 | 0.5579 | 0.5108 |
| nnls | 0.4265 | 0.2833 | 0.1721 | 0.0885 |
| omp | 0.4265 | 0.2792 | 0.1721 | 0.0885 |
| elastic_net | 0.4317 | 0.2860 | 0.1795 | 0.0967 |
| ridge | 0.4265 | 0.2792 | 0.1720 | 0.0883 |
| dot_product | 0.4661 | 0.3453 | 0.2288 | 0.1490 |

### Algorithm x Merging method (known category only) - Precision / Recall
| Algorithm | DARE | TIES | Linear | Task Arithmetic |
|---|---|---|---|---|
| lasso | 1.00 / 0.40 | 1.00 / 0.67 | 0.83 / 0.36 | 0.83 / 0.36 |
| nnls | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| omp | 1.00 / 1.00 | 0.56 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| elastic_net | 0.62 / 1.00 | 0.87 / 1.00 | 0.62 / 1.00 | 0.62 / 1.00 |
| ridge | 0.54 / 1.00 | 0.54 / 1.00 | 0.54 / 1.00 | 0.54 / 1.00 |
| dot_product | 0.55 / 1.00 | 0.54 / 1.00 | 0.55 / 1.00 | 0.55 / 1.00 |

### Algorithm x Merging method (known category only) - Sparsity
| Algorithm | DARE | TIES | Linear | Task Arithmetic |
|---|---|---|---|---|
| lasso | 1.67 | 2.83 | 1.50 | 1.50 |
| nnls | 4.33 | 4.33 | 4.33 | 4.33 |
| omp | 4.33 | 7.67 | 4.33 | 4.33 |
| elastic_net | 7.00 | 5.00 | 7.00 | 7.00 |
| ridge | 8.00 | 8.00 | 8.00 | 8.00 |
| dot_product | 7.83 | 8.00 | 7.83 | 7.83 |

### Algorithm x Merging method (mixed category only) - Precision / Recall
| Algorithm | DARE | TIES | Linear | Task Arithmetic |
|---|---|---|---|---|
| lasso | 0.67 / 0.42 | 1.00 / 0.92 | 0.50 / 0.25 | 0.50 / 0.25 |
| nnls | 0.31 / 1.00 | 0.41 / 1.00 | 0.30 / 1.00 | 0.30 / 1.00 |
| omp | 0.28 / 1.00 | 0.26 / 1.00 | 0.29 / 1.00 | 0.27 / 1.00 |
| elastic_net | 0.26 / 1.00 | 0.33 / 1.00 | 0.26 / 1.00 | 0.26 / 1.00 |
| ridge | 0.23 / 1.00 | 0.23 / 1.00 | 0.23 / 1.00 | 0.23 / 1.00 |
| dot_product | 0.25 / 1.00 | 0.23 / 1.00 | 0.25 / 1.00 | 0.25 / 1.00 |

=== ANALYSIS SUMMARY ===
- Total filtered results: 1296
- Algorithms: dot_product, elastic_net, lasso, nnls, omp, ridge
- Categories: known, mixed, unknown
- Merging methods: dare_linear, linear, task_arithmetic, ties
