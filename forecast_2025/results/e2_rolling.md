# E2: rolling-origin, LSTM vs LightGBM on identical rows (21 Nov)

Built by [e2_rolling.py](../e2_rolling.py). Folds: test month m = Mar..Dec, train on targets before (1st of m − 16 d), early-stop on the next 14 d, 2-day gap. R² at the horizon pooled over folds; 95 % day-block bootstrap. Compare with the blocked split (e1_compare.md): the drop is what unseen seasons cost.

## grid_1h (26,477 rows, 293 days)

| Method | R² [95 % CI] |
|---|---|
| LightGBM forecast | 0.888 [0.877, 0.898] |
| LightGBM noweather | 0.887 [0.876, 0.897] |
| baseline persistence | 0.845 [0.831, 0.857] |
| LSTM noweather | 0.788 [0.753, 0.819] |
| LSTM forecast | 0.784 [0.749, 0.815] |
| baseline same_time_last_week | 0.638 [0.571, 0.693] |
| baseline same_time_yesterday | 0.468 [0.379, 0.544] |

| A − B | R² diff [95 % CI] | P(A ≤ B) |
|---|---|---|
| LSTM forecast  −  LSTM noweather | -0.004 [-0.012, +0.004] | 0.81 |
| LightGBM forecast  −  LightGBM noweather | +0.001 [-0.001, +0.003] | 0.24 |
| LSTM forecast  −  LightGBM forecast | -0.104 [-0.135, -0.077] | 1.00 |

Per test month (R²):

| Month | n | LSTM forecast | LSTM noweather | LightGBM forecast | LightGBM noweather |
|---|---|---|---|---|---|
| 3 | 2,055 | 0.448 | 0.435 | 0.751 | 0.735 |
| 4 | 2,335 | 0.535 | 0.521 | 0.829 | 0.827 |
| 5 | 2,914 | 0.704 | 0.746 | 0.884 | 0.883 |
| 6 | 2,672 | 0.699 | 0.723 | 0.878 | 0.879 |
| 7 | 2,456 | 0.856 | 0.848 | 0.875 | 0.869 |
| 8 | 2,719 | 0.851 | 0.826 | 0.869 | 0.875 |
| 9 | 2,820 | 0.839 | 0.844 | 0.897 | 0.901 |
| 10 | 2,853 | 0.889 | 0.888 | 0.898 | 0.898 |
| 11 | 2,766 | 0.885 | 0.886 | 0.883 | 0.882 |
| 12 | 2,887 | 0.870 | 0.887 | 0.875 | 0.883 |

## grid_24h (24,545 rows, 286 days)

| Method | R² [95 % CI] |
|---|---|
| LightGBM forecast | 0.713 [0.671, 0.748] |
| LSTM forecast | 0.705 [0.675, 0.731] |
| LSTM noweather | 0.701 [0.668, 0.729] |
| LightGBM noweather | 0.696 [0.655, 0.733] |
| baseline same_time_last_week | 0.626 [0.551, 0.684] |
| baseline persistence | 0.445 [0.347, 0.529] |
| baseline same_time_yesterday | 0.445 [0.347, 0.529] |

| A − B | R² diff [95 % CI] | P(A ≤ B) |
|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.004 [-0.012, +0.021] | 0.32 |
| LightGBM forecast  −  LightGBM noweather | +0.017 [+0.006, +0.027] | 0.00 |
| LSTM forecast  −  LightGBM forecast | -0.008 [-0.041, +0.026] | 0.69 |

Per test month (R²):

| Month | n | LSTM forecast | LSTM noweather | LightGBM forecast | LightGBM noweather |
|---|---|---|---|---|---|
| 3 | 1,871 | 0.618 | 0.634 | 0.383 | 0.377 |
| 4 | 1,963 | 0.489 | 0.585 | 0.288 | 0.202 |
| 5 | 2,780 | 0.564 | 0.675 | 0.740 | 0.697 |
| 6 | 2,486 | 0.706 | 0.575 | 0.806 | 0.810 |
| 7 | 2,234 | 0.703 | 0.739 | 0.781 | 0.748 |
| 8 | 2,422 | 0.695 | 0.596 | 0.700 | 0.716 |
| 9 | 2,688 | 0.629 | 0.662 | 0.815 | 0.811 |
| 10 | 2,688 | 0.713 | 0.685 | 0.673 | 0.689 |
| 11 | 2,623 | 0.753 | 0.709 | 0.711 | 0.670 |
| 12 | 2,790 | 0.623 | 0.372 | 0.562 | 0.556 |

## solar_1h (83,508 rows, 303 days)

| Method | R² [95 % CI] |
|---|---|
| LightGBM forecast | 0.802 [0.784, 0.820] |
| LightGBM noweather | 0.793 [0.774, 0.810] |
| LSTM forecast | 0.783 [0.761, 0.802] |
| LSTM noweather | 0.770 [0.746, 0.790] |
| baseline csi_persistence | 0.744 [0.719, 0.766] |
| baseline same_time_yesterday | 0.598 [0.557, 0.637] |
| baseline persistence | 0.551 [0.527, 0.572] |

| A − B | R² diff [95 % CI] | P(A ≤ B) |
|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.013 [+0.003, +0.024] | 0.01 |
| LightGBM forecast  −  LightGBM noweather | +0.009 [+0.003, +0.015] | 0.00 |
| LSTM forecast  −  LightGBM forecast | -0.019 [-0.030, -0.009] | 1.00 |

Per test month (R²):

| Month | n | LSTM forecast | LSTM noweather | LightGBM forecast | LightGBM noweather |
|---|---|---|---|---|---|
| 3 | 7,515 | 0.893 | 0.888 | 0.925 | 0.931 |
| 4 | 7,501 | 0.897 | 0.908 | 0.899 | 0.906 |
| 5 | 8,856 | 0.662 | 0.700 | 0.796 | 0.778 |
| 6 | 9,273 | 0.662 | 0.566 | 0.672 | 0.668 |
| 7 | 9,503 | 0.714 | 0.642 | 0.667 | 0.664 |
| 8 | 9,026 | 0.696 | 0.680 | 0.665 | 0.655 |
| 9 | 8,366 | 0.676 | 0.663 | 0.697 | 0.680 |
| 10 | 8,112 | 0.686 | 0.690 | 0.705 | 0.697 |
| 11 | 8,000 | 0.862 | 0.845 | 0.884 | 0.841 |
| 12 | 7,356 | 0.951 | 0.954 | 0.955 | 0.953 |

## solar_24h (79,055 rows, 290 days)

| Method | R² [95 % CI] |
|---|---|
| LSTM forecast | 0.750 [0.725, 0.774] |
| LightGBM forecast | 0.744 [0.719, 0.768] |
| LSTM noweather | 0.702 [0.666, 0.734] |
| LightGBM noweather | 0.682 [0.647, 0.713] |
| baseline csi_persistence | 0.598 [0.555, 0.639] |
| baseline persistence | 0.597 [0.554, 0.638] |
| baseline same_time_yesterday | 0.597 [0.554, 0.638] |

| A − B | R² diff [95 % CI] | P(A ≤ B) |
|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.048 [+0.033, +0.066] | 0.00 |
| LightGBM forecast  −  LightGBM noweather | +0.062 [+0.045, +0.080] | 0.00 |
| LSTM forecast  −  LightGBM forecast | +0.006 [-0.007, +0.020] | 0.17 |

Per test month (R²):

| Month | n | LSTM forecast | LSTM noweather | LightGBM forecast | LightGBM noweather |
|---|---|---|---|---|---|
| 3 | 7,275 | 0.910 | 0.915 | 0.917 | 0.909 |
| 4 | 7,256 | 0.866 | 0.871 | 0.873 | 0.872 |
| 5 | 8,539 | 0.761 | 0.641 | 0.709 | 0.620 |
| 6 | 8,940 | 0.529 | 0.334 | 0.592 | 0.554 |
| 7 | 9,119 | 0.678 | 0.649 | 0.517 | 0.522 |
| 8 | 8,409 | 0.594 | 0.510 | 0.582 | 0.502 |
| 9 | 7,669 | 0.556 | 0.557 | 0.587 | 0.538 |
| 10 | 7,442 | 0.638 | 0.547 | 0.648 | 0.343 |
| 11 | 7,581 | 0.848 | 0.834 | 0.853 | 0.772 |
| 12 | 6,825 | 0.919 | 0.917 | 0.953 | 0.944 |
