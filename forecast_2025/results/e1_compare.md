# E1: all methods on identical test rows (21 Nov)

Built by [e1_compare.py](../e1_compare.py). R² at the horizon (last step). 95 % intervals: day-block bootstrap over test days (2000 resamples). *Clean weeks* = the 3 test weeks the deployed models' weights never saw (8–15 Oct, 29 Oct–5 Nov, 24–31 Dec). Seeds: the ONNX models are seed 0; seed spread is in b3_results.md.

## solar_1h (solar per kWp, ok days; 14,024 rows, 55 days)

| Method | R² all test weeks [95 % CI] |
|---|---|
| LSTM noweather | 0.847 [0.803, 0.885] |
| LightGBM forecast | 0.846 [0.806, 0.880] |
| LightGBM noweather | 0.842 [0.801, 0.879] |
| LSTM forecast | 0.841 [0.800, 0.877] |
| LSTM forecast+pre | 0.836 [0.791, 0.876] |
| LSTM forecast+pre_unguided | 0.827 [0.776, 0.870] |
| baseline csi_persistence | 0.772 [0.710, 0.826] |
| baseline same_time_yesterday | 0.660 [0.569, 0.742] |
| baseline persistence | 0.563 [0.501, 0.617] |

Paired differences (A − B, R²; P = share of resamples with A ≤ B):

| A − B | all weeks [95 % CI] | P(A ≤ B) |
|---|---|---|
| LSTM forecast  −  LSTM noweather | -0.006 [-0.017, +0.004] | 0.86 |
| LightGBM forecast  −  LightGBM noweather | +0.004 [-0.007, +0.015] | 0.24 |
| LSTM forecast  −  LightGBM forecast | -0.005 [-0.012, +0.003] | 0.91 |
| LSTM forecast  −  LSTM forecast+pre | +0.005 [-0.003, +0.013] | 0.12 |
| LSTM forecast+pre  −  LSTM forecast+pre_unguided | +0.010 [-0.002, +0.022] | 0.04 |
| LightGBM forecast  −  baseline csi_persistence | +0.074 [+0.051, +0.099] | 0.00 |
| LSTM forecast  −  baseline csi_persistence | +0.069 [+0.047, +0.093] | 0.00 |

by season: MSE skill vs baseline csi_persistence (nMAE)

| Method | monsoon (n 4,325, 14 d) | post-monsoon (n 3,501, 13 d) | summer (n 2,710, 11 d) | winter (n 3,488, 17 d) |
|---|---|---|---|---|
| LSTM noweather | +0.31 (0.30) | +0.30 (0.28) | +0.41 (0.10) | +0.59 (0.05) |
| LightGBM forecast | +0.28 (0.30) | +0.32 (0.27) | +0.37 (0.11) | +0.60 (0.05) |
| LightGBM noweather | +0.23 (0.31) | +0.32 (0.27) | +0.38 (0.11) | +0.52 (0.06) |
| LSTM forecast | +0.30 (0.30) | +0.28 (0.29) | +0.36 (0.12) | +0.43 (0.06) |
| LSTM forecast+pre | +0.25 (0.30) | +0.25 (0.28) | +0.43 (0.10) | +0.54 (0.06) |

by sky: MSE skill vs baseline csi_persistence (nMAE)

| Method | 1 cloudy (n 4,628, 23 d) | 2 mixed (n 4,707, 31 d) | 3 clear (n 4,689, 29 d) |
|---|---|---|---|
| LSTM noweather | +0.28 (0.35) | +0.35 (0.18) | +0.40 (0.08) |
| LightGBM forecast | +0.26 (0.35) | +0.36 (0.17) | +0.38 (0.09) |
| LightGBM noweather | +0.23 (0.36) | +0.37 (0.17) | +0.34 (0.10) |
| LSTM forecast | +0.26 (0.35) | +0.36 (0.18) | +0.27 (0.11) |
| LSTM forecast+pre | +0.21 (0.35) | +0.33 (0.18) | +0.34 (0.09) |

## grid_1h (grid SB00, MW; 3,968 rows, 52 days)

| Method | R² all test weeks [95 % CI] | R² clean weeks [95 % CI] |
|---|---|---|
| LightGBM forecast | 0.931 [0.911, 0.944] | 0.938 [0.913, 0.952] |
| LightGBM noweather | 0.928 [0.907, 0.942] | 0.934 [0.908, 0.948] |
| LSTM forecast | 0.914 [0.889, 0.930] | 0.923 [0.893, 0.940] |
| LSTM forecast_skip | 0.910 [0.884, 0.929] | 0.909 [0.868, 0.933] |
| LSTM v1 (no week) noweather | 0.909 [0.881, 0.928] | 0.904 [0.862, 0.928] |
| LSTM noweather_skip | 0.908 [0.882, 0.926] | 0.908 [0.869, 0.930] |
| LSTM v1 (no week) forecast | 0.908 [0.881, 0.925] | 0.918 [0.889, 0.934] |
| LSTM v1 (no week) forecast_unguided | 0.903 [0.872, 0.923] | 0.907 [0.861, 0.933] |
| LSTM noweather | 0.903 [0.874, 0.922] | 0.916 [0.886, 0.933] |
| LSTM forecast_unguided | 0.895 [0.861, 0.918] | 0.904 [0.857, 0.931] |
| LSTM forecast_unguided_skip | 0.883 [0.831, 0.915] | 0.880 [0.793, 0.925] |
| baseline persistence | 0.866 [0.835, 0.888] | 0.878 [0.832, 0.903] |
| deployed all_sensors_1h (live default) | 0.862 [0.819, 0.889] | 0.867 [0.812, 0.896] |
| deployed guided_h4 | 0.823 [0.765, 0.858] | 0.809 [0.725, 0.852] |
| deployed baseline_h4 | 0.809 [0.750, 0.846] | 0.806 [0.733, 0.847] |
| baseline same_time_last_week | 0.659 [0.472, 0.784] | 0.649 [0.366, 0.830] |
| baseline same_time_yesterday | 0.520 [0.313, 0.684] | 0.599 [0.325, 0.791] |

Paired differences (A − B, R²; P = share of resamples with A ≤ B):

| A − B | all weeks [95 % CI] | P(A ≤ B) | clean weeks [95 % CI] |
|---|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.011 [+0.004, +0.021] | 0.00 | +0.006 [-0.002, +0.015] |
| LightGBM forecast  −  LightGBM noweather | +0.003 [+0.001, +0.006] | 0.01 | +0.004 [+0.002, +0.008] |
| LSTM forecast  −  LightGBM forecast | -0.017 [-0.025, -0.011] | 1.00 | -0.016 [-0.026, -0.007] |
| LSTM forecast  −  LSTM forecast_unguided | +0.019 [+0.007, +0.034] | 0.00 | +0.019 [+0.004, +0.041] |
| LSTM forecast_skip  −  LSTM forecast | -0.003 [-0.013, +0.005] | 0.75 | -0.014 [-0.031, -0.004] |
| LSTM forecast  −  LSTM v1 (no week) forecast | +0.006 [-0.000, +0.013] | 0.04 | +0.005 [-0.005, +0.014] |
| LightGBM forecast  −  baseline persistence | +0.065 [+0.051, +0.082] | 0.00 | +0.060 [+0.045, +0.085] |
| LSTM forecast  −  baseline persistence | +0.047 [+0.032, +0.066] | 0.00 | +0.045 [+0.025, +0.072] |
| LSTM forecast  −  deployed all_sensors_1h (live default) | +0.052 [+0.032, +0.079] | 0.00 | +0.056 [+0.033, +0.092] |
| LightGBM forecast  −  deployed all_sensors_1h (live default) | +0.069 [+0.050, +0.098] | 0.00 | +0.071 [+0.048, +0.112] |
| deployed all_sensors_1h (live default)  −  baseline persistence | -0.004 [-0.040, +0.024] | 0.60 | -0.011 [-0.058, +0.025] |

by season: MSE skill vs baseline persistence (nMAE)

| Method | monsoon (n 1,027, 12 d) | post-monsoon (n 1,288, 16 d) | summer (n 465, 8 d) | winter (n 1,188, 16 d) |
|---|---|---|---|---|
| LightGBM forecast | +0.38 (0.06) | +0.52 (0.05) | +0.60 (0.05) | +0.44 (0.04) |
| LightGBM noweather | +0.35 (0.06) | +0.48 (0.06) | +0.61 (0.05) | +0.39 (0.05) |
| LSTM forecast | +0.26 (0.07) | +0.42 (0.06) | +0.49 (0.05) | +0.10 (0.06) |
| LSTM forecast_skip | +0.33 (0.07) | +0.30 (0.07) | +0.55 (0.05) | +0.15 (0.06) |
| LSTM noweather_skip | +0.27 (0.07) | +0.28 (0.07) | +0.56 (0.05) | +0.18 (0.06) |
| LSTM noweather | +0.06 (0.08) | +0.36 (0.06) | +0.48 (0.06) | +0.10 (0.06) |

by regime: MSE skill vs baseline persistence (nMAE)

| Method | calendar day (holiday/exam/event) (n 288, 6 d) | other hours (n 2,634, 51 d) | weekday 09-18 (n 1,046, 30 d) |
|---|---|---|---|
| LightGBM forecast | -0.18 (0.07) | +0.51 (0.04) | +0.52 (0.07) |
| LightGBM noweather | -0.34 (0.08) | +0.49 (0.04) | +0.50 (0.07) |
| LSTM forecast | -0.61 (0.10) | +0.34 (0.05) | +0.44 (0.07) |
| LSTM forecast_skip | -1.07 (0.11) | +0.34 (0.05) | +0.44 (0.07) |
| LSTM noweather_skip | -0.85 (0.10) | +0.29 (0.06) | +0.43 (0.07) |
| LSTM noweather | -0.83 (0.10) | +0.23 (0.06) | +0.39 (0.08) |

MAE (kW): LightGBM forecast 24, LightGBM noweather 25, LSTM forecast 28, LSTM forecast_skip 28, LSTM v1 (no week) noweather 29, LSTM noweather_skip 29, LSTM v1 (no week) forecast 29, LSTM v1 (no week) forecast_unguided 30

## solar_total_1h (campus solar total, kW, all days; 1,476 rows, 34 days)

| Method | R² all test weeks [95 % CI] | R² clean weeks [95 % CI] |
|---|---|---|
| LSTM forecast | 0.786 [0.728, 0.834] | 0.791 [0.702, 0.861] |
| LSTM noweather | 0.783 [0.721, 0.830] | 0.782 [0.687, 0.851] |
| deployed all_sensors_1h (live default) | 0.748 [0.680, 0.796] | 0.752 [0.647, 0.825] |
| deployed guided_h4 | 0.741 [0.673, 0.793] | 0.748 [0.644, 0.822] |
| baseline csi_persistence | 0.722 [0.636, 0.790] | 0.742 [0.606, 0.844] |
| deployed baseline_h4 | 0.709 [0.651, 0.749] | 0.702 [0.603, 0.763] |
| baseline persistence | 0.517 [0.415, 0.598] | 0.502 [0.330, 0.635] |
| baseline same_time_yesterday | 0.388 [0.176, 0.543] | 0.365 [0.016, 0.599] |

Paired differences (A − B, R²; P = share of resamples with A ≤ B):

| A − B | all weeks [95 % CI] | P(A ≤ B) | clean weeks [95 % CI] |
|---|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.004 [-0.023, +0.032] | 0.41 | +0.009 [-0.033, +0.053] |
| LSTM forecast  −  baseline csi_persistence | +0.065 [+0.030, +0.102] | 0.00 | +0.049 [-0.000, +0.110] |
| LSTM forecast  −  deployed all_sensors_1h (live default) | +0.038 [+0.007, +0.065] | 0.01 | +0.040 [-0.006, +0.085] |
| deployed all_sensors_1h (live default)  −  baseline csi_persistence | +0.026 [-0.008, +0.061] | 0.06 | +0.010 [-0.035, +0.065] |

by season: MSE skill vs baseline csi_persistence (nMAE)

| Method | monsoon (n 685, 14 d) | post-monsoon (n 570, 14 d) | winter (n 221, 6 d) |
|---|---|---|---|
| LSTM forecast | +0.28 (0.27) | +0.20 (0.28) | +0.01 (0.10) |
| LSTM noweather | +0.30 (0.27) | +0.18 (0.29) | -0.50 (0.12) |
| deployed all_sensors_1h (live default) | +0.16 (0.31) | +0.06 (0.33) | -0.68 (0.12) |
| deployed guided_h4 | +0.12 (0.32) | +0.04 (0.33) | -0.36 (0.11) |
| deployed baseline_h4 | +0.09 (0.32) | -0.01 (0.35) | -4.75 (0.23) |

## solar_24h (solar per kWp, ok days; 11,836 rows, 47 days)

| Method | R² all test weeks [95 % CI] |
|---|---|
| LightGBM forecast | 0.818 [0.768, 0.859] |
| LSTM forecast | 0.813 [0.760, 0.855] |
| LSTM noweather | 0.804 [0.752, 0.847] |
| LSTM forecast+pre | 0.798 [0.741, 0.845] |
| LSTM forecast+pre_unguided | 0.797 [0.733, 0.853] |
| LightGBM noweather | 0.763 [0.695, 0.818] |
| baseline csi_persistence | 0.679 [0.583, 0.759] |
| baseline persistence | 0.678 [0.582, 0.758] |
| baseline same_time_yesterday | 0.678 [0.582, 0.758] |

Paired differences (A − B, R²; P = share of resamples with A ≤ B):

| A − B | all weeks [95 % CI] | P(A ≤ B) |
|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.009 [-0.007, +0.024] | 0.14 |
| LightGBM forecast  −  LightGBM noweather | +0.055 [+0.016, +0.107] | 0.00 |
| LSTM forecast  −  LightGBM forecast | -0.006 [-0.028, +0.013] | 0.67 |
| LSTM forecast  −  LSTM forecast+pre | +0.015 [-0.005, +0.036] | 0.07 |
| LSTM forecast+pre  −  LSTM forecast+pre_unguided | +0.001 [-0.015, +0.022] | 0.49 |
| LightGBM forecast  −  baseline csi_persistence | +0.139 [+0.085, +0.198] | 0.00 |
| LSTM forecast  −  baseline csi_persistence | +0.134 [+0.085, +0.192] | 0.00 |

by season: MSE skill vs baseline csi_persistence (nMAE)

| Method | monsoon (n 3,713, 12 d) | post-monsoon (n 2,910, 11 d) | summer (n 2,460, 10 d) | winter (n 2,753, 14 d) |
|---|---|---|---|---|
| LightGBM forecast | +0.45 (0.33) | +0.47 (0.29) | +0.31 (0.13) | -0.43 (0.07) |
| LSTM forecast | +0.42 (0.35) | +0.47 (0.30) | +0.36 (0.12) | -1.47 (0.10) |
| LSTM noweather | +0.40 (0.35) | +0.43 (0.31) | +0.30 (0.14) | -0.53 (0.08) |
| LSTM forecast+pre | +0.36 (0.35) | +0.43 (0.31) | +0.27 (0.14) | -0.69 (0.07) |
| LightGBM noweather | +0.14 (0.44) | +0.41 (0.32) | +0.20 (0.17) | -1.25 (0.09) |

by sky: MSE skill vs baseline csi_persistence (nMAE)

| Method | 1 cloudy (n 3,908, 18 d) | 2 mixed (n 3,969, 27 d) | 3 clear (n 3,959, 26 d) |
|---|---|---|---|
| LightGBM forecast | +0.41 (0.40) | +0.41 (0.17) | +0.52 (0.12) |
| LSTM forecast | +0.40 (0.42) | +0.38 (0.19) | +0.52 (0.12) |
| LSTM noweather | +0.39 (0.42) | +0.37 (0.19) | +0.41 (0.13) |
| LSTM forecast+pre | +0.31 (0.43) | +0.38 (0.18) | +0.50 (0.12) |
| LightGBM noweather | +0.22 (0.49) | +0.36 (0.19) | +0.22 (0.16) |

## grid_24h (grid SB00, MW; 3,203 rows, 44 days)

| Method | R² all test weeks [95 % CI] | R² clean weeks [95 % CI] |
|---|---|---|
| LightGBM forecast | 0.873 [0.827, 0.903] | 0.881 [0.843, 0.908] |
| LightGBM noweather | 0.844 [0.788, 0.883] | 0.835 [0.768, 0.880] |
| LSTM v1 (no week) forecast | 0.799 [0.730, 0.852] | 0.795 [0.701, 0.858] |
| LSTM v1 (no week) noweather | 0.797 [0.731, 0.843] | 0.803 [0.721, 0.857] |
| LSTM noweather_skip | 0.787 [0.695, 0.849] | 0.794 [0.674, 0.862] |
| LSTM v1 (no week) forecast_unguided | 0.778 [0.708, 0.831] | 0.773 [0.689, 0.830] |
| LSTM forecast | 0.774 [0.676, 0.843] | 0.747 [0.600, 0.838] |
| LSTM forecast_unguided | 0.769 [0.672, 0.842] | 0.732 [0.585, 0.836] |
| LSTM forecast_skip | 0.766 [0.641, 0.848] | 0.778 [0.658, 0.861] |
| LSTM forecast_unguided_skip | 0.758 [0.647, 0.838] | 0.763 [0.633, 0.848] |
| LSTM noweather | 0.713 [0.601, 0.799] | 0.690 [0.534, 0.804] |
| deployed guided_24h | 0.660 [0.538, 0.743] | 0.741 [0.588, 0.833] |
| baseline same_time_last_week | 0.654 [0.435, 0.798] | 0.663 [0.336, 0.864] |
| deployed baseline_24h | 0.602 [0.501, 0.681] | 0.612 [0.467, 0.723] |
| deployed cnn_lstm_24h | 0.473 [0.267, 0.600] | 0.545 [0.305, 0.668] |
| baseline persistence | 0.452 [0.201, 0.659] | 0.568 [0.254, 0.793] |
| baseline same_time_yesterday | 0.452 [0.201, 0.659] | 0.568 [0.254, 0.793] |

Paired differences (A − B, R²; P = share of resamples with A ≤ B):

| A − B | all weeks [95 % CI] | P(A ≤ B) | clean weeks [95 % CI] |
|---|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.062 [+0.019, +0.119] | 0.00 | +0.057 [-0.002, +0.142] |
| LightGBM forecast  −  LightGBM noweather | +0.029 [+0.012, +0.052] | 0.00 | +0.047 [+0.022, +0.081] |
| LSTM forecast  −  LightGBM forecast | -0.099 [-0.183, -0.040] | 1.00 | -0.135 [-0.266, -0.046] |
| LSTM forecast  −  LSTM forecast_unguided | +0.005 [-0.029, +0.048] | 0.44 | +0.015 [-0.035, +0.083] |
| LSTM forecast_skip  −  LSTM forecast | -0.009 [-0.113, +0.084] | 0.57 | +0.032 [-0.106, +0.167] |
| LSTM forecast  −  LSTM v1 (no week) forecast | -0.024 [-0.094, +0.030] | 0.77 | -0.049 [-0.149, +0.033] |
| LightGBM forecast  −  baseline same_time_last_week | +0.219 [+0.091, +0.413] | 0.00 | +0.218 [+0.026, +0.528] |
| LSTM forecast  −  baseline same_time_last_week | +0.120 [-0.045, +0.335] | 0.09 | +0.083 [-0.179, +0.428] |
| LSTM forecast  −  deployed guided_24h | +0.114 [+0.010, +0.235] | 0.01 | +0.006 [-0.121, +0.153] |
| LightGBM forecast  −  deployed guided_24h | +0.213 [+0.140, +0.313] | 0.00 | +0.141 [+0.063, +0.273] |
| deployed guided_24h  −  baseline same_time_last_week | +0.006 [-0.142, +0.214] | 0.51 | +0.077 [-0.104, +0.379] |

by season: MSE skill vs baseline same_time_last_week (nMAE)

| Method | monsoon (n 847, 10 d) | post-monsoon (n 1,072, 14 d) | summer (n 371, 6 d) | winter (n 913, 14 d) |
|---|---|---|---|---|
| LightGBM forecast | +0.65 (0.09) | +0.70 (0.08) | +0.49 (0.06) | +0.35 (0.07) |
| LightGBM noweather | +0.66 (0.08) | +0.58 (0.09) | +0.39 (0.07) | +0.14 (0.09) |
| LSTM noweather_skip | +0.44 (0.11) | +0.43 (0.11) | +0.14 (0.08) | +0.15 (0.08) |
| LSTM forecast | +0.63 (0.10) | +0.29 (0.11) | +0.16 (0.07) | -0.13 (0.11) |
| LSTM forecast_skip | +0.28 (0.12) | +0.39 (0.11) | +0.33 (0.07) | +0.10 (0.08) |
| LSTM noweather | +0.45 (0.12) | +0.15 (0.13) | -0.20 (0.09) | -0.39 (0.12) |

by regime: MSE skill vs baseline same_time_last_week (nMAE)

| Method | calendar day (holiday/exam/event) (n 288, 6 d) | other hours (n 2,143, 41 d) | weekday 09-18 (n 772, 23 d) |
|---|---|---|---|
| LightGBM forecast | +0.55 (0.14) | +0.60 (0.06) | +0.70 (0.08) |
| LightGBM noweather | +0.46 (0.17) | +0.55 (0.07) | +0.58 (0.10) |
| LSTM noweather_skip | +0.43 (0.15) | +0.23 (0.09) | +0.49 (0.10) |
| LSTM forecast | +0.52 (0.16) | +0.32 (0.09) | +0.30 (0.12) |
| LSTM forecast_skip | +0.05 (0.20) | +0.31 (0.08) | +0.45 (0.11) |
| LSTM noweather | +0.48 (0.16) | +0.05 (0.10) | +0.14 (0.14) |

MAE (kW): LightGBM forecast 34, LightGBM noweather 38, LSTM noweather_skip 44, LSTM v1 (no week) forecast 44, LSTM forecast_skip 45, LSTM v1 (no week) noweather 45, LSTM forecast_unguided_skip 46, LSTM forecast 46

## solar_total_24h (campus solar total, kW, all days; 1,169 rows, 28 days)

| Method | R² all test weeks [95 % CI] | R² clean weeks [95 % CI] |
|---|---|---|
| LSTM forecast | 0.724 [0.641, 0.786] | 0.738 [0.637, 0.809] |
| LSTM noweather | 0.701 [0.622, 0.763] | 0.698 [0.595, 0.782] |
| deployed guided_24h | 0.677 [0.589, 0.739] | 0.673 [0.598, 0.741] |
| deployed baseline_24h | 0.636 [0.525, 0.714] | 0.680 [0.595, 0.747] |
| deployed cnn_lstm_24h | 0.565 [0.447, 0.652] | 0.547 [0.395, 0.656] |
| baseline csi_persistence | 0.432 [0.169, 0.611] | 0.375 [-0.081, 0.677] |
| baseline persistence | 0.432 [0.168, 0.611] | 0.375 [-0.082, 0.677] |
| baseline same_time_yesterday | 0.432 [0.168, 0.611] | 0.375 [-0.082, 0.677] |

Paired differences (A − B, R²; P = share of resamples with A ≤ B):

| A − B | all weeks [95 % CI] | P(A ≤ B) | clean weeks [95 % CI] |
|---|---|---|---|
| LSTM forecast  −  LSTM noweather | +0.023 [-0.016, +0.058] | 0.12 | +0.039 [-0.022, +0.097] |
| LSTM forecast  −  baseline csi_persistence | +0.292 [+0.136, +0.524] | 0.00 | +0.362 [+0.069, +0.784] |
| LSTM forecast  −  deployed guided_24h | +0.047 [-0.003, +0.094] | 0.03 | +0.065 [-0.000, +0.126] |
| deployed guided_24h  −  baseline csi_persistence | +0.245 [+0.078, +0.493] | 0.00 | +0.297 [-0.013, +0.738] |

by season: MSE skill vs baseline csi_persistence (nMAE)

| Method | monsoon (n 593, 12 d) | post-monsoon (n 443, 12 d) | winter (n 133, 4 d) |
|---|---|---|---|
| LSTM forecast | +0.42 (0.32) | +0.62 (0.27) | -8.62 (0.19) |
| LSTM noweather | +0.41 (0.32) | +0.54 (0.30) | -4.19 (0.14) |
| deployed guided_24h | +0.36 (0.34) | +0.51 (0.32) | -7.75 (0.18) |
| deployed baseline_24h | +0.17 (0.42) | +0.52 (0.33) | -6.26 (0.17) |
| deployed cnn_lstm_24h | +0.18 (0.42) | +0.31 (0.41) | -7.74 (0.19) |
