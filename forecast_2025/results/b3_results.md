# B3/B4: guided LSTM with weather, real 2025 test weeks

Built by [b3_guided_lstm.py](../b3_guided_lstm.py). Blocked split (test weeks). Solar: per kWp, daytime, pooled over plants on ok days; grid: SB00 MW, all hours. *Last step* = the value at the horizon (+1 h or +24 h); *all steps* = every step of the forecast pooled. Mean ± SD over seeds. Skill = 1 − MSE / MSE(best baseline) on the identical windows; the best baseline is chosen separately for the last step and for all steps.

## grid, 1h (test windows 3,998)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.866 | 0.917 | 0.073 | – | – |
| baseline: same_time_yesterday | 0.520 | 0.522 | 0.129 | – | – |
| baseline: same_time_last_week | 0.659 | 0.659 | 0.115 | – | – |
| LSTM noweather | 0.909 ± 0.006 | 0.925 | 0.064 | +0.316 | +0.105 |
| LSTM forecast | 0.914 ± 0.000 | 0.929 | 0.062 | +0.355 | +0.144 |
| LSTM forecast_unguided | 0.891 ± 0.004 | 0.909 | 0.070 | +0.182 | -0.093 |

## grid, 1h (test windows 3,998)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.866 | 0.917 | 0.073 | – | – |
| baseline: same_time_yesterday | 0.520 | 0.522 | 0.129 | – | – |
| baseline: same_time_last_week | 0.659 | 0.659 | 0.115 | – | – |
| LSTM noweather_skip | 0.906 ± 0.002 | 0.923 | 0.064 | +0.299 | +0.082 |
| LSTM forecast_skip | 0.915 ± 0.004 | 0.931 | 0.060 | +0.362 | +0.173 |
| LSTM forecast_unguided_skip | 0.892 ± 0.009 | 0.908 | 0.069 | +0.194 | -0.103 |

## grid, 24h (test windows 3,464)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.452 | -0.211 | 0.142 | – | – |
| baseline: same_time_yesterday | 0.452 | 0.371 | 0.142 | – | – |
| baseline: same_time_last_week | 0.654 | 0.625 | 0.116 | – | – |
| LSTM noweather | 0.743 ± 0.030 | 0.737 | 0.112 | +0.256 | +0.298 |
| LSTM forecast | 0.757 ± 0.017 | 0.763 | 0.107 | +0.298 | +0.369 |
| LSTM forecast_unguided | 0.765 ± 0.005 | 0.765 | 0.107 | +0.319 | +0.375 |

## grid, 24h (test windows 3,464)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.452 | -0.211 | 0.142 | – | – |
| baseline: same_time_yesterday | 0.452 | 0.371 | 0.142 | – | – |
| baseline: same_time_last_week | 0.654 | 0.625 | 0.116 | – | – |
| LSTM noweather_skip | 0.784 ± 0.003 | 0.718 | 0.097 | +0.375 | +0.249 |
| LSTM forecast_skip | 0.766 ± 0.001 | 0.723 | 0.100 | +0.324 | +0.262 |
| LSTM forecast_unguided_skip | 0.768 ± 0.010 | 0.750 | 0.101 | +0.329 | +0.335 |

## solar, 1h (test windows 27,815)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.563 | 0.716 | 0.382 | – | – |
| baseline: same_time_yesterday | 0.660 | 0.666 | 0.254 | – | – |
| baseline: csi_persistence | 0.773 | 0.816 | 0.210 | – | – |
| LSTM noweather | 0.844 ± 0.003 | 0.865 | 0.179 | +0.316 | +0.268 |
| LSTM forecast | 0.842 ± 0.000 | 0.866 | 0.181 | +0.304 | +0.271 |
| LSTM forecast+pre | 0.834 ± 0.002 | 0.860 | 0.190 | +0.272 | +0.241 |
| LSTM forecast+pre_unguided | 0.828 ± 0.001 | 0.856 | 0.191 | +0.244 | +0.216 |

## solar, 24h (test windows 23,807)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.678 | -1.356 | 0.243 | – | – |
| baseline: same_time_yesterday | 0.678 | 0.645 | 0.243 | – | – |
| baseline: csi_persistence | 0.679 | 0.521 | 0.242 | – | – |
| LSTM noweather | 0.798 ± 0.006 | 0.791 | 0.215 | +0.372 | +0.411 |
| LSTM forecast | 0.814 ± 0.001 | 0.811 | 0.208 | +0.420 | +0.467 |
| LSTM forecast+pre | 0.795 ± 0.003 | 0.792 | 0.212 | +0.362 | +0.416 |
| LSTM forecast+pre_unguided | 0.798 ± 0.001 | 0.794 | 0.208 | +0.371 | +0.419 |
