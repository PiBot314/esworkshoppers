# B3/B4: guided LSTM with weather, real 2025 test weeks

Built by [b3_guided_lstm.py](../../b3_guided_lstm.py). Blocked split (test weeks). Solar: per kWp, daytime, pooled over plants on ok days; grid: SB00 MW, all hours. *Last step* = the value at the horizon (+1 h or +24 h); *all steps* = every step of the forecast pooled. Mean ± SD over seeds. Skill = 1 − MSE / MSE(best baseline) on the identical windows; the best baseline is chosen separately for the last step and for all steps.

## grid, 1h (test windows 3,998)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.866 | 0.917 | 0.073 | – | – |
| baseline: same_time_yesterday | 0.520 | 0.522 | 0.129 | – | – |
| LSTM noweather | 0.909 ± 0.001 | 0.925 | 0.063 | +0.322 | +0.104 |
| LSTM forecast | 0.911 ± 0.003 | 0.924 | 0.063 | +0.333 | +0.085 |
| LSTM forecast_unguided | 0.903 ± 0.000 | 0.912 | 0.066 | +0.272 | -0.053 |

## grid, 24h (test windows 3,464)

| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |
|---|---|---|---|---|---|
| baseline: persistence | 0.452 | -0.211 | 0.142 | – | – |
| baseline: same_time_yesterday | 0.452 | 0.371 | 0.142 | – | – |
| LSTM noweather | 0.784 ± 0.014 | 0.760 | 0.103 | +0.605 | +0.619 |
| LSTM forecast | 0.804 ± 0.006 | 0.807 | 0.097 | +0.644 | +0.693 |
| LSTM forecast_unguided | 0.797 ± 0.019 | 0.797 | 0.099 | +0.630 | +0.677 |

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
