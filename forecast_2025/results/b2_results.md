# B2: LightGBM feature test on real 2025

Built by [b2_lgbm_feature_test.py](../b2_lgbm_feature_test.py). R² on the test rows (solar: daytime, per kWp, pooled over plants on ok days; grid: SB00, all hours). Feature groups are cumulative. Skill = 1 − MSE / MSE of the best baseline on the same rows. `era5` is the perfect-weather upper bound.

## solar, 1h (185,720 rows)

| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |
|---|---|---|---|---|
| baseline: persistence | 0.565 | 0.551 | – | – |
| baseline: same_time_yesterday | 0.631 | 0.591 | – | – |
| baseline: csi_persistence | 0.773 | 0.744 | – | – |
| LightGBM base | 0.837 | 0.790 | +0.280 | +0.180 |
| LightGBM base+clearsky | 0.842 | 0.798 | +0.304 | +0.212 |
| LightGBM base+clearsky+forecast | 0.846 | 0.808 | +0.321 | +0.251 |
| LightGBM base+clearsky+forecast+era5 | 0.848 | 0.814 | +0.329 | +0.272 |

Top features (without ERA5, gain share): `fc15_shortwave_radiation` 77%, `csi_persist` 11%, `hod_cos` 4%, `cs_T` 2%, `fc15_direct_radiation` 2%, `y_yday` 1%, `fc_relative_humidity_2m` 0%, `hod_sin` 0%

## solar, 6h (184,841 rows)

| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |
|---|---|---|---|---|
| baseline: persistence | -2.069 | -2.032 | – | – |
| baseline: same_time_yesterday | 0.631 | 0.591 | – | – |
| baseline: csi_persistence | 0.597 | 0.543 | – | – |
| LightGBM base | 0.751 | 0.686 | +0.325 | +0.233 |
| LightGBM base+clearsky | 0.744 | 0.691 | +0.307 | +0.245 |
| LightGBM base+clearsky+forecast | 0.817 | 0.761 | +0.504 | +0.416 |
| LightGBM base+clearsky+forecast+era5 | 0.827 | 0.779 | +0.531 | +0.460 |

Top features (without ERA5, gain share): `fc15_shortwave_radiation` 80%, `hod_cos` 7%, `fc15_direct_radiation` 5%, `cs_T` 3%, `y_yday` 1%, `fc_relative_humidity_2m` 1%, `doy_sin` 1%, `doy_cos` 0%

## solar, 24h (181,880 rows)

| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |
|---|---|---|---|---|
| baseline: persistence | 0.671 | 0.601 | – | – |
| baseline: same_time_yesterday | 0.671 | 0.601 | – | – |
| baseline: csi_persistence | 0.671 | 0.602 | – | – |
| LightGBM base | 0.763 | 0.694 | +0.279 | +0.232 |
| LightGBM base+clearsky | 0.759 | 0.701 | +0.268 | +0.250 |
| LightGBM base+clearsky+forecast | 0.815 | 0.749 | +0.437 | +0.371 |
| LightGBM base+clearsky+forecast+era5 | 0.833 | 0.782 | +0.491 | +0.452 |

Top features (without ERA5, gain share): `hod_cos` 76%, `cs_T` 9%, `d1_shortwave_radiation` 5%, `y_mean1h` 2%, `d1_relative_humidity_2m` 1%, `zen_T` 1%, `doy_cos` 1%, `anom_temp_d1` 1%

## grid, 1h (29,319 rows)

| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |
|---|---|---|---|---|
| baseline: persistence | 0.867 | 0.846 | – | – |
| baseline: same_time_yesterday | 0.496 | 0.450 | – | – |
| LightGBM base | 0.926 | 0.900 | +0.445 | +0.351 |
| LightGBM base+calendar | 0.927 | 0.901 | +0.450 | +0.358 |
| LightGBM base+calendar+clearsky | 0.927 | 0.901 | +0.451 | +0.359 |
| LightGBM base+calendar+clearsky+forecast | 0.930 | 0.905 | +0.476 | +0.385 |
| LightGBM base+calendar+clearsky+forecast+era5 | 0.930 | 0.904 | +0.476 | +0.379 |

Top features (without ERA5, gain share): `y_t` 68%, `y_t1` 13%, `y_mean1h` 6%, `y_yday` 2%, `y_week` 2%, `hod_cos` 1%, `zen_T` 1%, `fc15_diffuse_radiation` 1%

## grid, 6h (29,319 rows)

| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |
|---|---|---|---|---|
| baseline: persistence | -0.129 | -0.273 | – | – |
| baseline: same_time_yesterday | 0.497 | 0.449 | – | – |
| LightGBM base | 0.826 | 0.750 | +0.654 | +0.547 |
| LightGBM base+calendar | 0.855 | 0.765 | +0.711 | +0.574 |
| LightGBM base+calendar+clearsky | 0.846 | 0.760 | +0.694 | +0.566 |
| LightGBM base+calendar+clearsky+forecast | 0.870 | 0.786 | +0.741 | +0.612 |
| LightGBM base+calendar+clearsky+forecast+era5 | 0.868 | 0.793 | +0.737 | +0.624 |

Top features (without ERA5, gain share): `y_week` 41%, `y_yday` 25%, `dow_sin` 5%, `fc15_temperature_2m` 4%, `hod_sin` 3%, `doy_sin` 2%, `y_t` 2%, `y_mean24h` 2%

## grid, 24h (29,319 rows)

| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |
|---|---|---|---|---|
| baseline: persistence | 0.491 | 0.449 | – | – |
| baseline: same_time_yesterday | 0.491 | 0.449 | – | – |
| LightGBM base | 0.809 | 0.711 | +0.625 | +0.475 |
| LightGBM base+calendar | 0.851 | 0.739 | +0.707 | +0.525 |
| LightGBM base+calendar+clearsky | 0.843 | 0.727 | +0.693 | +0.504 |
| LightGBM base+calendar+clearsky+forecast | 0.870 | 0.732 | +0.745 | +0.514 |
| LightGBM base+calendar+clearsky+forecast+era5 | 0.865 | 0.749 | +0.735 | +0.545 |

Top features (without ERA5, gain share): `y_week` 37%, `y_mean1h` 25%, `dow_sin` 5%, `y_t` 4%, `doy_sin` 3%, `hod_sin` 3%, `doy_cos` 2%, `y_t4` 2%
