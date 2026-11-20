# NZ-Edge: Project Direction (post-mideval)

Nov 12, 2026 · @Emmanuel

## Core thesis

NZ-Edge becomes a general flexible-load scheduler for a campus, running on one edge board (QIDK / HDK8650). Borewell and Manjeera pumping is the first case, built on real 2025 IIITH data.

- **What it does:** forecasts load, solar and water demand, then schedules flexible loads every 15 minutes (model-predictive control) against a pluggable objective.
- **Why it matters now:** solar covers only about 18% of demand and never exceeds midday load, so shifting load does not raise solar use today. The present value is peak shaving: the grid peak is about 605 kW at 17:00, when solar is gone.
- **Why it generalises:** each load is written in one common format (power bounds, an energy or state constraint, a deadline or comfort band). Adding a new load means writing a config, not new code.
- **Positioning:** modelling flexible loads as virtual batteries is established demand-response work; cite it. The contribution is the end-to-end system: real campus data, forecasts, a general scheduler, edge deployment, counterfactual results.
- **One-line answer to "edge AI for net zero":** it does not get the campus to net zero. It shows where the gap is, catches what is broken, and moves flexible load to where it costs least, on a few watts.

## Architecture

&#91;embedded content: NZ-Edge architecture · one board, one control loop\]

Sensor streams are resampled on the board, forecast by a distilled model on the NPU, and scheduled by an LP / MILP against per-load configs; fault flags go out alongside the schedules.

## Demo: three loads, three structures

Generality is shown by handling structurally different loads with the same scheduler, not by claiming it.

| Load | Type | Data | Placeholders |
| --- | --- | --- | --- |
| Sump to overhead tank pumping | Storage (tank level) | Real 2025: tank level, PH01 to PH04 flows | Pump power ratings, tank capacities |
| AC pre-cooling | Storage (building thermal mass) | Synthetic 2026 AC / non-AC split | Thermal parameters, comfort band |
| GPU batch jobs on the campus cluster | Deadline (energy by a time) | None yet | Job traces, power per job |

- Interruptible jobs keep it a linear program; on/off pumps or non-interruptible jobs make it a MILP.
- **Placeholders are ranges, not single values.** Every result comes with a sensitivity analysis, e.g. "peak reduction stays between X and Y kW across plausible pump ratings".
- **Evaluation is counterfactual:** replay 2025 and compare actual operation with the scheduled one. No live control of real pumps is assumed.
- **Interpretable output from the LP duals:** e.g. the value of an extra kL of tank capacity in kW of peak reduction, or the cost of keeping tanks at least 30% full.
- **Pluggable objective:** demand charges, time-of-day tariffs, grid carbon intensity, or solar surplus once capacity passes about 1.45 times today's.

## Supporting sections

The evaluators' mid-eval suggestions each become a part of the scheduler story rather than a separate to-do.

**Campus energy picture (2025, 15-minute data)**

- Solar covers about 18% of demand: about 2,450 kWh/day of solar against about 11,100 kWh/day of grid draw. By month it ranges from 14% (April) to 23% (December).
- At noon solar is about 68% of grid draw; it is gone by 17:00, when grid draw peaks at about 605 kW.
- Net zero would need about 5.5 times current solar, plus net metering for midday export, before any storage.
- Present this as coverage and the gap to close, not as a "maximum consumption" limit.

**Fault detection (the evaluators' anomaly question)**

- Clear-sky index per solar node (pvlib, no weather data needed), compared against the median of its peers. All nodes dip together = cloud; one node collapses = fault.
- Define anomaly types: sensor faults (frozen, dropout), equipment faults (one node diverging), and unusual but real events (holidays, heatwaves, outages), which are not faults.
- Evaluate on faults injected into real 2025 solar data, reported separately from the 234 synthetic scenario events (which are mostly grid and building events, not solar faults).
- Detect frozen values at the native data rate, not in 15-minute residuals.

**Forecasting, feature importance and distillation**

- The forecasters now have a job: their accuracy feeds schedule quality, so the value of forecast accuracy can be measured.
- SHAP on LightGBM, with features grouped by node or quantity, since the 58 inputs are heavily correlated.
- Distil LightGBM into an LSTM, CNN-LSTM and a smaller LSTM, then quantise to INT8. Frame it as "can a small NPU-friendly student match the teacher?", with a real answer ready for "why not run LightGBM on the CPU?"

**CPU vs NPU**

- Report accuracy, latency and energy per inference together, comparing INT8 on the NPU with FP32 (and ideally INT8) on the CPU.
- Measure energy with an inline USB-C power meter, idle subtracted, over a long loop.
- Report the board's own daily kWh against what it helps save; at one inference per 15 minutes, idle draw dominates.

**Water**

- PH01 is Manjeera (municipal), PH02 to PH06 are borewells. Note that the data only has PH01 to PH04, and PH02 has about 3 days.
- A water balance, not a model: borewell extraction against a recharge estimate using CGWB methodology, with the caveat that the aquifer is shared with the rest of Gachibowli.
- Pumping energy per kL (borewell vs Manjeera) links water to the energy sections and to the scheduler.

## Fixes before anything else

These are quick and remove things an examiner would catch.

- [ ] Rename the models' "Demand" to "KH04 room load": it is the three KH04 room meters, about 169 kWh/day or 1.5% of campus load.
- [ ] Correct the grid gap figure: with solar behind the meter, the grid still supplies about 11,100 kWh/day, not 9,100.
- [ ] Decide on the 3 solar nodes left out of the 18% (including SL-PL96, from June 2025), or state why they are excluded.
- [ ] State how missing daytime solar readings were filled; zero-filling understates solar.
- [ ] Note that KH95 (about 11 kWh/day) is a sub-circuit, not the KH building meter.
- [ ] Drop the 5 to 9 March V1.0.0 water test rows and check columns across firmware changes (SL-VN03 on 3 April, PH flow nodes in August).

## Open questions and work order

**For estates / facilities**

- [ ] Is solar behind the substation meter? The 10:00 to 13:00 grid plateau suggests yes; if metered separately, coverage would be about 22%.
- [ ] How is the campus billed: demand charges, contract demand, kVAh, time-of-day rates?
- [ ] Pump power ratings and tank capacities.
- [ ] Node-to-pump-house mapping (check `metadata.xlsx`), and where PH05 and PH06 are.
- [ ] Date of the final evaluation, which sets how many distillation students fit.

**Work order**

1. Fixes above (a day or two).
2. CPU vs NPU measurement, since the board is set up (a few days).
3. Pump scheduler as an LP / MILP with placeholder ranges, counterfactual on 2025 data.
4. Fault injection plus the peer and clear-sky detector, with precision and recall.
5. LightGBM with grouped SHAP.
6. Distillation and INT8 quantisation sweep.
7. Second and third loads (AC pre-cooling, GPU jobs) through the same scheduler.
8. Campus energy picture and water balance, written up last.
