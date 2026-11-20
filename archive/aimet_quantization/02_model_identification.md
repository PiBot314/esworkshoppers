# 02 — Model Identification

## 1. Verdict

**All three ONNX models are multi-step, multi-output time-series *regression* (forecasting) networks: a stacked 2-layer unidirectional LSTM encoder followed by a 2-layer MLP head.** They are **not** anomaly detectors or classifiers. They output future power values, not an anomaly score or label.

Why this matters: you described the model as "designed to detect anomalous readings in sensors (faulty sensors or damage to panels)". The code doesn't do that. Anomaly detection *can* be built on top of these models **without retraining**, as a residual check (§4). That changes what "minimal loss of accuracy" means for quantization (doc 06 §3).

The sibling folder `../project/` contains a real anomaly detector: a dense autoencoder 20→32→16→8→16→32→20 with a reconstruction-error threshold of 14.79. It is already on the QIDK as `energy_autoencoder_int8.dlc`, quantized with `snpe-dlc-quant`. **You chose to keep it out of scope** and target the mideval LSTMs.

## 2. Architecture (read from the ONNX graphs, not the scripts)

All three models: ONNX IR 10, opset 18, exported by PyTorch 2.14 (dynamo exporter), static batch 1, `onnx.checker` passes.

```
input [1, 96, F]  (96 × 15-min steps = past 24 h, F features, all in [0,1])
  → Transpose → LSTM(hidden=H, forward)  → LSTM(hidden=H, forward)     # zero initial h/c
  → Gather(last timestep) → Gemm(H→M) → Relu → Gemm(M→O)
output [1, O]
```
ONNX op histogram (1 h model): `LSTM×2, Gemm×2, Relu×1, Slice×16, Transpose×4, Concat×4, Unsqueeze×4, Reshape×2, Gather×1`. The Slice/Concat ops come from PyTorch packing the gate weights (iofc→iofc reorder).

| Model file | F | H | M | O (meaning) | Params |
|---|---|---|---|---|---|
| `SL-PL00-00_lstm_qidk.onnx` | 10 | 64 | 32 | 4 = next 1 h of Palash plant power | 55,021 |
| `all_sensors_lstm_qidk.onnx` | 58 | 96 | 48 | 12 = 4 steps × (Solar, Grid, Demand) interleaved | 139,749 |
| `all_sensors_24h_lstm_qidk.onnx` | 58 | 128 | 128 | 288 = 96 steps × (Solar, Grid, Demand) | 282,153 |

* Inputs: per-feature MinMax-normalized [0,1] sensor channels, binary `was_imputed` flags (0/1), and 6 sin/cos time encodings in [−1,1]. The 58 features are listed in order in `all_sensors_qidk_meta.json`.
* Outputs: normalized targets. Multiply by `capacities_W` to get Watts. Outputs can be slightly negative (e.g. −0.10 on the 24 h sample), and the runners clamp solar to ≥0.
* Output layout: `reshape(H, 3)`, so index `[t, d]` with d ∈ {Solar, Grid, Demand}.

## 3. How the QAIRT converter sees it (checked with QAIRT 2.47)
`qairt-converter` **unrolls each LSTM over all 96 timesteps** (`unroll_lstm_time_steps=True` by default). The 1 h model becomes **192 `Lstm` ops + 390 `Reshape` + 2 `Split` + 2 `FullyConnected` + …**, about 12.8 M MACs. Both `Gemm`s map cleanly to `FullyConnected`. **No unsupported ops**, so no UDO/custom op is needed.

Quantization relevance:
* An LSTM's **recurrent state accumulates quantization error over 96 steps**. That's the main accuracy risk and the reason 16-bit activations (W8A16) are the expected sweet spot on HTP v75.
* Sigmoid/tanh gate outputs are bounded ([0,1] / [−1,1]), so they quantize well. The **cell state `c_t` is unbounded** and is the sensitive tensor.
* The input tensor mixes [0,1] features with [−1,1] time encodings. A single per-tensor input encoding over [−1,1] wastes one bit for most channels. That's acceptable at 16-bit and noticeable at 8-bit.

## 4. Anomaly detection on top of the forecaster (no retraining)
Residual-based detection, done entirely in post-processing:

```
r_t = actual_t − forecast_t          (per domain, in Watts, at +15 min)
anomaly if |r_t| > k · σ_hod(r)       σ estimated per hour-of-day on the validation split
```
* **Solar under-production** (persistent r < 0 in daylight) could mean panel damage or soiling, or an inverter fault.
* **Flat-line / zero / impossible values** while the model expects generation usually means a sensor fault.

Limitation to state honestly in the report: the multi-sensor models forecast **campus totals**, so they can say *that* solar is anomalous, not *which* plant. Only `SL-PL00-00_lstm_qidk.onnx` localizes to one plant. Localizing every plant would need per-node models (= training), which is out of scope.

Implication for quantization: the quantized model must preserve residual *distributions*, not just average error. The verification plan (doc 08) therefore compares **anomaly flags of INT vs FP32**, not only MAE/R².
