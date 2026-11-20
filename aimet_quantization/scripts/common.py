"""Shared paths, model registry and metrics for the AIMET/QIDK scripts."""
import json
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
AQ = os.path.abspath(os.path.join(HERE, ".."))                    # mideval/aimet_quantization
ROOT = os.path.abspath(os.path.join(AQ, ".."))                    # mideval
DEPLOY = os.path.join(ROOT, "model2", "qidk_deployment")
WINDOWS = os.path.join(AQ, "data", "windows")
OUT = os.path.join(AQ, "out")
CONFIG = os.path.join(AQ, "configs", "htp_v75_lstm_per_tensor.json")

MODELS = {
    "all_sensors_1h": dict(onnx="all_sensors_lstm_qidk.onnx", meta="all_sensors_qidk_meta.json",
                           horizon=4, domains=["Solar", "Grid", "Demand"]),
    "all_sensors_24h": dict(onnx="all_sensors_24h_lstm_qidk.onnx", meta="all_sensors_24h_qidk_meta.json",
                            horizon=96, domains=["Solar", "Grid", "Demand"]),
    "single_node": dict(onnx="SL-PL00-00_lstm_qidk.onnx", meta="qidk_model_meta.json",
                        horizon=4, domains=["Solar_PL00"]),
}
# Domains used for the headline accuracy (Demand excluded: its FP32 R^2 is already negative).
SCORED = {"all_sensors_1h": ["Solar", "Grid"], "all_sensors_24h": ["Solar", "Grid"], "single_node": ["Solar_PL00"]}


def onnx_path(name):
    return os.path.join(DEPLOY, MODELS[name]["onnx"])


def model_capacities(name):
    """Capacities the *deployed model* was trained with (to turn its outputs into Watts)."""
    meta = json.load(open(os.path.join(DEPLOY, MODELS[name]["meta"])))
    if name == "single_node":
        return [meta["scaler_max_W"] - meta["scaler_min_W"]]
    c = meta["capacities_W"]
    return [c["solar_max_W"], c["grid_max_W"], c["demand_max_W"]]


def data_capacities(name):
    """Capacities of the *rebuilt* data (to turn rebuilt normalized targets into Watts)."""
    info = json.load(open(os.path.join(WINDOWS, name, "info.json")))
    if name == "single_node":
        return [info["scaler_max_W"] - info["scaler_min_W"]]
    c = info["capacities_W"]
    return [c["solar_max_W"], c["grid_max_W"], c["demand_max_W"]]


def load(name, split):
    d = os.path.join(WINDOWS, name)
    X = np.load(os.path.join(d, f"X_{split}.npy")) if split != "calib" else np.load(os.path.join(d, "calib.npy"))
    y = np.load(os.path.join(d, f"y_{split}.npy")) if split in ("val", "test") else None
    return X, y


def input_name(session_or_model):
    if hasattr(session_or_model, "get_inputs"):
        return session_or_model.get_inputs()[0].name
    return session_or_model.graph.input[0].name


def run_session(session, X, batch=1):
    """Models are static batch 1."""
    name = input_name(session)
    return np.concatenate([session.run(None, {name: X[i:i + 1]})[0] for i in range(len(X))], axis=0)


def fp32_session(name):
    return ort.InferenceSession(onnx_path(name), providers=["CPUExecutionProvider"])


def to_watts(out, name, caps):
    """[N, H*D] normalized -> [N, H, D] Watts."""
    D = len(MODELS[name]["domains"])
    return out.reshape(len(out), -1, D) * np.asarray(caps)[None, None, :]


def r2(p, t):
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def domain_metrics(pred_n, y_n, name):
    """Metrics in Watts vs rebuilt ground truth. Predictions use the model's capacities, targets the data's."""
    P, T = to_watts(pred_n, name, model_capacities(name)), to_watts(y_n, name, data_capacities(name))
    caps = model_capacities(name)
    out = {}
    for d, dom in enumerate(MODELS[name]["domains"]):
        p, t = P[:, :, d], T[:, :, d]
        mae = float(np.mean(np.abs(p - t)))
        out[dom] = dict(MAE_W=mae, RMSE_W=float(np.sqrt(np.mean((p - t) ** 2))),
                        nMAE_pct=100 * mae / caps[d], R2=r2(p, t))
    return out


def headline_r2(metrics, name):
    return float(np.mean([metrics[d]["R2"] for d in SCORED[name]]))


def sqnr_db(ref, test):
    noise = np.sum((ref - test) ** 2)
    return float(100.0 if noise == 0 else min(100.0, 10 * np.log10(np.sum(ref ** 2) / noise)))


def fidelity(ref, test):
    return dict(SQNR_dB=sqnr_db(ref, test), max_abs=float(np.max(np.abs(ref - test))),
                mean_abs=float(np.mean(np.abs(ref - test))))
