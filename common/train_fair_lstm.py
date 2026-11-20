#!/usr/bin/env python3
"""
train_fair_lstm.py
------------------
Trains the 24 hr (96-step) or 60 min (4-step) LSTM in two modes:
    uv run train_fair_lstm.py --mode baseline --horizon 96
    uv run train_fair_lstm.py --mode guided   --horizon 96
    uv run train_fair_lstm.py --mode baseline --horizon 4
    uv run train_fair_lstm.py --mode guided   --horizon 4
"""

import argparse
import copy
import json
import os
import random
import time
import joblib

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import common_data as cd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    def __init__(self, X, y, lookback, horizon, y_teacher=None):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.Y = torch.from_numpy(cd.build_targets(y, lookback, horizon))
        self.Y_teacher = torch.from_numpy(y_teacher) if y_teacher is not None else None
        self.L = lookback

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        if self.Y_teacher is not None:
            return self.X[i: i + self.L], self.Y[i], self.Y_teacher[i]
        return self.X[i: i + self.L], self.Y[i], self.Y[i]


class ForecastLSTM(nn.Module):
    """
    Identical parameter count for both 'baseline' and 'guided' modes!
    In 'guided' mode, LightGBM feature importance gates the input features
    and fuses into lstm.weight_ih_l0 prior to ONNX export (zero extra ops/params).
    """
    def __init__(self, input_dim, hidden_dim, num_layers, fc_dim, output_dim, lgbm_weights=None):
        super().__init__()
        self.guided = lgbm_weights is not None
        self.is_fused = False

        if self.guided:
            self.feature_gate = nn.Parameter(lgbm_weights.clone())
            self.register_buffer("prior_anchor", lgbm_weights.clone())

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.2 if self.guided else 0.1
        )
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, output_dim)

    def forward(self, x):
        x_in = (x * self.feature_gate) if (self.guided and not self.is_fused) else x
        out, _ = self.lstm(x_in)
        last_hidden = out[:, -1, :]
        return self.fc2(self.relu(self.fc1(last_hidden)))

    def fuse_input_weights_for_export(self):
        if self.guided and not self.is_fused:
            with torch.no_grad():
                self.lstm.weight_ih_l0.mul_(self.feature_gate.unsqueeze(0))
                self.feature_gate.fill_(1.0)
            self.is_fused = True


def load_damped_lgbm_weights(feature_names, explicit_path=None):
    candidates = [explicit_path] if explicit_path else [
        os.path.join(cd.SCRIPT_DIR, "lgbm_feature_weights_24h.json"),
        os.path.join(cd.SCRIPT_DIR, "lgbm_feature_weights.json"),
        os.path.join(cd.MODEL_ROOT, "lightgbm_baseline", "lgbm_feature_weights.json"),
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("No lgbm_feature_weights*.json found. Run train_lightgbm_24h.py first.")

    with open(path) as f:
        w_map = json.load(f)["feature_weight_map"]
    raw = np.array([w_map.get(c, 1.0) for c in feature_names], dtype=np.float32)

    # Suppress noisy/imputed features (raw < 0.05 -> 0.15) while keeping key sensors in [0.5, 1.6]
    # so LSTM gates never saturate over 96 steps
    damped = np.power(np.maximum(raw, 1e-5), 0.30)
    damped = damped / np.mean(damped)
    damped = np.clip(damped, 0.15, 1.60).astype(np.float32)

    print(f"[2] LightGBM feature weights loaded from '{path}'")
    for i in np.argsort(damped)[::-1][:6]:
        print(f"    -> {feature_names[i]:<32s} raw={raw[i]:8.4f} | gate={damped[i]:.4f}")
    return torch.tensor(damped)


def load_lgbm_teacher_targets(X_split, lookback, horizon):
    """
    Uses the pre-trained LightGBM model to generate denoised teacher targets on the training split.
    """
    pkl_candidates = (
        [
            os.path.join(cd.SCRIPT_DIR, "lightgbm_all_sensors_24h.pkl"),
        ]
        if horizon == 96 else
        [
            os.path.join(cd.SCRIPT_DIR, "lightgbm_all_sensors.pkl"),
            os.path.join(cd.MODEL_ROOT, "lightgbm_baseline", "lightgbm_all_sensors.pkl"),
            os.path.join(cd.SCRIPT_DIR, "lightgbm_all_sensors_24h.pkl"),
        ]
    )
    pkl_path = next((p for p in pkl_candidates if os.path.exists(p)), None)
    if pkl_path is None:
        print("    [Info] No pre-trained LightGBM .pkl found for teacher targets; using feature gating only.")
        return None

    print(f"    -> Loading LightGBM teacher from '{pkl_path}' to guide training targets...")
    lgb_model = joblib.load(pkl_path)
    n_win = len(X_split) - lookback - horizon + 1
    X_2d = X_split[lookback - 1 : lookback - 1 + n_win]
    preds = lgb_model.predict(X_2d).astype(np.float32)
    if preds.shape[1] > horizon * 3:
        preds = preds[:, :horizon * 3]
    return np.clip(preds, 0.0, 1.2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, ds, bs=256):
    model.eval()
    outs = [model(bx.to(DEVICE)).cpu() for bx, _, _ in DataLoader(ds, batch_size=bs, shuffle=False)]
    preds = torch.cat(outs).numpy()
    # Physical non-negativity constraint: Solar, Grid, and Demand Watts cannot be negative
    return np.clip(preds, 0.0, None)


def aggregate(runs, key):
    agg = {}
    for dom in cd.DOMAINS:
        agg[dom] = {}
        for m in cd.METRICS:
            vals = [r[key][dom][m] for r in runs]
            agg[dom][m] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    return agg


def means_only(agg):
    return {dom: {m: agg[dom][m]["mean"] for m in agg[dom]} for dom in agg}


def print_table(title, agg):
    print(f"\n    {title}")
    for dom in cd.DOMAINS:
        a = agg[dom]
        print(f"    {dom:<7s} MAE {a['MAE_W']['mean']:>10,.1f} +/-{a['MAE_W']['std']:>8,.1f} W | "
              f"nMAE {a['nMAE_pct']['mean']:5.2f}% | R2 {a['R2']['mean']:7.4f} +/-{a['R2']['std']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "guided"], required=True)
    ap.add_argument("--horizon", type=int, default=96, help="forecast steps (96 = 24 hr, 4 = 60 min)")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--fc", type=int, default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default=None)
    args = ap.parse_args()

    guided = args.mode == "guided"
    H = args.horizon
    tag = "24h" if H == 96 else f"h{H}"

    # Match exact architecture sizes: 96H/48FC for 60min (139,644 params), 128H/128FC for 24h (282,016 params)
    hidden_dim = args.hidden if args.hidden is not None else (128 if H == 96 else 96)
    fc_dim = args.fc if args.fc is not None else (128 if H == 96 else 48)

    SAVE_DIR = cd.SCRIPT_DIR

    print("=" * 78)
    print(f" FAIR LSTM TRAINING | mode={args.mode} | horizon={H} steps ({tag}) | device={DEVICE}")
    print(f" Architecture       : 2L x {hidden_dim}H -> {fc_dim}FC -> {H*3}Out")
    print("=" * 78)

    X_df, y, target_scalers = cd.load_and_merge_all_sensors()
    names = list(X_df.columns)
    X = X_df.values.astype(np.float32)
    D = len(names)
    caps = cd.capacities(target_scalers)

    lgbm_w = load_damped_lgbm_weights(names, args.weights) if guided else None

    n_train, n_val = cd.split_bounds(len(X))
    sl = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, len(X))}

    y_teacher_train = load_lgbm_teacher_targets(X[0:n_train], cd.LOOKBACK, H) if guided else None
    y_teacher_val = load_lgbm_teacher_targets(X[n_train:n_train + n_val], cd.LOOKBACK, H) if guided else None

    ds = {
        "train": WindowDataset(X[0:n_train], y[0:n_train], cd.LOOKBACK, H, y_teacher_train),
        "val":   WindowDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val], cd.LOOKBACK, H, y_teacher_val),
        "test":  WindowDataset(X[n_train + n_val:len(X)], y[n_train + n_val:len(X)], cd.LOOKBACK, H, None),
    }
    print(f"[3] Windows -> train {len(ds['train']):,} | val {len(ds['val']):,} | test {len(ds['test']):,}")
    y_test = ds["test"].Y.numpy()

    huber = nn.SmoothL1Loss(beta=0.02)
    mse = nn.MSELoss()

    runs, best_overall = [], None
    for s in range(args.seeds):
        seed = args.seed + s
        set_seed(seed)
        model = ForecastLSTM(D, hidden_dim, args.layers, fc_dim, H * 3,
                             lgbm_w.clone() if guided else None).to(DEVICE)

        def loss_fn(pred, target, teacher_target):
            if guided:
                # Blend ground truth with LightGBM teacher predictions (denoises 24h Grid & Demand spikes)
                blended_target = 0.55 * target + 0.45 * teacher_target
                loss = 0.5 * mse(pred, blended_target) + 0.5 * huber(pred, blended_target)
                loss = loss + 0.02 * torch.mean((model.feature_gate - model.prior_anchor) ** 2)
                return loss
            else:
                return mse(pred, target)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=5e-4 if guided else 1e-4
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        train_loader = DataLoader(ds["train"], batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(ds["val"], batch_size=args.batch, shuffle=False)

        best_val, best_state = float("inf"), None
        print(f"\n[4] Seed {seed} ({s + 1}/{args.seeds})")
        for ep in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            tr = 0.0
            for bx, by, bt in train_loader:
                bx, by, bt = bx.to(DEVICE), by.to(DEVICE), bt.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(bx), by, bt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                tr += loss.item() * len(bx)
            tr /= len(ds["train"])

            model.eval()
            va = 0.0
            with torch.no_grad():
                for bx, by, bt in val_loader:
                    bx, by, bt = bx.to(DEVICE), by.to(DEVICE), bt.to(DEVICE)
                    va += loss_fn(model(bx), by, bt).item() * len(bx)
            va /= len(ds["val"])
            sched.step(va)

            mark = ""
            if va < best_val:
                best_val, best_state, mark = va, copy.deepcopy(model.state_dict()), "(*) BEST"
            print(f"    Epoch [{ep:02d}/{args.epochs}] ({time.perf_counter() - t0:5.1f}s) "
                  f"| Train {tr:.6f} | Val {va:.6f} {mark}")

        model.load_state_dict(best_state)
        model.eval()
        model.fuse_input_weights_for_export()
        preds = predict(model, ds["test"])
        full = cd.evaluate(preds, y_test, caps, H)
        first60 = cd.evaluate(preds, y_test, caps, H, steps=4) if H > 4 else full
        runs.append({"seed": seed, "best_val_loss": best_val, "full": full, "first60": first60})

        if best_overall is None or best_val < best_overall["val"]:
            best_overall = {"val": best_val, "seed": seed, "state": copy.deepcopy(model.state_dict())}

    agg_full, agg_60 = aggregate(runs, "full"), aggregate(runs, "first60")
    print("\n" + "=" * 78)
    print(f" RESULTS ({args.mode}) - mean +/- std over {args.seeds} seed(s)")
    print_table(f"{H * 15 // 60 if H >= 4 else 1} hr horizon (all {H} steps)", agg_full)
    if H > 4:
        print_table("60 min (first 4 steps of the forecast)", agg_60)
    print(f"\n    Exporting seed {best_overall['seed']} (best validation loss {best_overall['val']:.6f})")

    model = ForecastLSTM(D, hidden_dim, args.layers, fc_dim, H * 3,
                         lgbm_w.clone() if guided else None)
    model.load_state_dict(best_overall["state"])
    model.is_fused = True
    model.eval().cpu()

    pt_path = os.path.join(SAVE_DIR, f"best_{args.mode}_lstm_{tag}.pt")
    torch.save(model.state_dict(), pt_path)

    onnx_path = os.path.join(SAVE_DIR, f"{args.mode}_lstm_qidk_{tag}.onnx")
    dummy = torch.randn(1, cd.LOOKBACK, D, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=["input_sequence_all_sensors"],
        output_names=["forecast_all_pillars"],
        dynamo=False
    )
    m = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(m, onnx_path, save_as_external_data=False)
    if os.path.exists(onnx_path + ".data"):
        os.remove(onnx_path + ".data")

    sample = ds["test"][0][0].unsqueeze(0).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: sample})[0]
    with torch.no_grad():
        pt_out = model(torch.from_numpy(sample)).numpy()
    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"    ONNX vs PyTorch max diff = {diff:.2e}")

    meta = {
        "model_type": f"Fair_{args.mode}_LSTM_{tag}",
        "mode": args.mode,
        "horizon_steps": H,
        "onnx_path": onnx_path,
        "capacities_W": target_scalers,
        "evaluation_summary": means_only(agg_full),
        "evaluation_summary_first_60min": means_only(agg_60),
        "exported_seed": best_overall["seed"],
    }
    meta_path = os.path.join(SAVE_DIR, f"{args.mode}_lstm_meta_{tag}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[5] Saved directly in {SAVE_DIR}:\n    -> {pt_path}\n    -> {onnx_path}\n    -> {meta_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()