from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
DATA_PATH = ARTIFACTS_DIR / "energy_15m.csv"
CALIBRATION_PATH = ARTIFACTS_DIR / "energy_calibration.npy"
THRESHOLD_PATH = MODEL_DIR / "energy_autoencoder_threshold.json"
ONNX_PATH = MODEL_DIR / "energy_autoencoder.onnx"

FEATURE_COLUMNS = [
    "Voltage_mean",
    "Voltage_max",
    "R Current_mean",
    "R Current_max",
    "Y Current_mean",
    "Y Current_max",
    "B Current_mean",
    "B Current_max",
    "R Power_mean",
    "R Power_max",
    "Y Power_mean",
    "Y Power_max",
    "B Power_mean",
    "B Power_max",
    "Frequency_mean",
    "Frequency_max",
    "Power Factor_mean",
    "Power Factor_max",
    "Total Energy_mean",
    "Total Energy_max",
]


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def load_processed_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Missing feature column: {col}")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable rows after cleaning in {path}")
    return df


def train_model(data: np.ndarray) -> tuple[Autoencoder, StandardScaler, np.ndarray, dict]:
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model = Autoencoder(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        recon = model(X_tensor)
        loss = criterion(recon, X_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} - loss: {loss.item():.6f}")

    with torch.no_grad():
        recon = model(X_tensor)
        mse = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

    threshold = float(np.quantile(mse, 0.995))
    metrics = {"threshold": threshold, "mean_score": float(mse.mean()), "p99": float(np.quantile(mse, 0.99))}
    return model, scaler, mse, metrics


def export_onnx(model: Autoencoder, x_sample: np.ndarray, output_path: Path) -> None:
    """Export the Autoencoder to ONNX with a fixed input shape (batch=1).

    SNPE often requires static input dimensions. Exporting without dynamic_axes
    (and with opset_version >= 18) produces a model with concrete input shape
    which `snpe-onnx-to-dlc` can consume without --input_dim arguments.
    """
    model.eval()
    # Ensure a concrete (1, input_dim) tensor for export
    dummy = torch.tensor(x_sample[:1], dtype=torch.float32)
    if dummy.ndim == 1:
        dummy = dummy.unsqueeze(0)
    else:
        dummy = dummy.reshape(1, -1)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["reconstructed"],
        # Do NOT use dynamic_axes here; export a static shape for SNPE converter
        opset_version=18,
    )
    print(f"Saved ONNX model to {output_path}")


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    df = load_processed_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    model, scaler, scores, metrics = train_model(X)
    model.eval()

    threshold = metrics["threshold"]
    df["reconstruction_error"] = scores
    df["anomaly_flag"] = df["reconstruction_error"] > threshold
    anomalies = df[df["anomaly_flag"]].copy()
    anomalies.to_csv(ARTIFACTS_DIR / "energy_anomalies.csv", index=False)

    calibration = scaler.transform(X[:256])
    np.save(CALIBRATION_PATH, calibration.astype(np.float32))

    export_onnx(model, calibration, ONNX_PATH)

    with THRESHOLD_PATH.open("w", encoding="utf-8") as handle:
        json.dump({
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "mean_score": metrics["mean_score"],
            "p99": metrics["p99"],
            "num_anomalies": int(anomalies.shape[0]),
        }, handle, indent=2)

    print(f"Threshold: {threshold:.6f}")
    print(f"Detected {int(anomalies.shape[0])} anomalies")
    print(f"Saved calibration dataset to {CALIBRATION_PATH}")
    print(f"Saved threshold metadata to {THRESHOLD_PATH}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
