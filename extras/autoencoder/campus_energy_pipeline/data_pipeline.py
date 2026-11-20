from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "EM-Kh95-00.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
PROCESSED_PATH = ARTIFACTS_DIR / "energy_15m.csv"
SUMMARY_PATH = ARTIFACTS_DIR / "energy_15m_summary.csv"


def load_energy_csv(csv_path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [
        "id",
        "node_id",
        "timestamp",
        "version",
        "raw_ts",
        "Voltage",
        "R Current",
        "Y Current",
        "B Current",
        "R Power",
        "Y Power",
        "B Power",
        "Frequency",
        "Power Factor",
        "Total Energy",
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in [
        "Voltage",
        "R Current",
        "Y Current",
        "B Current",
        "R Power",
        "Y Power",
        "B Power",
        "Frequency",
        "Power Factor",
        "Total Energy",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def build_15min_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "Voltage",
        "R Current",
        "Y Current",
        "B Current",
        "R Power",
        "Y Power",
        "B Power",
        "Frequency",
        "Power Factor",
        "Total Energy",
    ]
    frame = df.set_index("timestamp")[feature_cols].resample("15min").agg(["mean", "max"])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.ffill().bfill()
    frame.columns = [f"{name}_{stat}" for name, stat in frame.columns]
    frame = frame.reset_index()
    frame["energy_delta_kwh"] = frame["Total Energy_max"].diff().fillna(0) / 1000.0
    frame = frame.dropna(subset=[c for c in frame.columns if c != "timestamp"]).reset_index(drop=True)
    return frame


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    raw = load_energy_csv(DATA_PATH)
    aggregated = build_15min_aggregates(raw)
    aggregated.to_csv(PROCESSED_PATH, index=False)

    summary = aggregated.describe().T.reset_index()
    summary.columns = ["metric", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved {len(aggregated)} rows to {PROCESSED_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
