"""
generate_lstm_dataset.py
-------------------------
End-to-End Pipeline to produce normalized, LSTM-ready datasets from sensor telemetry.

Implemented Improvements:
1. Unit Standardization: All power and reactive power metrics converted to Watts (W) and VAr.
2. Sign Rectification: Inverted solar export meters (e.g., SL-PL00-00) and standby drift corrected via abs().
3. TA-Compliant NaN Resolution: Residual gaps (unreachable via P=IV during sensor power-off) filled with 0.
4. Telemetry Mismatch Resolution: Corrected register swaps in SR-EM-KH04-01 and 02 (current_A, voltage_V, frequency_Hz).
5. Comprehensive Normalization: MinMax scaling applied with scaler parameters exported for inverse-transformation.
6. Multi-Format Output: Exports both high-performance Parquet and human-readable CSV.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configuration
INPUT_DIR = "lstm_ready"
BY_VERSION_DIR = "by_version"
OUTPUT_DIR = "lstm_normalized"
FORMATS = ["parquet", "csv"]  # Generate both Parquet and CSV


def build_normalized_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    json_path = os.path.join(INPUT_DIR, "features.json")
    with open(json_path, "r") as f:
        meta = json.load(f)

    scalers_dict = {}
    normalized_meta = {
        "freq": meta.get("freq", "15min"),
        "normalization": "MinMaxScaler(feature_range=(0, 1))",
        "nodes": {},
        "excluded": meta.get("excluded", {})
    }

    print(f"Starting dataset generation -> Output folder: '{OUTPUT_DIR}'\n" + "="*60)

    for node_id, node_info in meta["nodes"].items():
        print(f"Processing Node: {node_id}")
        pq_path = os.path.join(INPUT_DIR, f"{node_id}.parquet")
        if not os.path.exists(pq_path):
            print(f"  [WARNING] File not found: {pq_path}. Skipping.")
            continue

        df = pd.read_parquet(pq_path)
        
        # 1. Sign Correction & Standby Noise Removal
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not any(col.startswith(p) for p in ["hod", "dow", "doy"]):
                df[col] = df[col].abs()

        # 2. Unit Standardization (kW -> W, kVAr -> VAr)
        if "active_power_kW" in df.columns:
            df["active_power_W"] = df["active_power_kW"] * 1000.0
            df.drop(columns=["active_power_kW"], inplace=True)
        if "reactive_power_kVAr" in df.columns:
            df["reactive_power_VAr"] = df["reactive_power_kVAr"] * 1000.0
            df.drop(columns=["reactive_power_kVAr"], inplace=True)

        # 3. Mislabeling Fix for SR-EM Nodes (KH04-01 & KH04-02)
        if "freq_col_raw" in df.columns:
            df.rename(columns={"freq_col_raw": "current_A"}, inplace=True)
        
        # Pull true grid frequency from by_version if not present
        if node_id in ["SR-EM-KH04-01", "SR-EM-KH04-02"] and "frequency_Hz" not in df.columns:
            bv_path = os.path.join(BY_VERSION_DIR, node_id, "V2.0.0", "V2.0.0.parquet")
            if os.path.exists(bv_path):
                bv_df = pd.read_parquet(bv_path)
                bv_df["Timestamp"] = pd.to_datetime(bv_df["Timestamp"])
                bv_df.set_index("Timestamp", inplace=True)
                freq_resampled = bv_df["Power Factor"].resample("15min").mean().ffill(limit=4)
                df["frequency_Hz"] = freq_resampled.reindex(df.index)

        # 4. Fill Remaining NaNs with 0 (as instructed by TA)
        df.fillna(0.0, inplace=True)

        # 5. Resolve Target and Feature Names
        orig_target = node_info["target"]
        target_name = "active_power_W" if orig_target == "active_power_kW" else orig_target
        feature_cols = [c for c in df.columns if c != "was_imputed"]

        # 6. Feature Normalization (MinMaxScaler to [0, 1])
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_norm = pd.DataFrame(
            scaler.fit_transform(df[feature_cols]),
            index=df.index,
            columns=feature_cols
        )
        
        # Keep binary indicator was_imputed unscaled
        if "was_imputed" in df.columns:
            df_norm["was_imputed"] = df["was_imputed"].astype(int)

        # Record scaler parameters for inverse transformation
        node_scalers = {}
        for idx, col in enumerate(feature_cols):
            node_scalers[col] = {
                "min": float(scaler.data_min_[idx]),
                "max": float(scaler.data_max_[idx]),
                "scale": float(scaler.scale_[idx])
            }
        scalers_dict[node_id] = node_scalers

        # 7. Export Outputs (Parquet and CSV)
        if "parquet" in FORMATS:
            out_pq = os.path.join(OUTPUT_DIR, f"{node_id}.parquet")
            df_norm.to_parquet(out_pq)
            print(f"  -> Saved Parquet: {out_pq}")
            
        if "csv" in FORMATS:
            out_csv = os.path.join(OUTPUT_DIR, f"{node_id}.csv")
            df_norm.to_csv(out_csv)
            print(f"  -> Saved CSV:     {out_csv}")

        # Update metadata
        normalized_meta["nodes"][node_id] = {
            "target": target_name,
            "target_raw_min": node_scalers[target_name]["min"],
            "target_raw_max": node_scalers[target_name]["max"],
            "features": list(df_norm.columns),
            "rows": len(df_norm),
            "null_count": int(df_norm.isna().sum().sum())
        }

    # Save scaler parameters and metadata
    scalers_file = os.path.join(OUTPUT_DIR, "scaler_params.json")
    with open(scalers_file, "w") as f:
        json.dump(scalers_dict, f, indent=2)
    print(f"\n[INFO] Scaler parameters saved to: {scalers_file}")

    meta_file = os.path.join(OUTPUT_DIR, "features.json")
    with open(meta_file, "w") as f:
        json.dump(normalized_meta, f, indent=2)
    print(f"[INFO] Metadata saved to: {meta_file}")
    print("\nDataset generation completed successfully.")


if __name__ == "__main__":
    build_normalized_dataset()
