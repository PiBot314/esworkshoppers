"""P1: approximate rebuild of model2's lstm_ready/ -> lstm_normalized/ from the raw per-node telemetry.

Stage A (reconstructed here, the original script was not available):
    raw rows -> column mapping (MAPPING) -> 15-min UTC mean -> ffill(limit=4) + was_imputed -> IST cyclic features
    -> data/lstm_ready/{node}.parquet + features.json
Stage B (original code, unmodified): model2/generate_lstm_dataset.build_normalized_dataset()
    -> data/lstm_normalized/{node}.parquet, scaler_params.json, features.json
"""
import json
import os
import sys
sys.dont_write_bytecode = True   # importing model2 modules must not leave __pycache__ in model2/ (read-only input)

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))             # mideval/
RAW = os.path.abspath(os.path.join(ROOT, "..", "nodes", "parquet_sorted"))
KH95_CSV = os.path.abspath(os.path.join(ROOT, "..", "raw_data", "EM-Kh95-00.csv"))
DATA = os.path.join(HERE, "..", "data")

# lstm_ready column -> function(raw_df) -> Series. Units as they were *before* generate_lstm_dataset
# (which applies abs(), kW->W for active_power_kW / reactive_power_kVAr, and renames freq_col_raw -> current_A).
def col(name):
    return lambda d: pd.to_numeric(d[name], errors="coerce")

def mean_of(*names):
    return lambda d: pd.concat([pd.to_numeric(d[n], errors="coerce") for n in names], axis=1).mean(axis=1)

V3_METER = {  # SL-PL00-00 / SL-NI01-00 style multifunction meters
    "active_power_kW": col("3_Active_power"),
    "reactive_power_kVAr": col("3_Reactive_power"),
    "voltage_avg_V": col("Voltage_avg"),
    "neutral_current_A": col("NPhaseLC"),
}
INVERTER = {
    "active_power_W": col("ActivePower"),
    "ac_voltage_avg_V": mean_of("Voltage1", "Voltage2", "Voltage3"),
    "pv_voltage_avg_V": mean_of(*[f"PV{i}Voltage" for i in range(1, 7)]),
}
SR_V2 = {  # firmware register transposition: Current=voltage, Frequency=current, Power Factor=frequency
    "power_W": col("Power"),
    "voltage_V": col("Current"),
    "freq_col_raw": col("Frequency"),
    "frequency_Hz": col("Power Factor"),
}

MAPPING = {
    "SR-EM-KH04-00": ("SR-EM", "power_W", {"power_W": col("Power")}),
    "SR-EM-KH04-01": ("SR-EM", "power_W", SR_V2),
    "SR-EM-KH04-02": ("SR-EM", "power_W", SR_V2),
    "EM-CR-KH95-00": ("CSV", "power_total_W", {
        "power_total_W": lambda d: d["R Power"] + d["Y Power"] + d["B Power"],
        "power_R_W": col("R Power"),
        "power_B_W": col("B Power"),
        "voltage_V": col("Voltage"),
        "power_factor": col("Power Factor"),
    }),
    "EM-CR-SB00-00": ("EM", "gd_power_W", {
        "gd_power_W": col("GDPower"),
        "t1_power_W": col("T1Power"),
        "t2_power_W": col("T2Power"),
        "hv_voltage_VLL": col("GDVLL"),
        "lt1_voltage_VLN": col("LT1VLN"),
        "lt2_voltage_VLN": col("LT2VLN"),
    }),
    "SL-PL00-00": ("SL", "active_power_kW", V3_METER),
    # SL-VN95-00: 3_Active_power peaks at 4.6 while 3_Reactive_power peaks at 159 (documented active peak:
    # 156.3 kW) -> the two registers are swapped on this meter.
    "SL-VN95-00": ("SL", "active_power_kW", dict(V3_METER, active_power_kW=col("3_Reactive_power"),
                                                 reactive_power_kVAr=col("3_Active_power"))),
    "SL-NI01-00": ("SL", "active_power_W", {
        "active_power_W": col("3_Active_power"),            # this meter already reports W
        "reactive_power_VAr": col("3_Reactive_power"),
        "voltage_avg_V": col("Voltage_avg"),
        "neutral_current_A": col("NPhaseLC"),
    }),
    "SL-NI03-00": ("SL", "active_power_W", INVERTER),
    "SL-NI03-01": ("SL", "active_power_W", INVERTER),
    "SL-VN02-01": ("SL", "active_power_W", INVERTER),
}


# The original aligned timeline starts at 2025-01-23 04:15 (progress.txt). Our SB00 series has one extra leading
# bin at 04:00 built from a single reading (04:14:09); dropping it reproduces the original split boundaries.
LEADING_BINS_TO_DROP = {"EM-CR-SB00-00": 1}


def load_raw(node, family):
    if family == "CSV":
        d = pd.read_csv(KH95_CSV)
        d.columns = [c.strip("﻿") for c in d.columns]
        ts = pd.to_datetime(d.iloc[:, 2], utc=True)                # first "Timestamp" column
        return d.set_index(ts).sort_index()
    d = pd.read_parquet(os.path.join(RAW, family, f"{node}.parquet"))
    # The parquet index is the server-ingest time; for EM-CR-SB00-00 it only spans Aug-Nov (back-filled
    # uploads). The device's Sensor_Timestamp is the true measurement time for every node.
    ts = pd.to_datetime(pd.to_numeric(d["Sensor_Timestamp"], errors="coerce"), unit="s", utc=True)
    d = d.set_index(ts)
    return d[d.index.notna() & (d.index >= "2024-12-31")].sort_index()


def cyclic(index_utc):
    ist = index_utc.tz_convert("Asia/Kolkata")
    hod = ist.hour + ist.minute / 60.0
    out = {}
    for name, val, period in (("hod", hod, 24.0), ("dow", ist.dayofweek, 7.0), ("doy", ist.dayofyear, 365.25)):
        out[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(val) / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(val) / period)
    return pd.DataFrame(out, index=index_utc)


def build_lstm_ready(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {"freq": "15min", "nodes": {}, "excluded": {}}
    for node, (family, target, cols) in MAPPING.items():
        raw = load_raw(node, family)
        frame = pd.DataFrame({name: fn(raw) for name, fn in cols.items()}, index=raw.index)
        res = frame.resample("15min").mean()
        if node in LEADING_BINS_TO_DROP:
            res = res.iloc[LEADING_BINS_TO_DROP[node]:]
        missing = res[target].isna()
        # Inverters stop transmitting after dusk; the original data carries their last reading through the
        # night (flagged was_imputed=1), which is what the deployed models saw. Meters use ffill(limit=4).
        # The DC string voltage is not carried (it is 0 at night in the original windows).
        if cols is INVERTER:
            pv = res["pv_voltage_avg_V"].ffill(limit=4)
            res = res.ffill()
            res["pv_voltage_avg_V"] = pv
        else:
            res = res.ffill(limit=4)
        res["was_imputed"] = (missing & res[target].notna()).astype(int)
        res = pd.concat([res, cyclic(res.index)], axis=1)
        res.to_parquet(os.path.join(out_dir, f"{node}.parquet"))
        meta["nodes"][node] = {"target": target}
        print(f"  lstm_ready {node:14s} rows={len(res):6d} [{res.index.min()} .. {res.index.max()}]")
    with open(os.path.join(out_dir, "features.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_normalized():
    sys.path.insert(0, os.path.join(ROOT, "model2"))
    import generate_lstm_dataset as g                     # original code, configured via its module globals
    cwd = os.getcwd()
    os.chdir(DATA)
    try:
        g.INPUT_DIR, g.OUTPUT_DIR, g.FORMATS = "lstm_ready", "lstm_normalized", ["parquet"]
        g.BY_VERSION_DIR = "by_version_unused"
        g.build_normalized_dataset()
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    build_lstm_ready(os.path.join(DATA, "lstm_ready"))
    build_normalized()
