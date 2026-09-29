from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campus_energy_pipeline.chatbot_prompts import build_anomaly_prompt

ROOT = PROJECT_ROOT
DATA_PATH = ROOT / "artifacts" / "energy_15m.csv"
ANOMALY_PATH = ROOT / "artifacts" / "energy_anomalies.csv"

st.set_page_config(page_title="Campus Energy Monitor", layout="wide")
st.title("Campus Energy Monitoring Dashboard")

if not DATA_PATH.exists():
    st.warning("The processed dataset is not available yet. Run `python3 campus_energy_pipeline/data_pipeline.py` and `python3 campus_energy_pipeline/train_autoencoder.py` first.")
    st.stop()

energy = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
anomalies = pd.read_csv(ANOMALY_PATH) if ANOMALY_PATH.exists() else pd.DataFrame()

if "anomaly_flag" in energy.columns:
    anomaly_count = int(energy["anomaly_flag"].sum())
else:
    anomaly_count = int(anomalies.shape[0])

col1, col2, col3 = st.columns(3)
col1.metric("Latest 15m Load", f"{energy['B Power_mean'].iloc[-1]:.1f} kW")
col2.metric("Peak Power", f"{energy['B Power_max'].max():.1f} kW")
col3.metric("Anomalies", anomaly_count)

st.subheader("Realtime 15-minute energy trend")
trend = energy.set_index("timestamp")[["Voltage_mean", "B Power_mean", "R Power_mean", "Total Energy_mean"]]
st.line_chart(trend)

st.subheader("Detected anomaly narrative")
if not anomalies.empty:
    latest = anomalies.iloc[0].to_dict()
    st.write(build_anomaly_prompt(latest))
    st.dataframe(anomalies[["timestamp", "Voltage_max", "B Power_max", "R Power_max", "reconstruction_error"]].head(10), use_container_width=True)
else:
    st.info("No anomalies were detected in the current processed dataset.")

st.subheader("Operational guidance")
with st.expander("Interpretation notes"):
    st.markdown(
        """
        - The model flags the highest reconstruction-error windows as likely energy wastage anomalies.
        - Large power spikes or phase imbalance are prioritized because they usually correspond to avoidable HVAC, lighting, or equipment load events.
        - The dashboard is intended as a facilities decision aid, not a replacement for manual electrical tuning.
        """
    )
