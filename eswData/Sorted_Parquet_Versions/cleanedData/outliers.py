import os
import pandas as pd
import numpy as np

# Auto-detect paths
if os.path.exists("EM_cleaned.parquet"):
    input_dir = "."
    output_dir = os.path.join("..", "finalCleanedData")
elif os.path.exists(os.path.join("cleanedData", "EM_cleaned.parquet")):
    input_dir = "cleanedData"
    output_dir = "finalCleanedData"
else:
    input_dir = ".."
    output_dir = os.path.join("..", "finalCleanedData")

os.makedirs(output_dir, exist_ok=True)

# 1. EM (Campus Grid): Keep power dynamics untouched; fix ONLY the backward energy counter
print("Finalizing EM (Campus Grid)...")
em_df = pd.read_parquet(os.path.join(input_dir, "EM_cleaned.parquet"))
initial_energy = 2052.5  # Reference starting kWh from January
em_df["GDTotal_Energy_kWh"] = initial_energy + em_df["GDPower_reconstructed_kW"].cumsum()
em_df.to_parquet(os.path.join(output_dir, "EM_final.parquet"))

# 2. SL (Solar): Keep real weather/cloud dips untouched; recompute daily cumulative sums safely
print("Finalizing SL (Solar Inverter)...")
sl_df = pd.read_parquet(os.path.join(input_dir, "SL_cleaned.parquet"))
sl_df["Eac_today_kWh"] = sl_df["3_Active_power_kW"].groupby(sl_df.index.date).cumsum()
sl_df["Eac_total_kWh"] = np.maximum.accumulate(sl_df["Eac_total_kWh"].bfill())
sl_df.to_parquet(os.path.join(output_dir, "SL_final.parquet"))

# 3. SR-EM (Room Submeter): Preserve true low-occupancy baseloads and step-changes
print("Finalizing SR-EM (Room Submeter)...")
srem_df = pd.read_parquet(os.path.join(input_dir, "SR-EM_cleaned.parquet"))
srem_df["Energy_kWh"] = np.maximum.accumulate(srem_df["Energy_kWh"].bfill())
srem_df.to_parquet(os.path.join(output_dir, "SR-EM_final.parquet"))

print(f"\nSurgical finalization complete! Cleaned files saved to: '{os.path.abspath(output_dir)}'")