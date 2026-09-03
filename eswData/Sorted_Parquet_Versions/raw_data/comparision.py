import os
import matplotlib
# Use non-interactive backend for headless execution
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

# Create output folder named "comparison"
output_dir = 'comparison'
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Define Selected Columns Matrix
# -------------------------------------------------------------
em_cols = ['Timestamp', 'GDPower', 'GDTotal Energy', 'GDPower Factor', 'T1Current']
sl_cols = ['Timestamp', '3_Active_power', 'ActivePower', 'Eac_today', 'Eac_total', 
           'Power Factor', 'PF_Avg', 'PV1Power', 'PV2Power', 'PV3Power', 
           'PV4Power', 'PV5Power', 'PV6Power']
srem_cols = ['Timestamp', 'Power', 'Energy', 'Power Factor']

# -------------------------------------------------------------
# 2. Metadata-Aligned Parquet Loading & Hourly Resampling
# -------------------------------------------------------------
def load_and_resample(filepath, selected_cols):
    """
    Loads Parquet file according to metadata rules:
    1. Converts Unix epoch seconds to datetime if needed.
    2. Coerces numeric columns and deduplicates raw 60s timestamps.
    3. Resamples to 1-hour average active power (kW).
    """
    df = pd.read_parquet(filepath, columns=[c for c in selected_cols if c in pd.read_parquet(filepath).columns])
    
    # Handle integer epoch timestamp conversion (60s Unix seconds)
    if pd.api.types.is_integer_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    df = df.sort_values('Timestamp').set_index('Timestamp')
    
    # Coerce columns to numeric to handle string artifacts safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    numeric_df = df.select_dtypes(include=['number'])
    
    # Deduplicate timestamps by taking the mean of duplicates before resampling
    numeric_dedup = numeric_df.groupby(level=0).mean()
    
    # Resample to 1-hour means
    return numeric_dedup.resample('1h').mean()

print("Loading Parquet Datasets with Metadata Rules...")
em_hourly = load_and_resample('EM.parquet', em_cols)
sl_hourly = load_and_resample('SL.parquet', sl_cols)
srem_hourly = load_and_resample('SR-EM.parquet', srem_cols)

# Resolve solar active power column
sl_power_col = '3_Active_power' if '3_Active_power' in sl_hourly.columns else 'ActivePower'

# -------------------------------------------------------------
# 3. Combine Core Metrics into a Single Net-Zero DataFrame
# -------------------------------------------------------------
df_net_zero = pd.DataFrame({
    'Grid_Draw_Em': em_hourly['GDPower'],
    'Solar_Yield_Sl': sl_hourly[sl_power_col],
    'Room_Demand_Sr_em': srem_hourly['Power']
}).dropna()

print(f"Dataset Combined! Total Hourly Rows: {len(df_net_zero):,}")

# -------------------------------------------------------------
# 4. Pure Matplotlib Visualizations (Saved to 'comparison' folder)
# -------------------------------------------------------------

# PLOT 1: Full 1-Year Power Balance Overview
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df_net_zero.index, df_net_zero['Solar_Yield_Sl'], label='Campus Solar Yield (Sl)', color='orange', alpha=0.85, linewidth=1)
ax.plot(df_net_zero.index, df_net_zero['Room_Demand_Sr_em'], label='Indoor Room Load (Sr_em)', color='blue', alpha=0.75, linewidth=1)
ax.plot(df_net_zero.index, df_net_zero['Grid_Draw_Em'], label='Grid Draw (Em)', color='red', alpha=0.5, linewidth=0.8, linestyle='--')

ax.set_title('1-Year Campus Energy Overview (Active Power in kW)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel('Active Power (kW)', fontsize=11, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

save_path1 = os.path.join(output_dir, '1year_power_balance_overview.png')
plt.savefig(save_path1, dpi=200)
plt.close(fig)
print(f"Saved: {save_path1}")

# PLOT 2: Average 24-Hour Diurnal Energy Profile
diurnal = df_net_zero.groupby(df_net_zero.index.hour).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(diurnal.index, diurnal['Solar_Yield_Sl'], label='Solar Yield (Sl)', color='orange', marker='o', linewidth=2)
ax.plot(diurnal.index, diurnal['Room_Demand_Sr_em'], label='Room Demand (Sr_em)', color='blue', marker='s', linewidth=2)
ax.plot(diurnal.index, diurnal['Grid_Draw_Em'], label='Grid Draw (Em)', color='red', marker='^', linestyle='--', linewidth=1.5)

ax.set_title('Average 24-Hour Diurnal Energy Profile', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day (0–23)', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Active Power (kW)', fontsize=11, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()

save_path2 = os.path.join(output_dir, 'average_24hour_diurnal_profile.png')
plt.savefig(save_path2, dpi=200)
plt.close(fig)
print(f"Saved: {save_path2}")