import os
import sys
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1. Unified Data Loading & Standardization Function
# -------------------------------------------------------------
def load_and_prep_series(filepath, power_col, is_solar=False):
    """
    Loads parquet file, converts Unix epoch timestamps, deduplicates, 
    resamples to 1-hour means, and scales Watts to kW per metadata.
    """
    df = pd.read_parquet(filepath, columns=['Timestamp', power_col])
    
    # Handle Unix Epoch integer timestamps (seconds)
    if pd.api.types.is_integer_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    df = df.sort_values('Timestamp').set_index('Timestamp')
    df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
    
    # Deduplicate minute-logs, then resample to 1-hour means
    dedup = df[[power_col]].groupby(level=0).mean()
    hourly = dedup.resample('1h').mean()
    
    # Scale Watts (W) to Kilowatts (kW)
    hourly[power_col] = hourly[power_col] / 1000.0
    
    # Solar Inverter Specific Adjustments
    if is_solar:
        # Invert negative sign (CT orientation fix)
        hourly[power_col] = hourly[power_col] * -1.0
        # Clamp nighttime sensor noise below hardware resolution (10W = 0.01 kW) to exact 0
        hourly.loc[hourly[power_col] < 0.010, power_col] = 0.0
        
    return hourly

# -------------------------------------------------------------
# 2. Dual-Logger Class for File & Terminal Output
# -------------------------------------------------------------
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# -------------------------------------------------------------
# 3. Load Master DataStreams (Executed Once)
# -------------------------------------------------------------
print("Loading and standardizing all telemetry streams to kW...")
em_h = load_and_prep_series('EM.parquet', 'GDPower')
sl_h = load_and_prep_series('SL.parquet', '3_Active_power', is_solar=True)
srem_h = load_and_prep_series('SR-EM.parquet', 'Power')

# Full Hourly Grid DataFrame (Includes NaNs for statistical missingness audit)
df_full = pd.DataFrame({
    'Grid_Draw_kW': em_h['GDPower'],
    'Solar_Yield_kW': sl_h['3_Active_power'],
    'Room_Demand_kW': srem_h['Power']
})

# Concurrent Master DataFrame (Dropped NaNs for valid Net-Zero calculations)
df_concurrent = df_full.dropna().copy()

# Redirect standard output to both terminal and data_summary.txt
log_filename = "data_summary.txt"
sys.stdout = DualLogger(log_filename)

# =============================================================
# PART 1: Pandas EDA Statistical Summary
# =============================================================
print("==================================================================")
print("       PANDAS DATA FRAME COMPREHENSIVE EDA (kW SCALED)            ")
print("==================================================================")

print("\n--- 1. SHAPE & INDEX INFO ---")
print(f"Index Range: {df_full.index.min()} to {df_full.index.max()}")
print(f"Total Rows: {len(df_full):,} | Total Columns: {len(df_full.columns)}")

print("\n--- 2. MISSINGNESS AUDIT (df.isna() / df.notna()) ---")
null_counts = df_full.isna().sum()
valid_counts = df_full.notna().sum()
null_percents = df_full.isna().mean() * 100

missing_df = pd.DataFrame({
    'Valid_Count (notna)': valid_counts,
    'Missing_Count (isna)': null_counts,
    'Missing_Percent (%)': null_percents.round(2)
})
print(missing_df.to_string())

print("\n--- 3. DESCRIPTIVE STATISTICS IN kW (df.describe()) ---")
desc = df_full.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
desc['skewness'] = df_full.skew()
desc['kurtosis'] = df_full.kurtosis()
print(desc[['mean', 'std', 'min', '1%', '50%', '99%', 'max', 'skewness']].round(3).to_string())

print("\n--- 4. ZERO-VALUE & INACTIVITY AUDIT ---")
zero_counts = (df_full == 0).sum()
zero_percents = (df_full == 0).mean() * 100
zero_df = pd.DataFrame({
    'Zero_Value_Count': zero_counts,
    'Zero_Percent (%)': zero_percents.round(2)
})
print(zero_df.to_string())

print("\n--- 5. PEARSON CORRELATION MATRIX (df.corr()) ---")
print(df_full.corr().round(4).to_string())

# =============================================================
# PART 2: Net-Zero & Energy Balance Analysis
# =============================================================
df_concurrent['Net_Power_kW'] = df_concurrent['Solar_Yield_kW'] - df_concurrent['Room_Demand_kW']

total_solar_kwh = df_concurrent['Solar_Yield_kW'].sum()
total_room_kwh = df_concurrent['Room_Demand_kW'].sum()
total_grid_kwh = df_concurrent['Grid_Draw_kW'].sum()

surplus_hours = (df_concurrent['Net_Power_kW'] > 0).sum()
deficit_hours = (df_concurrent['Net_Power_kW'] < 0).sum()
balanced_hours = (df_concurrent['Net_Power_kW'] == 0).sum()

df_concurrent['Direct_Solar_Used_kW'] = np.minimum(df_concurrent['Solar_Yield_kW'], df_concurrent['Room_Demand_kW'])
total_direct_solar_kwh = df_concurrent['Direct_Solar_Used_kW'].sum()

self_consumption_rate = (total_direct_solar_kwh / total_solar_kwh * 100) if total_solar_kwh > 0 else 0
solar_coverage_ratio = (total_direct_solar_kwh / total_room_kwh * 100) if total_room_kwh > 0 else 0

print("\n==================================================================")
print("              CAMPUS NET-ZERO & ENERGY BALANCE REPORT             ")
print("==================================================================")
print(f"Analysis Window: {df_concurrent.index.min()} to {df_concurrent.index.max()}")
print(f"Total Evaluated Hours: {len(df_concurrent):,} hours")

print("\n--- 6. TOTAL ANNUAL ENERGY TOTALS (kWh) ---")
print(f"Total Solar Generation:   {total_solar_kwh:12,.2f} kWh")
print(f"Total Room Demand:       {total_room_kwh:12,.2f} kWh")
print(f"Total Campus Grid Draw:   {total_grid_kwh:12,.2f} kWh")

print("\n--- 7. NET POWER BALANCE HOURLY BREAKDOWN ---")
print(f"Deficit Hours (Demand > Solar) : {deficit_hours:5,} hrs ({deficit_hours/len(df_concurrent)*100:.1f}%)")
print(f"Surplus Hours (Solar > Demand) : {surplus_hours:5,} hrs ({surplus_hours/len(df_concurrent)*100:.1f}%)")
print(f"Zero Net Hours                 : {balanced_hours:5,} hrs ({balanced_hours/len(df_concurrent)*100:.1f}%)")

print("\n--- 8. KEY NET-ZERO PERFORMANCE METRICS ---")
print(f"Solar Self-Consumption Rate: {self_consumption_rate:.2f}%")
print(f"Room Solar Coverage Ratio:   {solar_coverage_ratio:.2f}%")
print(f"Max Hourly Solar Surplus:    {df_concurrent['Net_Power_kW'].max():.2f} kW")
print(f"Max Hourly Power Deficit:    {abs(df_concurrent['Net_Power_kW'].min()):.2f} kW")

print("\n==================================================================")
print(f"Pipeline Finished Successfully! Full report saved to: {log_filename}")
print("==================================================================")