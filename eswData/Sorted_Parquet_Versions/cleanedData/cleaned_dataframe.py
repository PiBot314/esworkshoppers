import os
import sys
import pandas as pd
import numpy as np

# Path resolution
if os.path.exists("EM_cleaned.parquet"):
    input_dir = "."
elif os.path.exists(os.path.join("..", "EM_cleaned.parquet")):
    input_dir = ".."
else:
    input_dir = "cleanedData"

# Dual logger for terminal and file output
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

output_txt = "cleaned_data_summary.txt"
sys.stdout = DualLogger(output_txt)

em_df = pd.read_parquet(os.path.join(input_dir, "EM_cleaned.parquet"))
sl_df = pd.read_parquet(os.path.join(input_dir, "SL_cleaned.parquet"))
srem_df = pd.read_parquet(os.path.join(input_dir, "SR-EM_cleaned.parquet"))

for df in [em_df, sl_df, srem_df]:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

datasets = [
    ("CAMPUS GRID (EM_cleaned.parquet)", em_df),
    ("SOLAR INVERTER (SL_cleaned.parquet)", sl_df),
    ("ROOM SUBMETER (SR-EM_cleaned.parquet)", srem_df)
]

print("==================================================================")
print("       CLEANED DATASETS COMPREHENSIVE STATISTICAL AUDIT           ")
print("==================================================================")

for title, df in datasets:
    print(f"\n--- {title} ---")
    print(f"Index Range: {df.index.min()} to {df.index.max()}")
    print(f"Total Rows: {len(df):,} | Total Columns: {len(df.columns)}")
    
    missing_df = pd.DataFrame({
        'Valid_Count': df.notna().sum(),
        'Missing_Count': df.isna().sum(),
        'Missing (%)': (df.isna().mean() * 100).round(2),
        'Zero_Count': (df == 0).sum(),
        'Zero (%)': ((df == 0).mean() * 100).round(2)
    })
    print("\n[Missingness & Zero-Value Profile]")
    print(missing_df.to_string())

    desc = df.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]).T
    desc['skewness'] = df.skew()
    print("\n[Descriptive Statistics]")
    print(desc[['mean', 'std', 'min', '1%', '50%', '99%', 'max', 'skewness']].round(3).to_string())

print("\n==================================================================")
print("             CAMPUS NET-ZERO & ENERGY BALANCE REPORT              ")
print("==================================================================")

grid_col = 'GDPower_reconstructed_kW' if 'GDPower_reconstructed_kW' in em_df.columns else em_df.columns[0]
solar_col = '3_Active_power_kW' if '3_Active_power_kW' in sl_df.columns else sl_df.columns[0]
room_col = 'Room_Demand_kW' if 'Room_Demand_kW' in srem_df.columns else srem_df.columns[0]

df_concurrent = pd.DataFrame({
    'Grid_Draw_kW': em_df[grid_col],
    'Solar_Yield_kW': sl_df[solar_col],
    'Room_Demand_kW': srem_df[room_col]
}).dropna()

df_concurrent['Net_Power_kW'] = df_concurrent['Solar_Yield_kW'] - df_concurrent['Room_Demand_kW']
total_solar_kwh = df_concurrent['Solar_Yield_kW'].sum()
total_room_kwh = df_concurrent['Room_Demand_kW'].sum()
total_grid_kwh = df_concurrent['Grid_Draw_kW'].sum()

surplus_hours = (df_concurrent['Net_Power_kW'] > 0).sum()
deficit_hours = (df_concurrent['Net_Power_kW'] < 0).sum()
balanced_hours = (df_concurrent['Net_Power_kW'] == 0).sum()

direct_solar_used = np.minimum(df_concurrent['Solar_Yield_kW'], df_concurrent['Room_Demand_kW'])
self_consumption_rate = (direct_solar_used.sum() / total_solar_kwh * 100) if total_solar_kwh > 0 else 0
coverage_ratio = (direct_solar_used.sum() / total_room_kwh * 100) if total_room_kwh > 0 else 0

print(f"Total Evaluated Hours: {len(df_concurrent):,} hours")
print(f"Total Solar Generation:   {total_solar_kwh:12,.2f} kWh")
print(f"Total Room Demand:        {total_room_kwh:12,.2f} kWh")
print(f"Total Campus Grid Draw:   {total_grid_kwh:12,.2f} kWh")
print(f"Deficit Hours (Demand > Solar) : {deficit_hours:5,} hrs ({deficit_hours/len(df_concurrent)*100:.1f}%)")
print(f"Surplus Hours (Solar > Demand) : {surplus_hours:5,} hrs ({surplus_hours/len(df_concurrent)*100:.1f}%)")
print(f"Solar Self-Consumption Rate: {self_consumption_rate:.2f}%")
print(f"Room Solar Coverage Ratio:   {coverage_ratio:.2f}%")
print(f"Max Hourly Solar Surplus:    {df_concurrent['Net_Power_kW'].max():.2f} kW")
print(f"Max Hourly Power Deficit:    {abs(df_concurrent['Net_Power_kW'].min()):.2f} kW")
print("==================================================================")
print(f"\nReport written to: {output_txt}")