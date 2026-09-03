import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

if os.path.exists("EM_cleaned.parquet"):
    input_dir = "."
elif os.path.exists(os.path.join("..", "EM_cleaned.parquet")):
    input_dir = ".."
else:
    input_dir = "cleanedData"

output_dir = 'cleaned_comparison'
os.makedirs(output_dir, exist_ok=True)

em = pd.read_parquet(os.path.join(input_dir, 'EM_cleaned.parquet'))
sl = pd.read_parquet(os.path.join(input_dir, 'SL_cleaned.parquet'))
srem = pd.read_parquet(os.path.join(input_dir, 'SR-EM_cleaned.parquet'))

for df in [em, sl, srem]:
    if not isinstance(df.index, pd.DatetimeIndex) and 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

grid_col = 'GDPower_reconstructed_kW' if 'GDPower_reconstructed_kW' in em.columns else em.columns[0]
solar_col = '3_Active_power_kW' if '3_Active_power_kW' in sl.columns else sl.columns[0]
room_col = 'Room_Demand_kW' if 'Room_Demand_kW' in srem.columns else srem.columns[0]

df_net_zero = pd.DataFrame({
    'Grid_Draw_kW': em[grid_col],
    'Solar_Yield_kW': sl[solar_col],
    'Room_Demand_kW': srem[room_col]
}).dropna()

# Plot 1: 1-Year Balance
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df_net_zero.index, df_net_zero['Solar_Yield_kW'], label='Campus Solar Yield (kW)', color='orange', alpha=0.85, linewidth=1)
ax.plot(df_net_zero.index, df_net_zero['Room_Demand_kW'], label='2-Room Demand (kW)', color='blue', alpha=0.75, linewidth=1)
ax.plot(df_net_zero.index, df_net_zero['Grid_Draw_kW'], label='Campus Grid Draw (kW)', color='red', alpha=0.45, linewidth=0.8, linestyle='--')

ax.set_title('Cleaned 1-Year Campus Energy Overview (Active Power in kW)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel('Active Power (kW)', fontsize=11, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

save_path1 = os.path.join(output_dir, 'cleaned_1year_power_balance.png')
plt.savefig(save_path1, dpi=200)
plt.close(fig)
print(f"Saved: {save_path1}")

# Plot 2: Diurnal Profile
diurnal = df_net_zero.groupby(df_net_zero.index.hour).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(diurnal.index, diurnal['Solar_Yield_kW'], label='Solar Yield (kW)', color='orange', marker='o', linewidth=2)
ax.plot(diurnal.index, diurnal['Room_Demand_kW'], label='2-Room Demand (kW)', color='blue', marker='s', linewidth=2)
ax.plot(diurnal.index, diurnal['Grid_Draw_kW'], label='Grid Draw (kW)', color='red', marker='^', linestyle='--', linewidth=1.5)

ax.set_title('Cleaned Average 24-Hour Diurnal Energy Profile', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day (0–23)', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Active Power (kW)', fontsize=11, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()

save_path2 = os.path.join(output_dir, 'cleaned_average_24hour_diurnal.png')
plt.savefig(save_path2, dpi=200)
plt.close(fig)
print(f"Saved: {save_path2}")