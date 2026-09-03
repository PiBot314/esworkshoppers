import os
import pandas as pd
import numpy as np

output_dir = "cleanedData"
os.makedirs(output_dir, exist_ok=True)

def prep_time_index(df):
    """Parses timestamps, sorts, deduplicates, and resamples to 1h."""
    if 'Timestamp' in df.columns:
        if pd.api.types.is_integer_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    numeric_df = df.select_dtypes(include='number')
    dedup = numeric_df.groupby(level=0).mean()
    return dedup.resample('1h').mean()

# ==============================================================================
# 1. EM.parquet (Campus Grid)
# ==============================================================================
print("Cleaning EM.parquet...")
em_raw = prep_time_index(pd.read_parquet("EM.parquet"))
em_clean = pd.DataFrame(index=em_raw.index)

# A. Raw columns
em_clean['GDPower_raw'] = em_raw['GDPower'] / 1000.0  # W -> kW
em_clean['T1Current_raw'] = em_raw['T1Current']
em_clean['GDPower_Factor_raw'] = em_raw['GDPower Factor']
em_clean['GDTotal_Energy_kWh'] = em_raw['GDTotal Energy'] / 1000.0

# B. Reconstructed columns
active_alt_kw = (em_raw['T1Power'].fillna(0) + em_raw['T2Power'].fillna(0)) / 1000.0
busbar_kw = ((em_raw['LT1Current'].fillna(0) + em_raw['LT2Current'].fillna(0)) * 415.0 * 0.95 * np.sqrt(3)) / 1000.0

em_power = em_clean['GDPower_raw'].where(em_clean['GDPower_raw'] > 0.5, active_alt_kw)
em_power = em_power.where(em_power > 0.5, busbar_kw)

# Robust hour median calculation
valid_power = em_power[em_power > 0.5]
hour_medians_em = valid_power.groupby(valid_power.index.hour).median()
em_clean['GDPower_reconstructed_kW'] = em_power.where(
    em_power > 0.5, 
    pd.Series(em_power.index.hour.map(hour_medians_em), index=em_power.index)
).fillna(148.7)

# T1Current corrected
alt_curr = em_raw['LT1Current'].fillna(0) + em_raw['LT2Current'].fillna(0)
curr_candidate = em_raw['T1Current'].where((em_raw['T1Current'] > 0) & (em_raw['T1Current'] < 5000), alt_curr)
valid_curr = curr_candidate[curr_candidate > 0]
curr_medians = valid_curr.groupby(valid_curr.index.hour).median()

em_clean['T1Current_corrected_A'] = curr_candidate.where(
    curr_candidate > 0, 
    pd.Series(curr_candidate.index.hour.map(curr_medians), index=curr_candidate.index)
).fillna(25.0)

# Power Factor & Energy corrected
em_clean['GDPower_Factor_corrected'] = em_raw['GDPower Factor'].abs().where(
    em_raw['GDPower Factor'].abs() <= 1.0, np.nan
).interpolate(method='time').fillna(0.95).clip(0.0, 1.0)
em_clean['GDTotal_Energy_kWh'] = em_clean['GDTotal_Energy_kWh'].interpolate(method='time').bfill()

em_clean.to_parquet(os.path.join(output_dir, "EM_cleaned.parquet"))
print("-> EM_cleaned.parquet successfully updated.")

# ==============================================================================
# 2. SL.parquet (Solar Inverter)
# ==============================================================================
print("Cleaning SL.parquet...")
sl_raw = prep_time_index(pd.read_parquet("SL.parquet"))
sl_clean = pd.DataFrame(index=sl_raw.index)

raw_solar = sl_raw['3_Active_power'].abs()
night_mask = (sl_clean.index.hour < 5) | (sl_clean.index.hour >= 19)
raw_solar.loc[night_mask] = 0.0

day_mask = ~night_mask
valid_day_solar = raw_solar[day_mask & (raw_solar > 0.1)]
solar_day_medians = valid_day_solar.groupby(valid_day_solar.index.hour).median()

solar_imputed = raw_solar.copy()
day_dropouts = day_mask & (raw_solar <= 0.05)
solar_imputed.loc[day_dropouts] = pd.Series(
    raw_solar.index.hour.map(solar_day_medians), 
    index=raw_solar.index
).loc[day_dropouts]

sl_clean['3_Active_power_kW'] = solar_imputed.fillna(0.0).clip(lower=0.0)
sl_clean['PF_Avg'] = sl_raw['PF_Avg'].abs().where(sl_raw['PF_Avg'].abs() <= 1.0, np.nan).interpolate(method='time').fillna(0.98).clip(0.0, 1.0)
sl_clean['Eac_today_kWh'] = sl_raw['Eac_today'].interpolate(method='time').fillna(0.0)
sl_clean['Eac_total_kWh'] = sl_raw['Eac_total'].interpolate(method='time').bfill()

sl_clean.to_parquet(os.path.join(output_dir, "SL_cleaned.parquet"))
print("-> SL_cleaned.parquet successfully updated.")

# ==============================================================================
# 3. SR-EM.parquet (Room Submeter)
# ==============================================================================
print("Cleaning SR-EM.parquet...")
srem_raw = prep_time_index(pd.read_parquet("SR-EM.parquet"))
srem_clean = pd.DataFrame(index=srem_raw.index)

# Raw columns
srem_clean['Power_raw'] = srem_raw['Power']
srem_clean['Current_raw'] = srem_raw['Current']
srem_clean['Energy_raw'] = srem_raw['Energy']

# Corrected columns: Power in Watts -> kW
p_kw = srem_raw['Power'] / 1000.0
valid_room = p_kw[p_kw > 0.05]
room_medians = valid_room.groupby(valid_room.index.hour).median()

srem_clean['Room_Demand_kW'] = p_kw.where(
    p_kw > 0.05, 
    pd.Series(p_kw.index.hour.map(room_medians), index=p_kw.index)
).interpolate(method='time').fillna(1.8)

srem_clean['Voltage_derived_V'] = srem_raw['Current'].interpolate(method='time').fillna(230.0)
srem_clean['Energy_kWh'] = (srem_raw['Energy'] / 1000.0).interpolate(method='time').bfill()

srem_clean.to_parquet(os.path.join(output_dir, "SR-EM_cleaned.parquet"))
print("-> SR-EM_cleaned.parquet successfully updated.")

print(f"\nDone! All files generated in: {os.path.abspath(output_dir)}")