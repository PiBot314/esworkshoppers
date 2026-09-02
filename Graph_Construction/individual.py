import os
import matplotlib
# Use non-interactive backend for headless execution
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

# Create output folder named "individual"
output_dir = 'individual'
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Master Column Selection Lists
# -------------------------------------------------------------
em_cols = ['GDPower', 'GDTotal Energy', 'GDPower Factor', 'T1Current']
sl_cols = ['3_Active_power', 'ActivePower', 'Eac_today', 'Eac_total', 
           'Power Factor', 'PF_Avg', 'PV1Power', 'PV2Power', 'PV3Power', 
           'PV4Power', 'PV5Power', 'PV6Power']
srem_cols = ['Power', 'Energy', 'Power Factor']

def get_unit_and_scaled_series(col_name, series):
    """
    Applies metadata rules:
    - Energy in Wh (EM/SREM) -> Divide by 1000 to convert to kWh.
    - Eac_today / Eac_total -> Already in kWh.
    - Power / PV -> Active Power in kW.
    - Current -> Amperes (A).
    - Power Factor -> Ratio [0.0 - 1.0].
    """
    if 'Energy' in col_name and col_name not in ['Eac_today', 'Eac_total']:
        return series / 1000.0, 'Energy (kWh)'
    elif 'Eac' in col_name:
        return series, 'Energy (kWh)'
    elif 'Power' in col_name or 'PV' in col_name:
        return series, 'Active Power (kW)'
    elif 'Current' in col_name:
        return series, 'Current (A)'
    elif 'Factor' in col_name or 'PF' in col_name:
        return series, 'Power Factor (0.0 - 1.0)'
    return series, col_name

def plot_all_parameters_separately(filepath, cols_to_plot, dataset_name):
    """
    Reads a Parquet file, applies metadata unit scaling, and saves an 
    individual standalone plot for EVERY parameter inside the 'individual' folder.
    """
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}: File not found.")
        return

    print(f"\n--- Generating Individual Plots for {dataset_name} ---")
    
    # Load dataset
    df = pd.read_parquet(filepath)
    
    # Convert integer epoch timestamp to datetime if stored as seconds
    if pd.api.types.is_integer_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    df = df.sort_values('Timestamp').set_index('Timestamp')
    
    # Convert and filter numeric columns
    numeric_cols = []
    for col in cols_to_plot:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_cols.append(col)
            
    if not numeric_cols:
        print(f"No valid numeric columns found for {dataset_name}.")
        return

    # Deduplicate timestamps by averaging duplicates before resampling
    df_dedup = df[numeric_cols].groupby(level=0).mean()

    # Resample to 1-hour means (aggregates 60s minute logs to hourly)
    df_hourly = df_dedup.resample('1h').mean()
    
    # Loop over every parameter and generate standalone plot
    for col in numeric_cols:
        series = df_hourly[col].dropna()
        if series.empty:
            print(f"Skipping {col}: No valid data points.")
            continue

        # Apply Wh -> kWh conversion and fetch unit label
        scaled_series, unit_label = get_unit_and_scaled_series(col, series)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(scaled_series.index, scaled_series.values, linewidth=1)
        
        # Enforce strict increasing Y-axis limits
        y_min = scaled_series.min()
        y_max = scaled_series.max()
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        else:
            padding = (y_max - y_min) * 0.05
            y_min -= padding
            y_max += padding

        ax.set_ylim(bottom=y_min, top=y_max)
        if ax.yaxis_inverted():
            ax.invert_yaxis()

        ax.set_title(f'{dataset_name} Dataset - Parameter: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(unit_label, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        # Save graph inside the 'individual' folder
        filename = f"{dataset_name}_{col.replace(' ', '_')}.png"
        filepath_save = os.path.join(output_dir, filename)
        plt.savefig(filepath_save, dpi=200)
        plt.close(fig)
        print(f"Saved: {filepath_save} (Y-Axis Unit: {unit_label})")

# -------------------------------------------------------------
# 2. Execute Plotting Across All Datasets
# -------------------------------------------------------------
plot_all_parameters_separately('EM.parquet', em_cols, 'EM')
plot_all_parameters_separately('SL.parquet', sl_cols, 'SL')
plot_all_parameters_separately('SR-EM.parquet', srem_cols, 'SREM')