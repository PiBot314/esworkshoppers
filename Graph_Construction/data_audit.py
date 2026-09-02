import os
import pandas as pd
import missingno as msno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create output folder
output_dir = 'audit_missingno'
os.makedirs(output_dir, exist_ok=True)

file_mapping = {
    'EM': ('EM.parquet', ['GDPower', 'GDTotal Energy', 'GDPower Factor', 'T1Current']),
    'SL': ('SL.parquet', ['3_Active_power', 'ActivePower', 'Eac_today', 'Eac_total', 'Power Factor', 'PV1Power', 'PV2Power']),
    'SREM': ('SR-EM.parquet', ['Power', 'Energy', 'Power Factor'])
}

def scale_energy_column(col_name, series):
    """
    Applies Wh -> kWh conversion for raw Energy metrics per metadata.
    """
    if 'Energy' in col_name and col_name not in ['Eac_today', 'Eac_total']:
        return series / 1000.0
    return series

for name, (filepath, cols) in file_mapping.items():
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}: File not found.")
        continue

    print(f"Generating metadata-aligned missingno visualizations for {name}...")
    
    # 1. Load Raw File
    df = pd.read_parquet(filepath)
    
    # Handle Unix Epoch integer timestamps (60s Unix seconds per metadata)
    if pd.api.types.is_integer_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    df = df.sort_values('Timestamp').set_index('Timestamp')
    
    # Filter selected columns
    valid_cols = [c for c in cols if c in df.columns]
    df_selected = df[valid_cols].copy()
    
    # Coerce columns to numeric and scale Wh to kWh
    for col in df_selected.columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        df_selected[col] = scale_energy_column(col, df_selected[col])

    # 2. Deduplicate Index (60s raw intervals)
    df_dedup = df_selected.groupby(level=0).mean()
    
    # 3. Resample to 1-hour means and reindex on a complete hourly grid
    df_hourly = df_dedup.resample('1h').mean()
    full_grid = pd.date_range(start=df_hourly.index.min(), end=df_hourly.index.max(), freq='1h')
    df_grid = df_hourly.reindex(full_grid)
    
    # --- Chart 1: Missing Matrix (Timeline View) ---
    fig, ax = plt.subplots(figsize=(13, 6))
    msno.matrix(df_grid, sparkline=False, fontsize=10, ax=ax)
    
    ax.set_title(f"{name} Dataset - Missing Data Timeline Matrix (1-Hour Grid)", fontsize=14, fontweight='bold', pad=25)
    ax.set_xlabel("Sensor / Telemetry Features", fontsize=11, fontweight='bold', labelpad=15)
    ax.set_ylabel("Hourly Time Index (Start -> End of Year)", fontsize=11, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_missing_matrix.png"), dpi=200)
    plt.close(fig)

    # --- Chart 2: Missing Bar Chart (Completeness Count) ---
    fig, ax = plt.subplots(figsize=(11, 6))
    msno.bar(df_grid, fontsize=10, color='royalblue', ax=ax)
    
    ax.set_title(f"{name} Dataset - Valid Data Record Counts", fontsize=14, fontweight='bold', pad=25)
    ax.set_xlabel("Sensor / Telemetry Features", fontsize=11, fontweight='bold', labelpad=15)
    ax.set_ylabel("Total Number of Valid Non-Null Rows", fontsize=11, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_missing_bar.png"), dpi=200)
    plt.close(fig)

    # --- Chart 3: Missing Correlation Heatmap ---
    if df_grid.isna().sum().sum() > 0:
        try:
            fig, ax = plt.subplots(figsize=(9, 7))
            msno.heatmap(df_grid, fontsize=10, ax=ax)
            
            ax.set_title(f"{name} Dataset - Missingness Correlation Between Features", fontsize=13, fontweight='bold', pad=25)
            ax.set_xlabel("Telemetry Feature A", fontsize=11, fontweight='bold', labelpad=15)
            ax.set_ylabel("Telemetry Feature B", fontsize=11, fontweight='bold', labelpad=15)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_missing_heatmap.png"), dpi=200)
            plt.close(fig)
        except Exception:
            pass

    print(f"Saved metadata-corrected missingno charts for {name} in ./{output_dir}/")

print("\nAudit Complete! Updated plots are available in 'audit_missingno'.")