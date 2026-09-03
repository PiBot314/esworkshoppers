import sys
import os
import pandas as pd
import numpy as np

# Dual logger for console and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("diagnostic.txt")

def inspect_dataset(file_path, label):
    print("=" * 80)
    print(f" DATASET AUDIT: {label} ({file_path})")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found.\n")
        return

    df = pd.read_parquet(file_path)
    total_rows = len(df)
    print(f"Total Rows: {total_rows:,} | Total Columns: {len(df.columns)}")
    
    # 1. Non-zero, missingness, and data-type metrics
    stats = []
    for col in df.columns:
        s = df[col]
        valid_cnt = s.notna().sum()
        missing_cnt = s.isna().sum()
        missing_pct = (missing_cnt / total_rows) * 100
        
        # Check numeric activity
        if pd.api.types.is_numeric_dtype(s):
            nonzero_cnt = (s != 0).sum()
            nonzero_pct = (nonzero_cnt / total_rows) * 100
            min_val = s.min()
            max_val = s.max()
            median_val = s.median()
        else:
            nonzero_cnt = valid_cnt
            nonzero_pct = (nonzero_cnt / total_rows) * 100
            min_val, max_val, median_val = "-", "-", "-"

        stats.append({
            'Column': col,
            'Dtype': str(s.dtype),
            'NonZero_Count': nonzero_cnt,
            'Active(%)': round(nonzero_pct, 1),
            'Missing(%)': round(missing_pct, 1),
            'Min': round(min_val, 2) if isinstance(min_val, (int, float)) else min_val,
            'Median': round(median_val, 2) if isinstance(median_val, (int, float)) else median_val,
            'Max': round(max_val, 2) if isinstance(max_val, (int, float)) else max_val
        })

    summary_df = pd.DataFrame(stats)
    print("\n--- PARAMETER ACTIVITY & NUMERICAL RANGES ---")
    print(summary_df.to_string(index=False))

    # 2. Sample 3 rows with actual non-zero activity
    print("\n--- SAMPLE ROW 0 ---")
    print(df.iloc[0].to_dict())

    # Find a row where numeric columns are actively populated
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        active_mask = (df[num_cols] > 0).sum(axis=1)
        if (active_mask > 0).any():
            best_row_idx = active_mask.idxmax()
            print(f"\n--- SAMPLE HIGHLY ACTIVE ROW (Index: {best_row_idx}) ---")
            print(df.loc[best_row_idx].to_dict())

    print("\n" + "#" * 80 + "\n")

# Run inspection across the three sources
files = [
    ("EM.parquet", "CAMPUS GRID METER (EM)"),
    ("SL.parquet", "SOLAR INVERTER (SL)"),
    ("SR-EM.parquet", "ROOM SUBMETER (SR-EM)")
]

for filename, label in files:
    # Check current folder, then fallback to trimmed version
    target = filename if os.path.exists(filename) else filename.replace(".parquet", "_trimmed.parquet")
    inspect_dataset(target, label)

print("Diagnostic complete! Saved full report to: diagnostic.txt")