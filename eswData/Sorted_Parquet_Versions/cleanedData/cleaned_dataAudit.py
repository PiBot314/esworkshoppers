import os
import sys
import pandas as pd
import missingno as msno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if os.path.exists("EM_cleaned.parquet"):
    input_dir = "."
elif os.path.exists(os.path.join("..", "EM_cleaned.parquet")):
    input_dir = ".."
else:
    input_dir = "cleanedData"

output_img_dir = 'cleaned_audit_missingno'
os.makedirs(output_img_dir, exist_ok=True)

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

output_txt = "cleaned_data_audit_summary.txt"
sys.stdout = DualLogger(output_txt)

file_mapping = {
    'EM': os.path.join(input_dir, 'EM_cleaned.parquet'),
    'SL': os.path.join(input_dir, 'SL_cleaned.parquet'),
    'SREM': os.path.join(input_dir, 'SR-EM_cleaned.parquet')
}

print("==================================================================")
print("               CLEANED DATASETS MISSING DATA AUDIT                ")
print("==================================================================")

for name, filepath in file_mapping.items():
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}: File not found.")
        continue

    print(f"\n--- AUDIT: {name} ({filepath}) ---")
    df = pd.read_parquet(filepath)
    if not isinstance(df.index, pd.DatetimeIndex) and 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

    total_rows = len(df)
    audit_data = []
    for col in df.columns:
        valid_cnt = df[col].notna().sum()
        missing_cnt = df[col].isna().sum()
        missing_pct = (missing_cnt / total_rows) * 100
        audit_data.append({
            'Column': col,
            'Total_Rows': total_rows,
            'Valid_Rows': valid_cnt,
            'Missing_Rows': missing_cnt,
            'Completeness (%)': round(100.0 - missing_pct, 2)
        })

    audit_table = pd.DataFrame(audit_data)
    print(audit_table.to_string(index=False))

    # Generate Chart 1: Matrix View
    fig, ax = plt.subplots(figsize=(12, 6))
    msno.matrix(df, sparkline=False, fontsize=10, ax=ax)
    ax.set_title(f"{name} Cleaned Dataset - Matrix (1-Hour Grid)", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    matrix_path = os.path.join(output_img_dir, f"{name}_cleaned_missing_matrix.png")
    plt.savefig(matrix_path, dpi=200)
    plt.close(fig)

    # Generate Chart 2: Completeness Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    msno.bar(df, fontsize=10, color='royalblue', ax=ax)
    ax.set_title(f"{name} Cleaned Dataset - Completeness Counts", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    bar_path = os.path.join(output_img_dir, f"{name}_cleaned_missing_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close(fig)

    print(f"Plots generated: {matrix_path}, {bar_path}")

print("\n==================================================================")
print(f"Missingness audit summary successfully exported to: {output_txt}")
print(f"Visualizations saved to folder: ./{output_img_dir}/")
print("==================================================================")