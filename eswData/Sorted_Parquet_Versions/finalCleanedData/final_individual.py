import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

# 1. Look right here first, then in subdirectories / parent directories
if os.path.exists("EM_final.parquet"):
    input_dir = "."
    output_dir = "final_individual"
elif os.path.exists(os.path.join("finalCleanedData", "EM_final.parquet")):
    input_dir = "finalCleanedData"
    output_dir = "final_individual"
elif os.path.exists(os.path.join("..", "finalCleanedData", "EM_final.parquet")):
    input_dir = os.path.join("..", "finalCleanedData")
    output_dir = os.path.join("..", "final_individual")
else:
    input_dir = "."
    output_dir = "final_individual"

os.makedirs(output_dir, exist_ok=True)

file_list = [
    (os.path.join(input_dir, 'EM_final.parquet'), 'EM_Final'),
    (os.path.join(input_dir, 'SL_final.parquet'), 'SL_Final'),
    (os.path.join(input_dir, 'SR-EM_final.parquet'), 'SREM_Final')
]

for filepath, dataset_label in file_list:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}: File not found.")
        continue

    print(f"Plotting individual columns for {dataset_label} from {filepath}...")
    df = pd.read_parquet(filepath)
    if not isinstance(df.index, pd.DatetimeIndex) and 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(series.index, series.values, linewidth=1, color='tab:blue')

        y_min, y_max = series.min(), series.max()
        padding = 1.0 if y_min == y_max else (y_max - y_min) * 0.05
        ax.set_ylim(bottom=y_min - padding, top=y_max + padding)

        ax.set_title(f'{dataset_label} - Parameter: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(col, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        out_name = f"{dataset_label}_{col.replace(' ', '_')}.png"
        save_path = os.path.join(output_dir, out_name)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

print(f"All final individual parameter plots written to: ./{output_dir}/")