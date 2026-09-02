import os
import sys
import pandas as pd

class DualLogger:
    """Redirects console print output to both the terminal and a log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def analyze_master_power(filepath="masterPower.parquet", log_filename="master_summary.txt"):
    # Initialize dual logging
    sys.stdout = DualLogger(log_filename)
    
    print(f"Loading dataset: {filepath}...")
    df = pd.read_parquet(filepath)
    
    # Enforce Datetime Index for temporal distribution tracking
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)
        
    print("\n==================================================================")
    print("       MASTER POWER PARQUET: COMPREHENSIVE EDA & NaN AUDIT        ")
    print("==================================================================")

    # --- 1. SHAPE & INDEX INFO ---
    print("\n--- 1. SHAPE & INDEX INFO ---")
    print(f"Index Range: {df.index.min()} to {df.index.max()}")
    print(f"Total Rows: {len(df):,} | Total Columns: {len(df.columns)}")

    # --- 2. GLOBAL MISSINGNESS AUDIT ---
    print("\n--- 2. GLOBAL MISSINGNESS AUDIT ---")
    null_counts = df.isna().sum()
    valid_counts = df.notna().sum()
    null_percents = df.isna().mean() * 100

    missing_df = pd.DataFrame({
        'Valid_Count': valid_counts,
        'Missing_Count': null_counts,
        'Missing_Percent (%)': null_percents.round(2)
    })
    print(missing_df.to_string())

    # --- 3. TEMPORAL NaN DISTRIBUTION (MONTHLY) ---
    print("\n--- 3. TEMPORAL NaN DISTRIBUTION (WHERE ARE THE GAPS?) ---")
    try:
        # Group missing values by month to identify systemic dropouts
        monthly_nans = df.isna().resample('M').sum()
        monthly_nans.index = monthly_nans.index.strftime('%Y-%m')
        
        # Filter out months that have zero missing data to keep the report clean
        monthly_nans = monthly_nans[(monthly_nans > 0).any(axis=1)]
        
        if monthly_nans.empty:
            print("No missing data detected across any month.")
        else:
            print("Monthly count of missing records per column:")
            print(monthly_nans.to_string())
    except Exception as e:
        print(f"Could not calculate temporal distribution: {e}")

    # --- 4. DESCRIPTIVE STATISTICS ---
    print("\n--- 4. DESCRIPTIVE STATISTICS ---")
    desc = df.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    desc['skewness'] = df.skew()
    print(desc[['mean', 'std', 'min', '1%', '50%', '99%', 'max', 'skewness']].round(3).to_string())

    # --- 5. ZERO-VALUE & INACTIVITY AUDIT ---
    print("\n--- 5. ZERO-VALUE AUDIT ---")
    zero_counts = (df == 0).sum()
    zero_percents = (df == 0).mean() * 100
    zero_df = pd.DataFrame({
        'Zero_Value_Count': zero_counts,
        'Zero_Percent (%)': zero_percents.round(2)
    })
    print(zero_df.to_string())

    print("\n==================================================================")
    print(f"Pipeline Finished Successfully! Full report saved to: {log_filename}")
    print("==================================================================")

fpath = os.path.join("..", "Sorted_Parquet", "masterPower.parquet")

if __name__ == "__main__":
    analyze_master_power(fpath)