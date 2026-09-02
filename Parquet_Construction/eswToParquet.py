import os
import pandas as pd
# import dotenv

# Define relative paths based on your setup
data_folder = os.path.join('..', 'eswData', 'esw')
output_folder = os.path.join('..', 'eswData', 'Sorted_Parquet')

os.makedirs(output_folder, exist_ok=True)

# 1. Process EM (Single-column sort on Column C -> save as Parquet)
print("Processing EM...")
em_path = os.path.join(data_folder, 'EM.csv')
df_em = pd.read_csv(em_path)

col_c = df_em.columns[2]  # Column C (index 2)
df_em.sort_values(by=col_c, ascending=False, inplace=True)

# Save as .parquet instead of .csv
df_em.to_parquet(os.path.join(output_folder, 'EM.parquet'), index=False)


# 2. Process SL and SR-EM (Two-column sort on Column C & D -> save as Parquet)
two_col_files = ['SL.csv', 'SR-EM.csv']

for file in two_col_files:
    print(f"Processing {file}...")
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)
    
    c_name = df.columns[2]  # Column C
    d_name = df.columns[3]  # Column D
    
    df.sort_values(by=[c_name, d_name], ascending=[False, False], inplace=True)
    
    # Replace .csv extension with .parquet for output
    parquet_filename = file.replace('.csv', '.parquet')
    df.to_parquet(os.path.join(output_folder, parquet_filename), index=False)

print("All CSVs converted, sorted, and saved as Parquet!")