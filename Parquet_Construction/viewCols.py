import pandas as pd
import os

base = os.path.join("..", "eswData", "Sorted_Parquet")
parquet_fnames = ["EM_trimmed.parquet", "SL_trimmed.parquet", "SR-EM_trimmed.parquet"]

ls = []

for fname in parquet_fnames:
    ls.append(list(pd.read_parquet(os.path.join(base,fname))))

for fname, col_list in zip(parquet_fnames, ls):
    print("-"*30, fname, "-"*30)
    print()
    print(col_list)
    print()
    print("-"*30)