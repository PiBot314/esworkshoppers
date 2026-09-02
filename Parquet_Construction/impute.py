import pandas as pd
import os

# def prep_time_index(df): 
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     df.set_index('Timestamp', inplace=True)
#     df.sort_index(inplace=True)
    
#     df = df[~df.index.duplicated(keep='first')]

#     # DUE TO ERRORS by timeline misalignment, I had to recreate the masterparquet.
#     # Resample numeric columns to 1-hour averages to align timestamps perfectly
#     # (numeric_only prevents agg errors on text columns like 'Version')
#     df = df.resample('1h').mean()
    
#     return df

def prep_time_index(df): 
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # 1. Remove exact duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    # 2. Isolate only the numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    # 3. Resample to 1-hour averages (no unsupported arguments needed)
    df = numeric_df.resample('1h').mean()
    
    return df

def clean_and_interpolate_pipeline(
    input_dir: str, 
    output_path: str
):
    # 1. Load the trimmed datasets
    em_df = pd.read_parquet(os.path.join(input_dir, "EM_trimmed.parquet"))
    sl_df = pd.read_parquet(os.path.join(input_dir, "SL_trimmed.parquet"))
    srem_df = pd.read_parquet(os.path.join(input_dir, "SR-EM_trimmed.parquet"))

    em_df = prep_time_index(em_df)
    sl_df = prep_time_index(sl_df)
    srem_df = prep_time_index(srem_df)
        
    # 2. Extract and rename the key columns into a single consolidated DataFrame
    # Using SL's index as the baseline time index since it is the most complete
    merged_df = pd.DataFrame(index=sl_df.index) 
    
    # Map the target features to their source variables
    merged_df['Grid_Draw_kW'] = em_df['GDPower']
    merged_df['Solar_Yield_kW'] = sl_df['3_Active_power']
    merged_df['Room_Demand_kW'] = srem_df['Power']
    
    # 3. Apply Imputation Strategies
    
    # Solar_Yield_kW (1 missing value / 0.01%): 
    # Standard time interpolation perfectly preserves the daylight generation curve
    merged_df['Solar_Yield_kW'] = merged_df['Solar_Yield_kW'].interpolate(method='time')
    
    # Grid_Draw_kW (81 missing values / 0.92%): 
    # Short-gap linear interpolation bridged using limit=3
    merged_df['Grid_Draw_kW'] = merged_df['Grid_Draw_kW'].interpolate(method='time', limit=3)
    
    # Room_Demand_kW (1,504 missing values / 17.17%): 
    # Dual-Tier Hybrid. limit=3 fills short gaps (<= 3 hours). 
    # The remaining multi-week dropouts are kept as NaN to prevent corrupting ML model training.
    merged_df['Room_Demand_kW'] = merged_df['Room_Demand_kW'].interpolate(method='time', limit=3)
    
    # 4. Export the clean dataset
    merged_df.to_parquet(output_path)
    
    # Validation summary
    print(f"Pipeline complete. Saved to: {output_path}")
    print("Remaining NaN values (expected in Room_Demand_kW for ML handling):")
    print(merged_df.isna().sum())
    
    return merged_df

base = os.path.join("..", "eswData", "Sorted_Parquet")
clean_df = clean_and_interpolate_pipeline(base, "masterPower.parquet")