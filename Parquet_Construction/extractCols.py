import pandas as pd
import os

base = os.path.join("..", "eswData", "Sorted_Parquet")

required_columns = {
    "EM": [
        'id', 'Node_id', 'Timestamp', 'Version', 'Sensor_Timestamp', 
        'GDVLL', 'GDVLN', 'GDCurrent', 'GDFrequency', 'GDPower', 
        'GDPower Factor', 'GDTotal Energy', 'T1VLL', 'T1VLN', 'T1Current'
    ],
    "SL": [
        'id', 'Node_id', 'Timestamp', 'Version', 'Sensor_Timestamp', 
        'VoltageR_N', 'VoltageY_N', 'VoltageB_N', 'Voltage_avg', 
        'VoltageR_Y', 'VoltageY_B', 'VoltageB_R', 'RPhaseLC', 'YPhaseLC', 
        'BPhaseLC', 'NPhaseLC', 'RPhaseactC', 'YPhaseactC', 'BPhaseactC', 
        'RPhaseRactC', 'YPhaseRactC', 'BPhaseRactC', 'Q1PF', 'Q2PF', 'Q3PF', 
        'PF_Avg', 'R_Active_power', 'Y_Active_power', 'B_Active_power', 
        '3_Active_power', 'R_Reactive_power', 'Y_Reactive_power', 
        'B_Reactive_power', '3_Reactive_power', 'R_Apparent_power', 
        'Y_Apparent_power', 'B_Apparent_power', '3_Apparent_power', 
        'RY_ph_ang', 'YB_ph_ang', 'BR_ph_ang', 'Frequency', 'TEI', 'TEE', 
        'Eac_today', 'Eac_total', 'ActivePower', 'Voltage_RS', 'Voltage_ST', 
        'Voltage_TR', 'Power Factor', 'Voltage1', 'Current1', 'Power1', 
        'Voltage2', 'Current2', 'Power2', 'Voltage3', 'Current3', 'Power3', 
        'PV1Voltage', 'PV1Current', 'PV2Voltage', 'PV2Current', 'PV3Voltage', 
        'PV3Current', 'PV4Voltage', 'PV4Current', 'PV5Voltage', 'PV5Current', 
        'PV6Voltage', 'PV6Current', 'PV1Power', 'PV2Power', 'PV3Power', 
        'PV4Power', 'PV5Power', 'PV6Power'
    ],
    "SR-EM": [
        'id', 'Node_id', 'Timestamp', 'Version', 'Sensor_Timestamp', 
        'Energy', 'Power', 'Current', 'Frequency', 'Power Factor'
    ]
}

parquet_fnames = ["EM.parquet", "SL.parquet", "SR-EM.parquet"]

for fname in parquet_fnames:
    heading = fname.replace(".parquet", "")
    input_path = os.path.join(base, fname)
    
    df = pd.read_parquet(input_path)
    
    # Calculate initial attributes
    initial_cols = list(df.columns)
    target_cols = required_columns[heading]
    
    # Determine columns kept, dropped, and missing
    cols_to_keep = [col for col in target_cols if col in initial_cols]
    dropped_cols = [col for col in initial_cols if col not in set(cols_to_keep)]
    missing_requested = [col for col in target_cols if col not in set(initial_cols)]
    
    # Slice dataframe and export
    df_trimmed = df[cols_to_keep]
    output_fname = f"{heading}_trimmed.parquet"
    output_path = os.path.join(base, output_fname)
    df_trimmed.to_parquet(output_path)
    
    # Display Diff Summary
    print(f"\n=================== DIFF SUMMARY: {heading} ===================")
    print(f"Initial Shape : {df.shape}")
    print(f"Final Shape   : {df_trimmed.shape}")
    print(f"Columns Kept  : {len(cols_to_keep)} / {len(initial_cols)}")
    print(f"Columns Dropped: {len(dropped_cols)}")
    
    if dropped_cols:
        print(f" -> Dropped Column List: {dropped_cols}")
        
    if missing_requested:
        print(f" -> WARNING (Requested columns not found in source file): {missing_requested}")
        
    print(f"Saved to: {output_fname}")
    print("=" * 55)