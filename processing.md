# Includes Basic Steps Followed for Data Cleaning and Analysis

1. Download Data as CSV

2. Convert to Parquet for efficiency of operations, and partial extractions via dataframe.py

3. Get summary via dataframe.py
    - Note: Getting different summary on running same operation twice. Analyze for future

4. Create Visualisations (Graphs) for ease of understanding via data_audit.py, comparision.py, individual.py
    - 4.5. Identify unnecessary columns and get brief understanding of data

5. Trim out unnecessary columns via extractCols.py. (For viewing columns: viewCols.py)

6. Impute data via impute.py
    - Drop columns that have insufficient data (by NaN, Null or Zero Values) to interpolate or extract meaning out of
    - Simple Impute Data for columns without large gaps 
    - Time Series Interpolate (Impute) Data for Columns with time series data with multiple missing values
    - As part of this align timelines to NON-EXACT so that data points are not lost because index not matching
    - [TODO] Use regression for imputation with large gaps

Current Status:
                Zero_Value_Count  Zero_Percent (%)
Grid_Draw_kW                 233              2.66
Solar_Yield_kW                 0              0.00
Room_Demand_kW                 0              0.00

Anomolously high levels of skew in Solar (negatives) and Grid consumption.

[IMP] These were key values:
```
    'Grid_Draw_kW': em_h['GDPower'],
    'Solar_Yield_kW': sl_h['3_Active_power'],
    'Room_Demand_kW': srem_h['Power']
```