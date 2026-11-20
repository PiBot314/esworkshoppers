──────
   1. Executive Project Objective & Architecture 

  The objective of this project is to build an Edge AI Net-Zero Building Management & Microgrid System deployed directly onto the Qualcomm Innovators Development Kit (QIDK) powered
  by the Qualcomm Snapdragon SoC and Hexagon NPU.

  The system orchestrates and forecasts the campus energy balance across three primary physical pillars:
  1. Solar PV Generation (SL Sensors): Rooftop photovoltaic arrays and solar plants (Palash, Vindhya, Nilgiri).
  2. Main Substation & Grid Incomer (EM Sensors): High-voltage incomers, distribution transformers, and grid import.
  3. High-Density Computing Demand (SR-EM Sensors): Server rooms, computing clusters, and departmental sub-meters.
  ──────
   2. Phase 1: Raw Data Audit & Identified Flaws 

  When inspecting the raw telemetry in  and early parquet files in , five critical data flaws were discovered that would have caused model divergence or invalid physics:

  1. Unit Inconsistencies (kW vs. W):
      • Certain sensors (e.g., inverters and meters like EM-CR-KH95-00 and SR-EM-KH04-00) reported power in Kilowatts (kW), while others reported in Watts (W). Mixing these directly
      in an LSTM would skew weight updates by a factor of 1,000.
  2. Missing Timesteps & NaN Values:
      • Raw Parquet and CSV files contained intermittent null readings caused by edge network disconnects and packet dropouts.
  3. Physical Polarity Errors (Negative Power):
      • Several sensors registered negative power readings due to inverted current transformer (CT) clamp polarities during installation.
  4. Timestamp Misalignment & Irregular Cadences:
      • Telemetry arrived at asynchronous, irregular time intervals (varying from 10 seconds to 5 minutes) across different hardware versions.
  5. Heterogeneous Schemas:
      • Column naming varied across devices (power_total, active_power, gd_power, apparent_power).

  ──────
   3. Phase 2: Systematic Data Pipeline & Normalization 
  To resolve all flaws, the automated data generation script generate_lstm_dataset.py was built:
  • Unit Standardization: All power-related channels were converted strictly to Watts (W):

    P      = P   × 1000
     Watts    kW
  • Physical Imputation:
      • Where power was NaN but voltage (V) and current (I) were recorded, power was computed using Ohm's/Joule's law (P = V × I).
      • For remaining short dropouts, forward-fill and time-weighted linear interpolation were applied.
      • A boolean indicator column (was_imputed) was added for each sensor to prevent the neural network from confusing synthetic fills with true telemetry.
  • Polarity Inversion Fix: Non-physical negative powers were corrected using absolute values (|P|) or non-negative physical clamping.
  • 15-Minute Uniform Resampling: All 11 sensors were resampled and synchronized to a strict 15-minute grid.
  • IST Cyclic Temporal Features: To provide time context without unbounded monotonic counters, 6 cyclic features were engineered using sine/cosine transforms:
      • sin, cos((2π × Hour)/24) (Diurnal cycle)
      • sin, cos((2π × DayOfWeek)/7) (Weekly work/weekend cycle)
      • sin, cos((2π × DayOfYear)/365.25) (Annual seasonal solar trajectory)
  • Min-Max Normalization to [0, 1]:
      • Scaled every feature to [0, 1] to prevent large numbers from destabilizing LSTM gradients.
      • Saved exact parameters to scaler_params.json so the edge runner can restore values to physical Watts.
  • Dual Output Formats:
      • Generated identical, clean datasets in both Parquet (for fast pipeline loading) and CSV (for human inspection) in  (packaged as lstm_normalized.zip).
  ──────
   4. Phase 3: The 11-Sensor Unified Network 
  The 11 campus sensors provide a total of 58 synchronized features at each 15-minute timestep:

   Sensor ID                          | Family                            | Physical Role                     | Key Channels
  ------------------------------------|-----------------------------------|-----------------------------------|----------------------------------------------------------------------
   SL-PL00-00                         | Solar PV                          | Palash Solar Plant                | Active Power (W), Reactive Power (VAr), Avg Voltage, Neutral Current
   SL-VN95-00                         | Solar PV                          | Vindhya Solar Plant               | Active Power (W), Reactive Power (VAr), Avg Voltage, Neutral Current
   SL-NI01-00                         | Solar PV                          | Nilgiri Solar Plant               | Active Power (W), Reactive Power (VAr), Avg Voltage, Neutral Current
   SL-NI03-00                         | Solar Inverter                    | Nilgiri Inverter 00               | Active Power (W), AC Voltage, DC PV Voltage
   SL-NI03-01                         | Solar Inverter                    | Nilgiri Inverter 01               | Active Power (W), AC Voltage, DC PV Voltage
   SL-VN02-01                         | Solar Inverter                    | Vindhya Inverter 01               | Active Power (W), AC Voltage, DC PV Voltage
   EM-CR-KH95-00                      | Energy Meter                      | Control Room Meter                | Total Power (W), Phase R & B Power, Voltage, Power Factor
   EM-CR-SB00-00                      | Substation Incomer                | Main Substation Meter             | Grid Power (W), Transformer T1 & T2 Power, HV & LT Voltages
   SR-EM-KH04-00                      | Server Meter                      | Lab Main Incomer                  | Total Server Load (W)
   SR-EM-KH04-01                      | Sub-Meter                         | Lab Branch 01                     | Power (W), Voltage, Current, Frequency
   SR-EM-KH04-02                      | Sub-Meter                         | Lab Branch 02                     | Power (W), Voltage, Current, Frequency
  ──────
   5. Phase 4: Comprehensive Benchmarking Suite 

  The benchmarking suite in run_comprehensive_benchmark.py produced machine-verified metrics in benchmark_results.json:

  ### A. Data Storage & I/O Benchmark (Parquet vs. CSV)

  • Disk Footprint: Parquet compressed the dataset from 6.13 MB (CSV) down to 1.48 MB (75.8% space savings).
  • Read Throughput: Parquet loaded in 4.41 ms vs. 29.98 ms for CSV (6.8 × faster).
  • Write Throughput: Parquet wrote in 14.15 ms vs. 22.68 ms for CSV (1.6 × faster).
  • Type Safety: Parquet preserves strict float32 and int64 datatypes without string re-parsing.

  ### B. Edge Hardware Complexity (Qualcomm QIDK Target)

  • Single-Node Model: 58, 500 MACs, 249 KB binary footprint, ≈ 0.038 ms estimated latency on Qualcomm Hexagon NPU.
  • Unified All-Sensors Model: 141, 696 MACs, 586 KB binary footprint, ≈ 0.092 ms estimated latency on Qualcomm Hexagon NPU.
  • Verdict: Consumes < 0.01% of the Qualcomm QIDK's 10+ TOPs compute envelope.

  ### C. Predictive Accuracy: LSTM vs. Three Baselines

  Benchmarked across Solar Generation (546 kW peak) and Substation Grid Draw (887 kW peak):

   Domain                      | Model                         | MAE (W)                     | RMSE (W)                    | Normalized MAE             | R² Score
  -----------------------------|-------------------------------|-----------------------------|-----------------------------|----------------------------|----------------------------
   Solar Generation            | Multi-Sensor LSTM             | 19,646.5 W                  | 31,900.2 W                  | 3.60%                      | 0.9554
                               | Baseline 1: Naïve Persistence | 28,683.4 W                  | 51,213.9 W                  | 5.25%                      | 0.9080
                               | Baseline 2: 4-Step Moving Avg | 31,617.4 W                  | 49,603.8 W                  | 5.79%                      | 0.8872
                               | Baseline 3: 24h Diurnal Lag   | 46,243.6 W                  | 76,211.2 W                  | 8.47%                      | 0.7725
   Substation Grid             | Multi-Sensor LSTM             | 31,252.2 W                  | 46,127.0 W                  | 3.52%                      | 0.7940
                               | Baseline 1: Naïve Persistence | 37,215.1 W                  | 57,094.2 W                  | 4.19%                      | 0.7410
                               | Baseline 2: 4-Step Moving Avg | 40,511.9 W                  | 58,112.5 W                  | 4.57%                      | 0.6974
                               | Baseline 3: 24h Diurnal Lag   | 46,812.3 W                  | 68,419.0 W                  | 5.28%                      | 0.6019
  ──────
   6. Phase 5: The Net-Zero Physical Formulation 

  In response to your query on why solar should be subtracted rather than added:

  • Campus Demand (Demand): Power absorbed by buildings.
  • Solar Generation (Solar): Clean power generated on campus.
  • Net Grid Balance:

    Net Deficit (W) = Demand (W) - Solar Generation (W)

                                        ⎛Solar Generation⎞
    Solar Self-Sufficiency Ratio (\%) = ⎜────────────────⎟ × 100%
                                        ⎝ Campus Demand  ⎠

  ### Physical Regimes:

  1. Solar ≥ Demand (Net-Zero Surplus):
      • The campus is 100% self-sufficient.
      • The surplus power (Solar - Demand) can charge batteries or export clean energy back to the grid.
  2. Solar < Demand (Net Deficit):
      • On-site solar cannot meet total consumption.
      • Residual power (Demand - Solar) must be imported from the external power grid.


  │ Edge Clamping: All inference scripts now enforce Solar = max(0, Solar) to prevent neural network night-time leakage.
  ──────
   7. Phase 6: Dual-Horizon Multi-Sensor Models 

  To meet real-world edge operational demands, two complementary models were developed:

  ### Model A: Short-Term Real-Time Dispatch Forecaster (H = 4 steps / +1 Hour)

  • Script: train_all_sensors_lstm.py
  • Input: [1, 96, 58] (past 24h across all 11 sensors).
  • Output: [1, 12] (next 1h: 4 steps of 15 min × 3 domains: Solar, Grid, Demand).
  • Target: Inverter power adjustments, battery ramp-rate smoothing, and micro-grid frequency regulation.
  • Accuracy: Solar R² = 0.9554, Grid R² = 0.7940.
  • ONNX Model: all_sensors_lstm_qidk.onnx (586 KB).

  ### Model B: Day-Ahead Planning Forecaster (H = 96 steps / +24 Hours)

  • Script: train_24h_multisensor_lstm.py
  • Input: [1, 96, 58] (past 24h across all 11 sensors).
  • Output: [1, 288] (next 24h: 96 steps of 15 min × 3 domains = 288 values).
  • Target: Day-ahead grid procurement, battery charge/discharge scheduling (charging during peak solar hours, discharging during evening peak), building thermal pre-cooling, and
  peak tariff shaving.
  • Accuracy (Full 24-Hour Horizon):
      • Solar: R² = 0.8045, MAE = 41.7 kW (7.64% of 546 kW peak).
      • Grid: R² = 0.7410, MAE = 34.9 kW (3.94% of 887 kW peak).
  • ONNX Model: all_sensors_24h_lstm_qidk.onnx (1.14 MB).
  ──────
   8. Phase 7: Qualcomm QIDK Edge Deployment Package 

  All models were formatted for Qualcomm Snapdragon Hexagon NPU execution within :

  1. Static Tensors: Fixed batch dimension (B = 1) with no dynamic shapes, as required by the Qualcomm QNN converter (qnn-onnx-converter).
  2. Bitwise Parity: Validated that PyTorch checkpoints match ONNX Runtime predictions with numerical differences below 2.38 × 10⁻⁷.
  3. Execution Scripts:
      • qidk_24h_netzero_inference.py: Standalone 24-hour day-ahead Net-Zero inference engine with hourly energy breakdown.
      • qidk_all_sensors_inference.py: Standalone 1-hour real-time dispatch inference engine.
      • qidk_edge_inference.py: Single-node inverter inference runner.
      • qidk_deploy.sh: Automated ADB shell script to push models, load metadata, transfer NumPy test tensors, and execute on-device inference on a physical QIDK board.
      • qidk_deployment.zip: Portable deployment archive.

  ──────
   9. Complete Directory & File Catalog 

    /home/claryfray29/esw/model/
    │
    ├── progress.txt                         <- Comprehensive 560+ line engineering report
    ├── benchmark_results.json               <- Verified I/O, Edge Latency, and Accuracy JSON data
    ├── generate_lstm_dataset.py             <- Data cleaning, unit standardization, and normalization pipeline
    ├── train_and_export_qidk.py             <- Single-node PyTorch training & ONNX export script
    ├── train_all_sensors_lstm.py             <- 11-sensor 1-hour LSTM training & ONNX export script
    ├── train_24h_multisensor_lstm.py        <- 11-sensor 24-hour day-ahead LSTM training & ONNX export script
    ├── run_comprehensive_benchmark.py       <- Automated benchmarking suite (Parquet vs CSV, QIDK, 3 Baselines)
    │
    ├── lstm_normalized/                     <- Clean normalized dataset directory
    │   ├── features.json                    <- Schema metadata (58 features, 0 nulls)
    │   ├── scaler_params.json               <- Min/Max/Scale inverse transform parameters
    │   └── *.parquet / *.csv                <- Clean data files for all 11 sensors
    ├── lstm_normalized.zip                  <- Zipped archive of clean normalized datasets
    │
    ├── qidk_deployment/                     <- Qualcomm QIDK edge deployment bundle
    │   ├── all_sensors_24h_lstm_qidk.onnx   <- 24h Day-Ahead static ONNX model [1, 96, 58] -> [1, 288]
    │   ├── best_24h_all_sensors_lstm.pt     <- 24h Day-Ahead PyTorch trained weights checkpoint
    │   ├── all_sensors_24h_qidk_meta.json   <- 24h Metadata, scalers, and domain mappings
    │   ├── sample_input_all_sensors_24h_1x96x58.npy <- 24h Day-Ahead test tensor
    │   ├── qidk_24h_netzero_inference.py    <- 24h Standalone edge runner
    │   │
    │   ├── all_sensors_lstm_qidk.onnx       <- 1h Short-Term static ONNX model [1, 96, 58] -> [1, 12]
    │   ├── best_all_sensors_lstm.pt         <- 1h Short-Term PyTorch trained weights checkpoint
    │   ├── all_sensors_qidk_meta.json       <- 1h Metadata, scalers, and domain mappings
    │   ├── sample_input_all_sensors_1x96x58.npy <- 1h Short-Term test tensor
    │   ├── qidk_all_sensors_inference.py    <- 1h Standalone edge runner
    │   │
    │   ├── SL-PL00-00_lstm_qidk.onnx        <- Single-node static ONNX model [1, 96, 10] -> [1, 4]
    │   ├── best_lstm.pt                     <- Single-node PyTorch weights checkpoint
    │   ├── qidk_model_meta.json             <- Single-node metadata and scalers
    │   ├── sample_input_1x96.npy            <- Single-node test tensor
    │   ├── qidk_edge_inference.py           <- Single-node edge runner
    │   └── qidk_deploy.sh                   <- ADB deployment script to push and run on QIDK board
    │
    └── qidk_deployment.zip                  <- Complete portable QIDK deployment archive
  ──────
   10. Quick Command Reference (For Future Execution) 

  When you are ready to run inference or deploy to hardware in the future:

  1. 24-Hour Day-Ahead Net-Zero Edge Inference:
    /home/claryfray29/esw/esworkshoppers/venv/bin/python3 qidk_deployment/qidk_24h_netzero_inference.py

  2. 1-Hour Real-Time Micro-Dispatch Edge Inference:
    /home/claryfray29/esw/esworkshoppers/venv/bin/python3 qidk_deployment/qidk_all_sensors_inference.py

  3. Full Automated Benchmarking Suite:
    /home/claryfray29/esw/esworkshoppers/venv/bin/python3 run_comprehensive_benchmark.py

  4. Deploy Directly to Physical QIDK Board via ADB:
    cd /home/claryfray29/esw/model/qidk_deployment
    ./qidk_deploy.sh

