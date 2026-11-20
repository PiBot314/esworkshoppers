# QIDK benchmark: baseline vs guided LSTM (and CNN-LSTM, LightGBM)

## Accuracy on the test split (R², per domain)

Same inputs for every row. `SQNR` = agreement with the FP32 ONNX model on the PC (higher is closer; >40 dB is effectively identical).

| Model | Runs on | SQNR dB | Solar R² | Grid R² | Demand R² | Solar nMAE % | Grid nMAE % |
|---|---|---|---|---|---|---|---|
| baseline_h4 | PC ONNX FP32 (reference) | - | 0.909 | 0.607 | -0.004 | 4.92 | 5.64 |
| baseline_h4 | NPU W8A16 (QNN HTP) | 46.5 | 0.909 | 0.615 | 0.025 | 4.91 | 5.56 |
| baseline_h4 | CPU FP32 (QNN CPU) | 136.0 | 0.909 | 0.607 | -0.004 | 4.92 | 5.64 |
| baseline_h4 | CPU FP32 (ONNX Runtime) | 141.6 | 0.909 | 0.607 | -0.004 | 4.92 | 5.64 |
| guided_h4 | PC ONNX FP32 (reference) | - | 0.954 | 0.721 | -0.268 | 3.49 | 4.51 |
| guided_h4 | NPU W8A16 (QNN HTP) | 44.9 | 0.954 | 0.718 | -0.286 | 3.47 | 4.55 |
| guided_h4 | CPU FP32 (QNN CPU) | 136.4 | 0.954 | 0.721 | -0.268 | 3.49 | 4.51 |
| guided_h4 | CPU FP32 (ONNX Runtime) | 142.0 | 0.954 | 0.721 | -0.268 | 3.49 | 4.51 |
| baseline_24h | PC ONNX FP32 (reference) | - | 0.778 | -0.060 | -2.918 | 7.81 | 7.88 |
| baseline_24h | NPU W8A16 (QNN HTP) | 44.4 | 0.777 | -0.067 | -2.905 | 7.81 | 7.90 |
| baseline_24h | CPU FP32 (QNN CPU) | 134.9 | 0.778 | -0.060 | -2.918 | 7.81 | 7.88 |
| baseline_24h | CPU FP32 (ONNX Runtime) | 139.6 | 0.778 | -0.060 | -2.918 | 7.81 | 7.88 |
| guided_24h | PC ONNX FP32 (reference) | - | 0.800 | -0.067 | -3.995 | 7.27 | 7.47 |
| guided_24h | NPU W8A16 (QNN HTP) | 44.3 | 0.801 | -0.064 | -4.002 | 7.26 | 7.45 |
| guided_24h | CPU FP32 (QNN CPU) | 134.8 | 0.800 | -0.067 | -3.995 | 7.27 | 7.47 |
| guided_24h | CPU FP32 (ONNX Runtime) | 139.7 | 0.800 | -0.067 | -3.995 | 7.27 | 7.47 |
| cnn_lstm_24h | PC ONNX FP32 (reference) | - | 0.739 | -1.776 | -6.933 | 9.21 | 15.77 |
| cnn_lstm_24h | NPU W8A16 (QNN HTP) | 44.2 | 0.738 | -1.781 | -7.077 | 9.24 | 15.79 |
| cnn_lstm_24h | CPU FP32 (QNN CPU) | 134.7 | 0.739 | -1.776 | -6.933 | 9.21 | 15.77 |
| cnn_lstm_24h | CPU FP32 (ONNX Runtime) | 137.3 | 0.739 | -1.776 | -6.933 | 9.21 | 15.77 |
| lightgbm_24h | PC ONNX FP32 (reference) | - | 0.752 | 0.198 | -1.726 | 7.59 | 6.83 |

## Latency per inference (one window, batch 1)

| Model | Config | Time per inference ms | NPU compute ms | p95 ms |
|---|---|---|---|---|
| baseline_24h | CPU FP32 (QNN CPU) | 30.193 | - | - |
| baseline_24h | NPU W8A16 (QNN HTP) | 5.020 | 3.470 | - |
| baseline_24h | CPU FP32 (ONNX Runtime, 1 thread) | 1.046 | - | 1.079 |
| baseline_24h | CPU FP32 (ONNX Runtime, 4 threads) | 0.863 | - | 0.902 |
| baseline_h4 | CPU FP32 (QNN CPU) | 16.129 | - | - |
| baseline_h4 | NPU W8A16 (QNN HTP) | 4.855 | 3.297 | - |
| baseline_h4 | CPU FP32 (ONNX Runtime, 1 thread) | 0.588 | - | 0.607 |
| baseline_h4 | CPU FP32 (ONNX Runtime, 4 threads) | 0.532 | - | 0.551 |
| cnn_lstm_24h | CPU FP32 (QNN CPU) | 3.544 | - | - |
| cnn_lstm_24h | NPU W8A16 (QNN HTP) | 1.868 | 0.973 | - |
| cnn_lstm_24h | CPU FP32 (ONNX Runtime, 1 thread) | 0.157 | - | 0.238 |
| cnn_lstm_24h | CPU FP32 (ONNX Runtime, 4 threads) | 0.175 | - | 0.266 |
| guided_24h | CPU FP32 (QNN CPU) | 30.911 | - | - |
| guided_24h | NPU W8A16 (QNN HTP) | 5.050 | 3.489 | - |
| guided_24h | NPU W8A16, power_saver profile | 5.052 | 3.494 | - |
| guided_24h | CPU FP32 (ONNX Runtime, 1 thread) | 1.056 | - | 1.089 |
| guided_24h | CPU FP32 (ONNX Runtime, 4 threads) | 0.860 | - | 0.897 |
| guided_h4 | CPU FP32 (QNN CPU) | 16.262 | - | - |
| guided_h4 | NPU W8A16 (QNN HTP) | 4.807 | 3.283 | - |
| guided_h4 | CPU FP32 (ONNX Runtime, 1 thread) | 0.611 | - | 0.634 |
| guided_h4 | CPU FP32 (ONNX Runtime, 4 threads) | 0.548 | - | 0.573 |
| lightgbm_24h | CPU FP32 (ONNX Runtime, 1 thread) | 2.070 | - | 2.118 |
| lightgbm_24h | CPU FP32 (ONNX Runtime, 4 threads) | 1.936 | - | 2.145 |
