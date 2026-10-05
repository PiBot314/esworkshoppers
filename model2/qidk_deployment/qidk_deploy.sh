#!/usr/bin/env bash
# ==============================================================================
# qidk_deploy.sh
# Deployment script to push model artifacts and run inference on Qualcomm QIDK
# Supports Single-Node, Campus 1-Hour, and Campus 24-Hour Day-Ahead Models
# ==============================================================================

set -e

DEVICE_DIR="/data/local/tmp/nz_edge_ai"

echo "=== 1. Checking ADB Device Connection ==="
adb devices
echo ""

echo "=== 2. Creating Destination Directory on QIDK ==="
adb shell "mkdir -p ${DEVICE_DIR}"

echo "=== 3. Pushing Model & Runtime Files to QIDK ==="
# Single-node
adb push SL-PL00-00_lstm_qidk.onnx ${DEVICE_DIR}/
adb push sample_input_1x96.npy ${DEVICE_DIR}/
adb push qidk_model_meta.json ${DEVICE_DIR}/
adb push qidk_edge_inference.py ${DEVICE_DIR}/

# 1-Hour Campus-wide Multi-sensor
adb push all_sensors_lstm_qidk.onnx ${DEVICE_DIR}/
adb push sample_input_all_sensors_1x96x58.npy ${DEVICE_DIR}/
adb push all_sensors_qidk_meta.json ${DEVICE_DIR}/
adb push qidk_all_sensors_inference.py ${DEVICE_DIR}/

# 24-Hour Day-Ahead Campus-wide Multi-sensor
adb push all_sensors_24h_lstm_qidk.onnx ${DEVICE_DIR}/
adb push all_sensors_24h_qidk_meta.json ${DEVICE_DIR}/
adb push qidk_24h_netzero_inference.py ${DEVICE_DIR}/

echo ""
echo "=== 4. Executing On-Device Inference on QIDK Board ==="
echo "--- Running 24-Hour Day-Ahead Campus Net-Zero Inference ---"
adb shell "cd ${DEVICE_DIR} && python3 qidk_24h_netzero_inference.py"

echo ""
echo "--- Running 1-Hour Real-Time Dispatch Campus Inference ---"
adb shell "cd ${DEVICE_DIR} && python3 qidk_all_sensors_inference.py"

echo ""
echo "=== 5. (Optional) Qualcomm QNN Native Converter Step ==="
echo "If compiling for Qualcomm Hexagon NPU using Qualcomm QNN SDK:"
echo "  qnn-onnx-converter --input_network all_sensors_24h_lstm_qidk.onnx --output_path all_sensors_24h.cpp"
echo "  qnn-model-lib-generator -c all_sensors_24h.cpp -b all_sensors_24h.bin -o liball_sensors_24h.so"
echo "=============================================================================="
