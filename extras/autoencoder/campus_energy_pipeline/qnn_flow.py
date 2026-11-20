from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
MODEL_PATH = MODEL_DIR / "energy_autoencoder.onnx"
CALIBRATION_PATH = ARTIFACTS_DIR / "energy_calibration.npy"
DLC_PATH = MODEL_DIR / "energy_autoencoder.dlc"
THRESHOLD_PATH = MODEL_DIR / "energy_autoencoder_threshold.json"


def build_qnn_commands() -> dict[str, str]:
    dlc_cmd = (
        f"snpe-onnx-to-dlc --input_network {MODEL_PATH} "
        f"--output_path {DLC_PATH} --input_shape "
        '"input[1,20]"'
    )
    net_run_cmd = (
        f"snpe-net-run --container {DLC_PATH} "
        f"--input_list {CALIBRATION_PATH} --output_dir {ARTIFACTS_DIR}"
    )
    return {"dlc_compile": dlc_cmd, "benchmark": net_run_cmd}


def render_qnn_flow() -> None:
    commands = build_qnn_commands()
    print("QNN / SNPE flow for the campus energy autoencoder")
    print("-" * 60)
    print("1. Compile ONNX model to DLC:")
    print(commands["dlc_compile"])
    print("2. Run the quantized model on-device using ADB / snpe-net-run:")
    print(commands["benchmark"])
    print("3. Benchmark metrics to record:")
    print("   - end-to-end latency (ms)")
    print("   - memory footprint (MB)")
    print("   - FP32 vs INT8 reconstruction error")

    if shutil.which("snpe-net-run") is None or shutil.which("snpe-onnx-to-dlc") is None:
        print("\nSNPE tools were not detected in this environment. Install QAIRT/SNPE to run the actual .dlc conversion on QIDK.")

    if THRESHOLD_PATH.exists():
        with THRESHOLD_PATH.open("r", encoding="utf-8") as handle:
            threshold = json.load(handle)
        print(f"\nAutoencoder threshold: {threshold['threshold']:.6f}")


if __name__ == "__main__":
    render_qnn_flow()
