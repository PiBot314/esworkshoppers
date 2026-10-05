"""P0 smoke test: can AIMET-ONNX 2.12 load, simulate and quantize the frozen LSTM ONNX models?

Checks per model:
  V1  pass-through sim (float32/float32) == plain ONNX Runtime
  W8A16 sim on the HTP v75 config computes encodings and runs
"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from aimet_onnx.quantsim import QuantizationSimModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "..", "..", "model2", "qidk_deployment")
CONFIG = os.path.join(HERE, "..", "configs", "htp_v75_lstm_per_tensor.json")
MODELS = {
    "all_sensors_1h": ("all_sensors_lstm_qidk.onnx", "sample_input_all_sensors_1x96x58.npy"),
    "all_sensors_24h": ("all_sensors_24h_lstm_qidk.onnx", "sample_input_all_sensors_24h_1x96x58.npy"),
    "single_node": ("SL-PL00-00_lstm_qidk.onnx", "sample_input_1x96.npy"),
}


def main(names):
    for name in names:
        onnx_file, sample_file = MODELS[name]
        model = onnx.load(os.path.join(DEPLOY, onnx_file))
        x = np.load(os.path.join(DEPLOY, sample_file)).astype(np.float32)
        in_name = model.graph.input[0].name

        ref = ort.InferenceSession(model.SerializeToString()).run(None, {in_name: x})[0]

        sim_fp = QuantizationSimModel(onnx.load(os.path.join(DEPLOY, onnx_file)),
                                      param_type="float32", activation_type="float32", config_file=CONFIG,
                                      dummy_input={in_name: x})
        out_fp = sim_fp.session.run(None, {in_name: x})[0]
        print(f"[{name}] V1 pass-through max|d| = {np.abs(out_fp - ref).max():.3e}")

        sim = QuantizationSimModel(onnx.load(os.path.join(DEPLOY, onnx_file)),
                                   param_type="int8", activation_type="int16",
                                   quant_scheme="tf_enhanced", config_file=CONFIG,
                                   dummy_input={in_name: x})
        sim.compute_encodings([{in_name: x}])
        out_q = sim.session.run(None, {in_name: x})[0]
        n_q = len(sim.qc_quantize_op_dict)
        lstm_q = [k for k in sim.qc_quantize_op_dict if "lstm" in k.lower()]
        print(f"[{name}] W8A16 sim max|d| vs FP32 = {np.abs(out_q - ref).max():.3e}; "
              f"{n_q} quantizers, LSTM-related: {lstm_q}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
