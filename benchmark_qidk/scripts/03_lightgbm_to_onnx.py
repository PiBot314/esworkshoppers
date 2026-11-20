#!/usr/bin/env python3
"""Convert the teammate's LightGBM (MultiOutputRegressor of 288 LGBMRegressors, 24 h horizon) to ONNX so it
can run on the QIDK CPU with ONNX Runtime, and check it matches LightGBM's own predictions.
Needs lightgbm, skl2onnx and onnxmltools (run with the bench venv)."""
import os
import time

import joblib
import numpy as np
import onnxruntime as ort
from lightgbm import LGBMRegressor
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
FINAL = os.path.dirname(B)

update_registered_converter(LGBMRegressor, "LightGbmLGBMRegressor",
                            calculate_linear_regressor_output_shapes, convert_lightgbm)

model = joblib.load(os.path.join(FINAL, "common", "lightgbm_all_sensors_24h.pkl"))
onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([1, 58]))],
                      target_opset={"": 17, "ai.onnx.ml": 3})
out = os.path.join(B, "models", "lightgbm_24h.onnx")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "wb").write(onx.SerializeToString())
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

X = np.load(os.path.join(B, "data", "X_test_last.npy"))[:200]
ref = model.predict(X).astype(np.float32)
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
t = time.perf_counter()
got = np.concatenate([s.run(None, {"input": X[i:i + 1]})[0] for i in range(len(X))])
print(f"PC onnxruntime: {(time.perf_counter() - t) / len(X) * 1000:.2f} ms/inference;"
      f" max |onnx - lightgbm| = {np.abs(got - ref).max():.2e}")
