# 03 — Tooling Audit (what's available, what's missing)

Everything below was checked directly on 2026-10-05 (commands run, not assumed).

## 1. Host machine
| Item | Status |
|---|---|
| OS / CPU / RAM / disk | Fedora (kernel 7.2), 20 threads, **15 GB RAM (~2–3 GB free at check time)**, 234 GB free on /home |
| GPU | RTX 3050 Ti Laptop, 4 GB, driver 615.71. Optional, since these models are tiny and CPU AIMET is fine |
| Python | system 3.14.7, `~/.local/bin/python3.13`. **No Python 3.10 installed** |
| uv | 0.10.0 ✅ (can fetch a managed Python 3.10) |
| `mideval/.venv` | Py 3.14, Sphinx only. **No onnx, onnxruntime, torch, aimet** |
| Docker / Podman | ✅ both present |
| `adb` | ✅ `/usr/bin/adb`, QIDK visible |

## 2. QAIRT SDK (Docker image `qairt:latest`, 18 GB, built ~Aug 2026)
| Item | Status |
|---|---|
| QAIRT SDK | **2.47.0.260601** at `/Tools/qairt_docker/qairt/current` ✅ |
| Android NDK | r26c ✅ |
| Python in image | 3.10.12, venv `/usr/venv`: onnx 1.16.1, onnxruntime 1.16.1, torch 1.13.1+cpu, numpy 2.2.6, **protobuf 3.19.6** |
| Converters / tools | `qairt-converter`, `qairt-quantizer`, `qnn-onnx-converter`, `qnn-context-binary-generator`, `qnn-net-run`, `snpe-*`, `qairt-accuracy-debugger` ✅ |
| HTP v75 libs (SM8650) | `lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so`, `lib/aarch64-android/libQnnHtp*.so` ✅ |
| Container `qairt_container` | exists, **stopped** ("Exited (255) 2 weeks ago"). `run_qairt_docker.sh` already mounts `mideval/ → /local/` |
| **BLOCKER B1** | **`import onnx` fails** (`cannot import name 'builder' from google.protobuf.internal`): onnx 1.16.1 needs protobuf ≥ 3.20.2. Because of this, **`qairt-converter` fails on any ONNX model.** Verified: in a throwaway container, `pip install protobuf==3.20.3` fixes it and the 1 h model converts successfully. (TensorFlow 2.10 in the image then warns about protobuf, which doesn't matter for ONNX.) |
| **Gotcha G1** | `docker run … bash -c` does **not** activate the SDK env (it's in `.bashrc`). Use `source /usr/venv/bin/activate && source $QAIRT_SDK_ROOT/bin/envsetup.sh` explicitly, or `docker exec -it … bash` |
| **Gotcha G2** | Files written to the bind mount by the container are owned by **root**. Run with `--user $(id -u):$(id -g)`, or `chown` afterwards |
| Note | `bin/aimet_env_setup.sh` in the SDK is for the proprietary *aimetpro-torch* tarball, so it's **not** the route for ONNX AutoQuant |

## 3. AIMET
| Item | Status |
|---|---|
| AIMET installed anywhere? | **No** (not on host, not in the image) |
| Latest `aimet-onnx` on PyPI | 2.40.0 (cp310-abi3, Py ≥ 3.10, onnxruntime ≥ 1.19, onnx < 1.23) |
| **AutoQuant for ONNX** | **Removed after 2.12.0.** Checked the GitHub tags: `aimet_onnx/auto_quant_v2.py` exists in 2.0–2.12 and is a 404 from 2.13 on. 2.40 ships only a leftover `common/auto_quant.py` (diagnostics helper) |
| **⇒ Usable AutoQuant release** | **`aimet-onnx==2.12.0`**, a **cp310-only** wheel (manylinux_2_34; Fedora glibc OK). Needs `numpy<2`, `onnxruntime>=1.19`, `torch` |
| Dependency resolution | **Verified with `uv pip compile`** (dry run): aimet-onnx 2.12.0 + onnxruntime 1.19.2 + onnx 1.16.2 + numpy 1.26.4 + torch 2.14.1+cpu resolve cleanly for Py 3.10. uv picked protobuf 7.x, so **pin `protobuf<5`** for onnx 1.16 |
| API (2.12) | `from aimet_onnx.auto_quant_v2 import AutoQuant`. `AutoQuant(model, dummy_input: dict, data_loader, eval_callback(session, num_samples)->float, param_bw=8, output_bw=8, quant_scheme, config_file, results_dir)` → `.run_inference()`, `.optimize(allowed_accuracy_drop)`. Also `AutoQuantWithAutoMixedPrecision` |
| What AutoQuant 2.12 does | quant-scheme selection (tf / tf_enhanced) → W32 early-exit check → BatchNorm fold → CLE → AdaRound. **No SeqMSE** inside AutoQuant |
| HTP configs | `htp_quantsim_config_v75.json` (and `_per_channel_linear`) ship with AIMET, matching SM8650 |
| LSTM handling | AIMET-ONNX QuantSim treats the `LSTM` node as **one opaque op**: it quantizes its inputs, weights and outputs, but **not** the internal gates/cell state. QAIRT/HTP unrolls and quantizes those internals. **There will be a sim-to-target gap**, which is why doc 08 measures on device |

## 4. QIDK device
| Item | Status |
|---|---|
| ADB serial | `d42f06b` (USB) ✅ |
| SoC | **SM8650 (Snapdragon 8 Gen 3) → Hexagon HTP v75**, Android 14, aarch64 |
| Python on device | **None.** `qidk_deploy.sh` as written can't run (`adb shell python3 …`) |
| `/data/local/tmp/` | `campus_energy/` (other project's autoencoder + SNPE runtime chunks), `qairt-2.37.0/`, `qairt-2.37.1/`, `llama`, `qairt-whisper`, … |
| **Version mismatch** | Device runtimes are **2.37.x**, host SDK is **2.47**. Context binaries and DLCs must run with the **same** runtime version they were built with, so push the 2.47 runtime into a **new** directory (doc 07). Don't touch the existing folders |
| Gotcha G3 | Running `adb` on the host started a host adb server on :5037. The container uses `--net host` and ships its own adb. Mixing them kills each other's server, so **use the host `adb` only** |

## 5. Data
| Item | Status |
|---|---|
| `model2/lstm_normalized/` (calib + eval data) | **Missing** |
| `lstm_ready/`, `by_version/` and the script that built `lstm_ready` | **Missing** |
| Raw per-node telemetry `../nodes/parquet_sorted/{EM,SL,SR-EM}/*.parquet` | ✅ all 11 required nodes present (epoch-seconds `Sensor_Timestamp`, 2025-01 → 2025-12) |
| Ground-truth anchors for a rebuild | sample `.npy` windows, `progress.txt` row counts and target ranges, meta capacities ✅ |

## 6. Summary: what blocks AutoQuant today, and the fix
| # | Gap | Remedy (doc 04 / 05) |
|---|---|---|
| B1 | QAIRT image's onnx/protobuf broken | Derive `qairt-fixed` image with `protobuf==3.20.3` |
| B2 | No AIMET, and AutoQuant needs aimet-onnx 2.12 on Py 3.10 | `uv` venv `mideval/.venv-aimet` (Py 3.10) |
| B3 | No calibration or eval data | Rebuild `lstm_normalized/` from `../nodes` and verify against the anchors |
| B4 | No on-device Python / ORT, runtime 2.37 vs 2.47 | Native `qnn-net-run` flow with a pushed 2.47 runtime |
