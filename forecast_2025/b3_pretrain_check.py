"""Follow-up to B3: does simulated-plant pre-training help with a lighter recipe? (solar, 24 h)

B3's recipe (6 x 400k simulated windows, then fine-tune at lr 1e-3) lost ~0.02 R2 vs training on real 2025 only.
Here, per seed: pre-train 1 or 2 epochs, then fine-tune at the from-scratch lr (2e-3). The pre-training length is
chosen on the VALIDATION weeks; the test numbers are reported for every setting alongside 'forecast' (no pre-training).
Output: results/b3_pretrain_check.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b3_guided_lstm as B  # noqa: E402

H, hname = 96, "24h"
df = pd.read_parquet(B.DATA / "data_15min.parquet")
ser, names_p, names_c, wp, wc = B.series_solar(df, H)
W = B.Windows(ser, H, len(B.PLANTS))
sim = pd.read_parquet(B.DATA / "sim_2015_2024.parquet")
W_sim = B.Windows(B.series_solar(sim, H, sim=True)[0], H, len(B.PLANTS))
gp, gc = B.gate_prior("solar", hname, names_p, names_c)
res = {}
for pre_ep in (0, 1, 2):
    runs = []
    for seed in range(2):
        torch.manual_seed(seed); np.random.seed(seed)
        net = B.Net(len(names_p), len(names_c), H, len(B.PLANTS), gp, gc).to(B.DEV)
        if pre_ep:
            B.train(net, W_sim, pre_ep, 2e-3, per_epoch=400_000, patience=9, tag=f"pre{pre_ep} s{seed} pre")
        val = B.train(net, W, 30, 2e-3, tag=f"pre{pre_ep} s{seed}")
        o, y, d, _ = B.predict(net, W, "test")
        runs.append({"val_mse": val, **B.scores(o, y, d)})
        print(f"pre_epochs {pre_ep} seed {seed}: val {val:.5f} test last R2 {runs[-1]['last_step']['R2']:.4f} "
              f"all {runs[-1]['all_steps']['R2']:.4f}", flush=True)
    res[f"pre_epochs_{pre_ep}"] = {"val_mse_mean": float(np.mean([r["val_mse"] for r in runs])),
                                   "test_last_R2_mean": float(np.mean([r["last_step"]["R2"] for r in runs])),
                                   "test_all_R2_mean": float(np.mean([r["all_steps"]["R2"] for r in runs])), "runs": runs}
best = min(res, key=lambda k: res[k]["val_mse_mean"])
res["chosen_on_val"] = best
json.dump(res, open(HERE / "results" / "b3_pretrain_check.json", "w"), indent=1)
print("chosen on validation:", best, json.dumps({k: {m: round(v[m], 4) for m in ("val_mse_mean", "test_last_R2_mean",
      "test_all_R2_mean")} for k, v in res.items() if k != "chosen_on_val"}, indent=1))
