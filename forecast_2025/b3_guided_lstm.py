"""B3/B4: guided LSTM with future covariates, shared over nodes, on REAL 2025 (+ simulated 2015-2024 pre-training).
plan_19112026.md §3 B3-B4, plan_20112026.md.

Template (same shape for every variant, so models can be fused/batched on the NPU later):
  past window   96 x F per node (value, missing flag, clear-sky, observed-irradiance proxy, time [, calendar])
  LSTM          2 layers, hidden 64, over the past window (the NPU body)
  head          [last hidden | future covariates for the H target steps | node one-hot] -> 128 -> H  (dense)
  guided gate   one learnable multiplier per past input and per future covariate, initialised from B2's LightGBM
                gain shares (damped as in common/train_fair_lstm.py), L2-anchored to that prior; folded into
                lstm.weight_ih_l0 / the head's first layer at export (zero extra ops)
  solar output  x (clear-sky > 0): physically 0 at night; clipped at >= 0 in use (linear while training)
Families: solar (6 plants, per kWp, ok days only), grid (SB00, MW).
Variants per family x horizon (1 h = 4 steps, 24 h = 96 steps):
  noweather     weather inputs zeroed (clear-sky + time [+ calendar] only)
  forecast      forecast inputs as the board would have them (1 h: 15-min historical forecast; 24 h: issued a day before)
  forecast+pre  (solar) pre-trained on the simulated 2015-2024 plants (ERA5 standing in for forecasts), then
                fine-tuned on real 2025
  unguided      (forecast[+pre], gate fixed at 1) -- the guided gate's effect
Split: split_blocked (window kept when origin and last target are in the same split; early stopping on val; report
test). Scores: R2 / nMAE at the last step (h) and pooled over all steps, solar on daytime targets; baselines
(persistence, same time yesterday, clear-sky-index persistence) on identical rows.
Outputs: forecast_2025/results/b3_<family>_<h>.json, b3_results.md, models/<family>_<h>_<variant>.onnx (+ .json meta)
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
DATA = FINAL / "datasets" / "training_2025"
OUT = HERE / "results"
MOD = HERE / "models"
OUT.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLANTS = ["PL00_00", "VN95_00", "NI01_00", "NI03_00", "NI03_01", "VN02_01"]
L = 96
log = lambda *a: print(*a, flush=True)
SKIP = False   # set by --skip (grid: baseline skip connection)
TAG = ""       # set by --tag: appended to variant names and the results file


# ----------------------------------------------------------------------------------------------- data
def cyc(idx):
    ist = idx - pd.Timedelta(minutes=7, seconds=30)
    hod = ist.hour + ist.minute / 60
    return {"hod_sin": np.sin(2 * np.pi * hod / 24), "hod_cos": np.cos(2 * np.pi * hod / 24),
            "dow_sin": np.sin(2 * np.pi * ist.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * ist.dayofweek / 7),
            "doy_sin": np.sin(2 * np.pi * ist.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * ist.dayofyear / 365.25)}


def series_solar(src, H, sim=False):
    """Per plant: past features P [N,F], future covariates C [N,Cf], target y [N] (NaN = unusable), split [N]."""
    meta = json.load(open(DATA / "meta.json"))
    c = cyc(src.index)
    cs = src["ghi_clearsky"].to_numpy() / 1000
    if sim:
        ghi_obs = src["era5_shortwave_radiation"]; ghi_fc = ghi_obs
        cloud = src["era5_cloud_cover"]; temp = src["era5_temperature_2m"]
    else:
        ghi_obs = src["fc15_shortwave_radiation"]                       # what the board sees as recent irradiance
        if H <= 24:
            ghi_fc, cloud, temp = src["fc15_shortwave_radiation"], src["fc_cloud_cover"], src["fc15_temperature_2m"]
        else:
            ghi_fc, cloud, temp = src["d1_shortwave_radiation"], src["d1_cloud_cover"], src["d1_temperature_2m"]
    out = []
    for i, p in enumerate(PLANTS):
        if sim:
            y = src[f"sim_{p}_W"].to_numpy() / meta["solar"][f"SL-{p.replace('_', '-')}"]["est_capacity_W"]
            split = np.where(src["year"].to_numpy() == 2024, "val", "train")
        else:
            cap = meta["solar"][f"SL-{p.replace('_', '-')}"]["est_capacity_W"]
            ok = (src[f"solar_{p}_status"] == "ok").to_numpy()
            y = np.where(ok, src[f"solar_{p}_W"].to_numpy() / cap, np.nan)
            split = src["split_blocked"].to_numpy()
        miss = np.isnan(y)
        P = np.stack([np.nan_to_num(y), miss.astype(float), cs, ghi_obs.to_numpy() / 1000,
                      c["hod_sin"], c["hod_cos"], c["doy_sin"], c["doy_cos"]], 1)
        C = np.stack([cs, ghi_fc.to_numpy() / 1000, cloud.to_numpy() / 100, temp.to_numpy() / 50], 1)
        out.append(dict(P=P, C=C, y=y, split=split, node=i, daymask=cs > 0))
    names_p = ["y", "missing", "clearsky", "ghi_obs", "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
    names_c = ["clearsky", "ghi_fc", "cloud_fc", "temp_fc"]
    weather_p, weather_c = [3], [1, 2, 3]
    return out, names_p, names_c, weather_p, weather_c


def series_grid(src, H):
    c = cyc(src.index)
    y = src["grid_W"].to_numpy() / 1e6
    miss = np.isnan(y)
    cs = src["ghi_clearsky"].to_numpy() / 1000
    if H <= 24:
        ghi_fc, temp, rh = src["fc15_shortwave_radiation"], src["fc15_temperature_2m"], src["fc_relative_humidity_2m"]
    else:
        ghi_fc, temp, rh = src["d1_shortwave_radiation"], src["d1_temperature_2m"], src["d1_relative_humidity_2m"]
    cal = [src[k].to_numpy().astype(float) for k in ("in_session", "holiday", "exam", "event")]
    # Load one day and one week before each target step (v2, 20 Nov). Both are known at the origin for H <= 96, and
    # same-time-last-week is LightGBM's top grid feature (37 % of gain at 24 h), which a 96-step window can't see.
    # Indexed by absolute time, so C[tau] = y[tau - lag]; gaps: week -> day -> 0.
    lag = lambda k: np.concatenate([np.full(k, np.nan), y[:-k]])
    y_yday = lag(96)
    y_week = np.where(np.isnan(lag(672)), y_yday, lag(672))
    y_yday, y_week = np.nan_to_num(y_yday), np.nan_to_num(y_week)
    P = np.stack([np.nan_to_num(y), miss.astype(float), cs, src["fc15_shortwave_radiation"].to_numpy() / 1000,
                  src["fc15_temperature_2m"].to_numpy() / 50, c["hod_sin"], c["hod_cos"], c["dow_sin"], c["dow_cos"],
                  c["doy_sin"], c["doy_cos"], *cal], 1)
    C = np.stack([cs, ghi_fc.to_numpy() / 1000, temp.to_numpy() / 50, rh.to_numpy() / 100, c["hod_sin"], c["hod_cos"],
                  c["dow_sin"], c["dow_cos"], *cal, y_yday, y_week], 1)
    names_p = ["y", "missing", "clearsky", "ghi_obs", "temp_obs", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
               "doy_sin", "doy_cos", "in_session", "holiday", "exam", "event"]
    names_c = ["clearsky", "ghi_fc", "temp_fc", "rh_fc", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
               "in_session", "holiday", "exam", "event", "y_yday", "y_week"]
    return ([dict(P=P, C=C, y=y, split=src["split_blocked"].to_numpy(), node=0, daymask=np.ones(len(y), bool))],
            names_p, names_c, [3, 4], [1, 2, 3])


class Windows:
    """All series concatenated on the device; windows are gathered by origin index."""

    def __init__(self, series, H, n_nodes):
        self.H = H
        P, C, Y, D, N, S, idx = [], [], [], [], [], [], {}
        off = 0
        for s in series:
            n = len(s["y"])
            P.append(s["P"]); C.append(s["C"]); Y.append(s["y"]); D.append(s["daymask"])
            N.append(np.full(n, s["node"])); S.append(s["split"])
            t = np.arange(L - 1, n - H)
            y_now = s["y"][t]
            sp0, sp1 = s["split"][t], s["split"][t + H]
            anyy = np.zeros(len(t), bool)
            ok_t = np.isfinite(s["y"])
            # at least one usable target in the window's horizon, a known current value
            cum = np.concatenate([[0], np.cumsum(ok_t)])
            anyy = (cum[t + H + 1] - cum[t + 1]) > 0
            good = anyy & np.isfinite(y_now)
            for name in ("train", "val", "test"):
                m = good & (sp0 == name) & (sp1 == name)
                idx.setdefault(name, []).append(t[m] + off)
            off += n
        f = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=DEV)
        self.P, self.C, self.Y = f(P), torch.nan_to_num(f(C)), f(Y)
        self.D = torch.tensor(np.concatenate(D), device=DEV)
        self.node = torch.tensor(np.concatenate(N), device=DEV, dtype=torch.long)
        self.n_nodes = n_nodes
        self.idx = {k: torch.tensor(np.concatenate(v), device=DEV, dtype=torch.long) for k, v in idx.items()}
        self.ar_p = torch.arange(-L + 1, 1, device=DEV)
        self.ar_f = torch.arange(1, H + 1, device=DEV)

    def batch(self, t):
        past = self.P[t[:, None] + self.ar_p]                 # [B, L, F]
        fut = self.C[t[:, None] + self.ar_f]                  # [B, H, Cf]
        y = self.Y[t[:, None] + self.ar_f]                    # [B, H]
        day = self.D[t[:, None] + self.ar_f]
        onehot = nn.functional.one_hot(self.node[t], self.n_nodes).float()
        return past, fut, onehot, y, day


# ----------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, F, Cf, H, n_nodes, gate_p, gate_c, guided=True, solar=True, hidden=64):
        super().__init__()
        self.guided, self.solar = guided, solar
        self.gate_p = nn.Parameter(torch.tensor(gate_p), requires_grad=guided)
        self.gate_c = nn.Parameter(torch.tensor(gate_c), requires_grad=guided)
        self.register_buffer("prior_p", torch.tensor(gate_p)); self.register_buffer("prior_c", torch.tensor(gate_c))
        self.lstm = nn.LSTM(F, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden + H * Cf + n_nodes, 128)
        self.fc2 = nn.Linear(128, H)
        self.H, self.Cf = H, Cf
        # grid v2 option: forecast = a*y_yday + b*y_week (per step, learnable, init 0.5/0.5) + LSTM correction
        self.skip = None
        if not solar and SKIP:
            self.skip = nn.Parameter(torch.full((2, H), 0.5))

    def forward(self, past, fut, onehot):
        out, _ = self.lstm(past * self.gate_p)
        z = torch.cat([out[:, -1], (fut * self.gate_c).flatten(1), onehot], 1)
        o = self.fc2(torch.relu(self.fc1(z)))
        if self.skip is not None:                               # raw (ungated) day/week-ago covariates = last two
            o = o + self.skip[0] * fut[..., -2] + self.skip[1] * fut[..., -1]
        if self.solar:                                          # covariate 0 = clear-sky: 0 at night
            # linear while training (a ReLU here killed whole output units), clipped at >= 0 in use/export
            o = (o if self.training else torch.relu(o)) * (fut[..., 0] > 0)
        return o

    def prior_penalty(self):
        if not self.guided:
            return 0.0
        return ((self.gate_p - self.prior_p) ** 2).mean() + ((self.gate_c - self.prior_c) ** 2).mean()


def damp(raw):
    d = np.power(np.maximum(np.asarray(raw, float), 1e-5), 0.30)
    return np.clip(d / d.mean(), 0.15, 1.60).astype(np.float32)


def gate_prior(family, hname, names_p, names_c):
    """Map B2's LightGBM gain shares (forecast config, no ERA5) onto the LSTM's inputs."""
    imp = json.load(open(OUT / f"b2_importance_{family}_{hname}.json"))
    g = lambda *keys: sum(v for k, v in imp.items() if any(k.startswith(x) for x in keys))
    fcg = ("fc15_shortwave", "fc15_direct", "fc15_diffuse", "d1_shortwave", "anom_ghi")
    MAPP = {"y": ("y_",), "missing": (), "clearsky": ("cs_T", "zen_T", "csi_"), "ghi_obs": fcg,
            "temp_obs": ("fc15_temperature", "d1_temperature", "anom_temp"), "in_session": ("in_session",),
            "holiday": ("holiday",), "exam": ("exam",), "event": ("event",)}
    MAPC = {"clearsky": ("cs_T", "zen_T", "csi_persist"), "ghi_fc": fcg, "cloud_fc": ("fc_cloud", "d1_cloud", "anom_cloud"),
            "temp_fc": ("fc15_temperature", "d1_temperature", "anom_temp"), "rh_fc": ("fc_relative", "d1_relative"),
            "in_session": ("in_session",), "holiday": ("holiday",), "exam": ("exam",), "event": ("event",)}
    rp = [g(*MAPP.get(n, (n,))) for n in names_p]
    rc = [g(*MAPC.get(n, (n,))) for n in names_c]
    return damp(rp), damp(rc)


def masked_mse(o, y, day):
    m = torch.isfinite(y) & day
    return ((o - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)


def train(net, W, epochs, lr, per_epoch=None, patience=4, bs=512, tag=""):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    best, best_state, bad = np.inf, None, 0
    tr = W.idx["train"]
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), device=DEV)]
        if per_epoch:
            perm = perm[:per_epoch]
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(perm), bs):
            past, fut, oh, y, day = W.batch(perm[i:i + bs])
            loss = masked_mse(net(past, fut, oh), y, day) + 1e-3 * net.prior_penalty()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        v = evaluate_loss(net, W, "val")
        sched.step(v)
        log(f"    {tag} ep {ep + 1:2d} train {tot / nb:.5f} val {v:.5f} ({time.time() - t0:.0f}s)")
        if v < best - 1e-6:
            best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict(net, W, split, bs=2048):
    net.eval()
    O, Y, D, T = [], [], [], []
    ix = W.idx[split]
    for i in range(0, len(ix), bs):
        t = ix[i:i + bs]
        past, fut, oh, y, day = W.batch(t)
        O.append(net(past, fut, oh)); Y.append(y); D.append(day); T.append(t)
    return torch.cat(O).cpu().numpy(), torch.cat(Y).cpu().numpy(), torch.cat(D).cpu().numpy(), torch.cat(T)


@torch.no_grad()
def evaluate_loss(net, W, split):
    o, y, d, _ = predict(net, W, split)
    m = np.isfinite(y) & d
    return float(np.mean((o[m] - y[m]) ** 2))


def scores(o, y, d):
    def sc(oo, yy):
        mse = np.mean((oo - yy) ** 2)
        return {"R2": float(1 - mse / np.var(yy)), "nMAE": float(np.mean(np.abs(oo - yy)) / np.mean(np.abs(yy))),
                "MSE": float(mse), "n": int(len(yy))}
    m_last = np.isfinite(y[:, -1]) & d[:, -1]
    m_all = np.isfinite(y) & d
    return {"last_step": sc(o[m_last, -1], y[m_last, -1]), "all_steps": sc(o[m_all], y[m_all])}


def baseline_preds(W, t, H, family):
    """[B,H] predictions of the baselines for origins t (normalised units)."""
    T = t[:, None] + W.ar_f
    pers = W.Y[t][:, None].expand(-1, H).clone()               # origin value is always known (window filter)
    yday = W.Y[T - 96]
    yday = torch.where(torch.isfinite(yday), yday, pers)       # missing day-ago value -> persistence (as B2), not 0
    Y = yday
    out = {"persistence": pers, "same_time_yesterday": yday}
    if family == "grid":                                       # LightGBM's top grid feature, so also a baseline
        Tw = (T - 672).clamp(min=0)
        week = torch.where((T >= 672) & torch.isfinite(W.Y[Tw]), W.Y[Tw], yday)
        out["same_time_last_week"] = week
    if family == "solar":
        cs_t = W.P[t, 2]
        # clear-sky index now (last bin with clear-sky >= 0.05, i.e. 50 W/m2, within 4 h)
        k = torch.zeros(len(t), device=DEV); found = torch.zeros(len(t), dtype=torch.bool, device=DEV)
        for lag in range(0, 16):
            c = W.P[t - lag, 2]; v = W.P[t - lag, 0]; miss = W.P[t - lag, 1] > 0
            use = (~found) & (c >= 0.05) & (~miss)
            k = torch.where(use, v / c.clamp(min=1e-3), k); found |= use
        k = k.clamp(0, 1.5)
        out["csi_persistence"] = torch.where(found[:, None], k[:, None] * W.C[T, 0], Y)
    return {n: v.cpu().numpy() for n, v in out.items()}


# ----------------------------------------------------------------------------------------------- export
def export(net, W, path, names_p, names_c, extra):
    net = net.cpu().eval()
    with torch.no_grad():                                       # fold the gates (zero extra ops on the NPU)
        net.lstm.weight_ih_l0.mul_(net.gate_p[None, :])
        hid = net.lstm.hidden_size
        gc = net.gate_c.repeat(net.H)                             # flatten order: [H, Cf] -> step-major
        net.fc1.weight[:, hid:hid + net.H * net.Cf].mul_(gc[None, :])
        net.gate_p.fill_(1.0); net.gate_c.fill_(1.0)
    F, Cf, n = len(names_p), len(names_c), W.n_nodes
    args = (torch.zeros(1, L, F), torch.zeros(1, net.H, Cf), torch.zeros(1, n))
    torch.onnx.export(net, args, str(path), input_names=["past", "future", "node"], output_names=["forecast"],
                      dynamic_axes={"past": {0: "b"}, "future": {0: "b"}, "node": {0: "b"}, "forecast": {0: "b"}},
                      opset_version=17, dynamo=False)
    json.dump({"inputs": {"past": names_p, "future": names_c, "node": n}, "lookback": L, "horizon": net.H,
               "params": int(sum(p.numel() for n_, p in net.named_parameters() if not n_.startswith("gate"))), **extra},
              open(path.with_suffix(".json"), "w"), indent=1)
    net.to(DEV)


# ----------------------------------------------------------------------------------------------- main
def run(family, hname, H, seeds, quick=False):
    df = pd.read_parquet(DATA / "data_15min.parquet")
    if family == "solar":
        ser, names_p, names_c, wp, wc = series_solar(df, H)
        n_nodes = len(PLANTS)
    else:
        ser, names_p, names_c, wp, wc = series_grid(df, H)
        n_nodes = 1
    gp, gc = gate_prior(family, hname, names_p, names_c)
    log(f"[{family} {hname}] gate prior past {dict(zip(names_p, gp.round(2)))} future {dict(zip(names_c, gc.round(2)))}")
    W = Windows(ser, H, n_nodes)
    log(f"  windows: " + ", ".join(f"{k} {len(v):,}" for k, v in W.idx.items()))
    # no-weather copy: weather inputs zeroed
    ser_nw = [dict(s, P=s["P"].copy(), C=s["C"].copy()) for s in ser]
    for s in ser_nw:
        s["P"][:, wp] = 0; s["C"][:, wc] = 0
    W_nw = Windows(ser_nw, H, n_nodes)
    W_sim = None
    if family == "solar":
        sim = pd.read_parquet(DATA / "sim_2015_2024.parquet")
        sser, *_ = series_solar(sim, H, sim=True)
        W_sim = Windows(sser, H, n_nodes)
        log(f"  simulated windows: train {len(W_sim.idx['train']):,}, val {len(W_sim.idx['val']):,}")

    res = {"family": family, "horizon": hname, "H": H, "windows": {k: int(len(v)) for k, v in W.idx.items()},
           "inputs": {"past": names_p, "future": names_c}, "gate_prior": {"past": gp.tolist(), "future": gc.tolist()},
           "variants": {}}
    o0, y0, d0, t_test = None, None, None, None
    variants = [("noweather", W_nw, False, True), ("forecast", W, False, True)]
    if family == "solar":
        variants += [("forecast+pre", W, True, True), ("forecast+pre_unguided", W, True, False)]
    else:
        variants += [("forecast_unguided", W, False, False)]
    ep_real = 3 if quick else 30
    variants = [(n + TAG, *rest) for n, *rest in variants]
    for name, WW, pre, guided in variants:
        runs = []
        for seed in range(seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            net = Net(len(names_p), len(names_c), H, n_nodes, gp if guided else np.ones_like(gp),
                      gc if guided else np.ones_like(gc), guided=guided, solar=family == "solar").to(DEV)
            t0 = time.time()
            if pre:
                train(net, W_sim, 2 if quick else 6, 2e-3, per_epoch=400_000, patience=2, tag=f"{name} s{seed} pre")
            train(net, WW, ep_real, 1e-3 if pre else 2e-3, tag=f"{name} s{seed}")
            o, y, d, t = predict(net, WW, "test")
            runs.append(scores(o, y, d) | {"train_s": round(time.time() - t0)})
            if seed == 0:
                export(net, WW, MOD / f"{family}_{hname}_{name}.onnx", names_p, names_c,
                       {"family": family, "variant": name, "test": runs[-1],
                        "units": "per kWp (W / est. capacity)" if family == "solar" else "MW"})
                if t_test is None:
                    o0, y0, d0, t_test = o, y, d, t
            log(f"  {name} seed {seed}: last-step R2 {runs[-1]['last_step']['R2']:.4f}, all-steps R2 "
                f"{runs[-1]['all_steps']['R2']:.4f}")
        agg = {k: {m: float(np.mean([r[k][m] for r in runs])) for m in ("R2", "nMAE", "MSE")} for k in ("last_step", "all_steps")}
        agg_sd = {k: float(np.std([r[k]["R2"] for r in runs])) for k in ("last_step", "all_steps")}
        res["variants"][name] = {"mean": agg, "R2_std": agg_sd, "runs": runs}
    # baselines on the identical test windows
    res["baselines"] = {}
    for bname, bp in baseline_preds(W, t_test.to(DEV), H, family).items():
        res["baselines"][bname] = scores(bp, y0, d0)
    add_skill(res)
    json.dump(res, open(OUT / f"b3_{family}_{hname}{TAG}.json", "w"), indent=1)
    return res


def add_skill(res):
    """Skill vs the best baseline, chosen separately for the last step and for all steps."""
    res["best_baseline"] = {}
    for k in ("last_step", "all_steps"):
        b = max(res["baselines"], key=lambda n: res["baselines"][n][k]["R2"])
        res["best_baseline"][k] = b
        for v in res["variants"].values():
            v[f"skill_{k}"] = 1 - v["mean"][k]["MSE"] / res["baselines"][b][k]["MSE"]


def report():
    lines = ["# B3/B4: guided LSTM with weather, real 2025 test weeks", "",
             "Built by [b3_guided_lstm.py](../b3_guided_lstm.py). Blocked split (test weeks). Solar: per kWp, daytime, "
             "pooled over plants on ok days; grid: SB00 MW, all hours. *Last step* = the value at the horizon "
             "(+1 h or +24 h); *all steps* = every step of the forecast pooled. Mean ± SD over seeds. Skill = "
             "1 − MSE / MSE(best baseline) on the identical windows; the best baseline is chosen separately for the "
             "last step and for all steps.", ""]
    for f in sorted(OUT.glob("b3_*_*.json")):
        r = json.load(open(f))
        if "family" not in r:                                  # e.g. b3_pretrain_check.json
            continue
        add_skill(r)
        json.dump(r, open(f, "w"), indent=1)
        lines += [f"## {r['family']}, {r['horizon']} (test windows {r['windows']['test']:,})", "",
                  "| Method | R² last step | R² all steps | nMAE last | Skill last | Skill all |", "|---|---|---|---|---|---|"]
        for b, v in r["baselines"].items():
            lines.append(f"| baseline: {b} | {v['last_step']['R2']:.3f} | {v['all_steps']['R2']:.3f} | "
                         f"{v['last_step']['nMAE']:.3f} | – | – |")
        for n, v in r["variants"].items():
            lines.append(f"| LSTM {n} | {v['mean']['last_step']['R2']:.3f} ± {v['R2_std']['last_step']:.3f} | "
                         f"{v['mean']['all_steps']['R2']:.3f} | {v['mean']['last_step']['nMAE']:.3f} | "
                         f"{v['skill_last_step']:+.3f} | {v['skill_all_steps']:+.3f} |")
        lines.append("")
    (OUT / "b3_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="solar", choices=["solar", "grid"])
    ap.add_argument("--h", default="1h", choices=["1h", "24h"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--skip", action="store_true", help="grid: day/week-ago skip connection (v2)")
    ap.add_argument("--tag", default="", help="suffix for result/model files")
    a = ap.parse_args()
    SKIP, TAG = a.skip, a.tag
    if not a.report:
        run(a.family, a.h, {"1h": 4, "24h": 96}[a.h], a.seeds, a.quick)
    report()
