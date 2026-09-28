"""Does steering move the predictor's forecast? Sweep over layer, method, and strength. Context-only
protocol (src/behavior.py): the encoder sees frames 1-8; the predictor forecasts frames 9-16.

For one variable (--dataset), 96 test clips, 4 targets. Edits (pooled delta added to every token of the
residual stream after block L, then the encoder finishes and the predictor forecasts):
  subspace   multi-probe subspace steering, INLP (N = 10) fit on train pooled features at layer L
  spline     replace PCA-64 component with the manifold point, manifold fit at layer L
  shift      move along the curve: x + P[s(target) - s(v_hat)], v_hat from a val-fit pooled probe
  centroid   replace the whole pooled activation with the train centroid of the target value
             (strongest possible uniform shift: all 1,024 dims)
  random     random direction with the norm of the subspace edit
Positive control (layer 12 only):
  donor      replace all (context) tokens at layer L with those of a train clip whose label = target (ceiling)
Strength sweep at the layers listed in --scale-layers: subspace and centroid deltas × {2, 4, 8}.
Metric: % of clips whose forecast decodes within tolerance of the target; also mean decoded value.
Writes outputs/results/behavior_sweep_<dataset>.json incrementally.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.behavior import encode_to, fit_readout, forecast_from
from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.inlp import run_inlp
from src.manifold import fit_manifold
from src.model import load_encoder
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL
from src.steer import clamp_coords, steer, target_vec

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--layers", default="4,8,12,16,20,24")
ap.add_argument("--scale-layers", default="12,24")
ap.add_argument("--n-test", type=int, default=96)
ap.add_argument("--only", default=None, help="comma list of methods; merge into the existing json")
args = ap.parse_args()
name = args.dataset
LAYERS = [int(v) for v in args.layers.split(",")]
SCALE_LAYERS = [int(v) for v in args.scale_layers.split(",")]
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}[name]
TARGETS = {"direction": [0.0, 90.0, 180.0, 270.0], "speed": [0.5, 1.5, 2.5, 3.5], "acceleration": [1.0, 4.0, 7.0, 10.0]}[name]
K = json.loads((OUT / "results" / "manifolds.json").read_text())[name]["K"]
BS = 8
err = (lambda a, b: circ_err(a, b)) if name == "direction" else (lambda a, b: np.abs(a - b))
dec = (lambda p, X: angle_deg(p.predict(X))) if name == "direction" else (lambda p, X: p.predict(X)[:, 0])

model = load_encoder()
df = load_manifest(name)
lab = df[LABEL[name]].values
y = targets(name, df)
F = load_features(name, "vjepa2_ctx").astype(np.float64)
idx = split_idx(name)
tr, va, te = idx["train"], idx["val"], idx["test"][: args.n_test]
readout = fit_readout(model, name, df, y, tr, va)
out_path = OUT / "results" / f"behavior_sweep_ctx_{name}.json"
res = {"dataset": name, "tolerance": TOL, "targets": TARGETS, "layers": {}, "scale": {}, "donor": {}}
ONLY = args.only.split(",") if args.only else None
if ONLY:
    res = json.loads(out_path.read_text())
    SCALE_LAYERS = []


def score(decoded_by_target):
    d = {}
    for t, v in decoded_by_target.items():
        e = err(v, float(t))
        d[str(t)] = {"within_pct": float((e <= TOL).mean() * 100),
                     "mean_decoded": float(np.degrees(np.angle(np.exp(1j * np.radians(v)).mean())) % 360) if name == "direction" else float(v.mean())}
    d["within_pct"] = float(np.mean([d[str(t)]["within_pct"] for t in decoded_by_target]))
    return d


def centroid_of(X_pooled_train, lab_train, t):
    vals = np.unique(np.round(lab_train, 6))
    v = vals[np.argmin(np.abs(vals - t) if name != "direction" else circ_err(vals, t))]
    return X_pooled_train[np.round(lab_train, 6) == v].mean(0)


# clean baseline
clean = None if ONLY else np.concatenate([dec(readout, forecast_from(model, encode_to(model, df, te[s:s + BS], 0), 0)) for s in range(0, len(te), BS)])
if not ONLY:
  res["clean"] = {"within_original_pct": float((err(clean, lab[te]) <= TOL).mean() * 100),
                "within_target_pct": float(np.mean([(err(clean, float(t)) <= TOL).mean() * 100 for t in TARGETS]))}
  print("clean", res["clean"], flush=True)

rng = np.random.default_rng(0)
for L in LAYERS:
    X = F[:, L]
    inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, 10, eval_splits=("val",))
    man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", K)
    pool_probe = fit_ridge(X[va], y[va], alpha=1.0)
    cents = {t: centroid_of(X[tr], lab[tr], t) for t in TARGETS}
    dec_by = {m: {t: [] for t in TARGETS} for m in (ONLY or ("subspace", "spline", "shift", "centroid", "random"))}
    scales = [2, 4, 8] if L in SCALE_LAYERS else []
    dec_sc = {f"{m}_x{s}": {t: [] for t in TARGETS} for m in ("subspace", "centroid") for s in scales}
    donor_by = {t: [] for t in TARGETS}
    for s in range(0, len(te), BS):
        rows = te[s:s + BS]
        h = encode_to(model, df, rows, L)
        xL = h.mean(1).double().cpu().numpy()
        v_hat = dec(pool_probe, xL)
        if name != "direction":
            v_hat = np.clip(v_hat, lab[tr].min(), lab[tr].max())
        for t in TARGETS:
            V, c = clamp_coords(inlp, 10, target_vec(name, float(t)))
            D = {"subspace": steer(inlp.z(xL), V, c) * inlp.sd + inlp.mu - xL,
                 "spline": man.steer(xL, float(t)) - xL,
                 "shift": man.shift(xL, float(t), v_hat) - xL,
                 "centroid": cents[t][None] - xL}
            g = rng.standard_normal(xL.shape)
            D["random"] = g * (np.linalg.norm(D["subspace"], axis=1) / np.linalg.norm(g, axis=1))[:, None]
            for m, d in D.items():
                if m not in dec_by:
                    continue
                dec_by[m][t].append(dec(readout, forecast_from(model, h + torch.from_numpy(d).float().cuda()[:, None], L)))
            for sc in scales:
                for m in ("subspace", "centroid"):
                    dec_sc[f"{m}_x{sc}"][t].append(dec(readout, forecast_from(model, h + torch.from_numpy(sc * D[m]).float().cuda()[:, None], L)))
            if L == 12 and not ONLY:
                # donor: all context tokens replaced by those of a train clip with the target value
                vals = np.round(lab[tr], 6)
                cand = tr[np.argsort(np.abs(vals - t) if name != "direction" else circ_err(vals, t))[:len(rows)]]
                hd = encode_to(model, df, cand, L)
                donor_by[t].append(dec(readout, forecast_from(model, hd, L)))
    res["layers"].setdefault(str(L), {}).update({m: score({t: np.concatenate(v) for t, v in dec_by[m].items()}) for m in dec_by})
    if scales:
        res["scale"][str(L)] = {m: score({t: np.concatenate(v) for t, v in dec_sc[m].items()}) for m in dec_sc}
    if L == 12 and not ONLY:
        res["donor"]["12"] = score({t: np.concatenate(v) for t, v in donor_by.items()})
        print("donor L12", res["donor"]["12"]["within_pct"], flush=True)
    print(f"L{L}", {m: round(v["within_pct"], 1) for m, v in res["layers"][str(L)].items()},
          {m: round(v["within_pct"], 1) for m, v in res["scale"].get(str(L), {}).items()}, flush=True)
    out_path.write_text(json.dumps(res, indent=1))
