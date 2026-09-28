"""Causal test: is it the token pattern or the token average of a steering edit that moves the forecast?

Context-only predictor protocol (src/behavior.py). At layer L, the per-token spline edit
    D_p = s_p(target) - s_p(v_hat)         (one curve per token position, v_hat from a val-fit pooled probe)
is split into
    full      D
    mean      mean_p D_p, added to every token (a pooled edit)
    pattern   D - mean_p D_p (changes how tokens differ; token average unchanged)
    mean_nm   the mean part rescaled to the Frobenius norm of the pattern part (same edit size, uniform)
Each is multiplied by a strength in {1, 2, 4, 8} chosen per (part, layer) on VAL clips by % on target;
reported on TEST clips: % on target, and for speed / acceleration % with forecast direction kept (±15°).
Also reports the norm of each part relative to the full edit.
Writes outputs/results/behavior_pattern_<dataset>.json.
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
from src.manifold import Curve, coord
from src.model import load_encoder
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--layers", default="4,8,12,16,20,24")
ap.add_argument("--n-val", type=int, default=32)
ap.add_argument("--n-test", type=int, default=64)
ap.add_argument("--tag", default="")
args = ap.parse_args()
name = args.dataset
LAYERS = [int(v) for v in args.layers.split(",")]
SCALES = [1.0, 2.0, 4.0, 8.0]
PARTS = ["full", "mean", "pattern", "mean_nm"]
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
idx = split_idx(name)
tr, va, te = idx["train"], idx["val"], idx["test"][: args.n_test]
vsel = va[: args.n_val]
readout = fit_readout(model, name, df, y, tr, va)
th_readout = None if name == "direction" else fit_readout(model, name, df, targets("direction", df), tr, va)
th_true = df.theta_degrees.values
Fp = load_features(name, "vjepa2_ctx").astype(np.float64)
out_path = OUT / "results" / f"behavior_pattern_{name}{args.tag}.json"
res = {"dataset": name, "tolerance": TOL, "targets": TARGETS, "scales": SCALES, "layers": {}}

for L in LAYERS:
    X = Fp[:, L]
    pool_probe = fit_ridge(X[va], y[va], alpha=1.0)
    u_tr = coord(name, lab[tr])
    lo, hi = (0.0, 2 * np.pi) if name == "direction" else (u_tr.min(), u_tr.max())
    curve = Curve(name, "lsq", K, lo, hi)
    Apinv = torch.from_numpy(np.linalg.pinv(curve.design(u_tr))).float().cuda()
    Bc = None
    for s in range(0, len(tr), BS):
        h = encode_to(model, df, tr[s:s + BS], L).float()
        c = torch.einsum("kb,bnd->knd", Apinv[:, s:s + len(h)], h)
        Bc = c if Bc is None else Bc + c
    S = lambda u: torch.einsum("mk,knd->mnd", torch.from_numpy(curve.design(np.atleast_1d(u))).float().cuda(), Bc)
    norms = {p: [] for p in PARTS}

    def evaluate(rows, record_norms=False):
        on = {p: {sc: [] for sc in SCALES} for p in PARTS}
        kept = {p: {sc: [] for sc in SCALES} for p in PARTS}
        for s in range(0, len(rows), BS):
            r = rows[s:s + BS]
            h = encode_to(model, df, r, L).float()
            v_hat = dec(pool_probe, h.mean(1).double().cpu().numpy())
            if name != "direction":
                v_hat = np.clip(v_hat, lab[tr].min(), lab[tr].max())
            S_hat = S(coord(name, v_hat))
            for t in TARGETS:
                D = S(coord(name, float(t))) - S_hat                               # [b, 1024, d]
                Dm = D.mean(1, keepdim=True)
                Dp = D - Dm
                nm = Dp.flatten(1).norm(dim=1) / (Dm.expand_as(D).flatten(1).norm(dim=1) + 1e-8)
                parts = {"full": D, "mean": Dm, "pattern": Dp, "mean_nm": Dm * nm[:, None, None]}
                if record_norms:
                    fn = D.flatten(1).norm(dim=1)
                    for p, d in parts.items():
                        norms[p].append((d.expand_as(D).flatten(1).norm(dim=1) / fn).cpu().numpy())
                for p, d in parts.items():
                    for sc in SCALES:
                        f = forecast_from(model, h + sc * d, L)
                        on[p][sc].append(err(dec(readout, f), float(t)) <= TOL)
                        if th_readout is not None:
                            kept[p][sc].append(circ_err(angle_deg(th_readout.predict(f)), th_true[r]) <= 15)
        return on, kept

    on_v, _ = evaluate(vsel)
    best = {p: max(SCALES, key=lambda sc: np.concatenate(on_v[p][sc]).mean()) for p in PARTS}
    on_t, kept_t = evaluate(te, record_norms=True)
    rL = res["layers"][str(L)] = {}
    for p in PARTS:
        rL[p] = {"best_scale_on_val": best[p],
                 "on_target_pct": float(np.concatenate(on_t[p][best[p]]).mean() * 100),
                 "test_on_target_by_scale": {str(sc): float(np.concatenate(on_t[p][sc]).mean() * 100) for sc in SCALES},
                 "norm_rel_full": float(np.concatenate(norms[p]).mean())}
        if th_readout is not None:
            rL[p]["direction_kept_pct"] = float(np.concatenate(kept_t[p][best[p]]).mean() * 100)
    print(f"L{L}", {p: (rL[p]["best_scale_on_val"], round(rL[p]["on_target_pct"], 1), round(rL[p]["norm_rel_full"], 2)) for p in PARTS}, flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    Bc = None
    torch.cuda.empty_cache()
