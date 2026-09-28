"""Token-level (per-position) manifold steering, judged by the predictor's forecast. Context-only protocol.

At layer L, for every token position p (1,024 = 4 time x 16 x 16), fit a curve s_p(u) through the train
clips' tokens at p as a function of the label's intrinsic coordinate u (same least-squares basis as the
pooled manifold: Fourier harmonics for direction, cubic B-spline otherwise). One shared design matrix, so
the fit is B = pinv(A) @ H, accumulated over clips.
Steering (displacement along each position's curve, keeps clip-specific content such as disk position):
    h_p <- h_p + s_p(u_target) - s_p(u_hat)
where u_hat is the clip's own value decoded by a pooled ridge probe at layer L fit on val (no label use).
Also: 'replace' h_p <- s_p(u_target) (the typical clip at the target value; clip content discarded).
Metrics: % of forecasts within tolerance of the target; for speed / acceleration also % whose forecast
direction stays within 15 deg of the true direction (off-target), using a direction readout fit on
clean forecasts of the same dataset's train clips.
Writes outputs/results/behavior_token_spline_<dataset>.json incrementally.
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
ap.add_argument("--n-test", type=int, default=96)
args = ap.parse_args()
name = args.dataset
LAYERS = [int(v) for v in args.layers.split(",")]
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
readout = fit_readout(model, name, df, y, tr, va)
th_readout = None if name == "direction" else fit_readout(model, name, df, targets("direction", df), tr, va)
Fpool = load_features(name, "vjepa2_ctx").astype(np.float64)
u_tr = coord(name, lab[tr])
lo, hi = (0.0, 2 * np.pi) if name == "direction" else (u_tr.min(), u_tr.max())
curve = Curve(name, "lsq", K, lo, hi)
A = curve.design(u_tr)                                              # [n_tr, n_basis]
Apinv = torch.from_numpy(np.linalg.pinv(A)).float().cuda()          # [n_basis, n_tr]
res = {"dataset": name, "tolerance": TOL, "targets": TARGETS, "layers": {}}
out_path = OUT / "results" / f"behavior_token_spline_{name}.json"
th_true = df.theta_degrees.values[te]
for L in LAYERS:
    # fit per-position curves: B = pinv(A) @ H, H = train tokens [n_tr, 1024*D], streamed over clips
    Bc = None
    for s in range(0, len(tr), BS):
        h = encode_to(model, df, tr[s:s + BS], L).float()           # [b, 1024, D]
        contrib = torch.einsum("kb,bnd->knd", Apinv[:, s:s + len(h)], h)
        Bc = contrib if Bc is None else Bc + contrib
    design = lambda u: torch.from_numpy(curve.design(np.atleast_1d(u))).float().cuda()     # [m, n_basis]
    S = lambda u: torch.einsum("mk,knd->mnd", design(u), Bc)                               # [m, 1024, D]
    pool_probe = fit_ridge(Fpool[va, L], y[va], alpha=1.0)
    got = {k: {t: [] for t in TARGETS} for k in ("displace", "replace")}
    th_keep = {k: [] for k in ("displace", "replace")}
    for s in range(0, len(te), BS):
        rows = te[s:s + BS]
        h = encode_to(model, df, rows, L).float()
        v_hat = dec(pool_probe, h.mean(1).double().cpu().numpy())
        if name != "direction":
            v_hat = np.clip(v_hat, lab.min(), lab.max())
        u_hat = coord(name, v_hat)
        S_hat = S(u_hat)                                              # [b, 1024, D]
        for t in TARGETS:
            St = S(coord(name, t))                                    # [1, 1024, D]
            for k, hh in (("displace", h + St - S_hat), ("replace", St.expand_as(h).contiguous())):
                f = forecast_from(model, hh, L)
                got[k][t].append(dec(readout, f))
                if th_readout is not None:
                    th_keep[k].append((circ_err(angle_deg(th_readout.predict(f)), th_true[s:s + BS]) <= 15))
    r = res["layers"][str(L)] = {}
    for k in got:
        per = {str(t): float((err(np.concatenate(v), float(t)) <= TOL).mean() * 100) for t, v in got[k].items()}
        r[k] = {"within_pct": float(np.mean(list(per.values()))), "per_target": per,
                "mean_decoded": {str(t): (float(np.degrees(np.angle(np.exp(1j * np.radians(np.concatenate(v))).mean())) % 360)
                                          if name == "direction" else float(np.concatenate(v).mean())) for t, v in got[k].items()}}
        if th_keep[k]:
            r[k]["direction_kept_pct"] = float(np.concatenate(th_keep[k]).mean() * 100)
    print(f"L{L}", {k: (round(v["within_pct"], 1), round(v.get("direction_kept_pct", -1), 1)) for k, v in r.items()},
          "means displace", {t: round(v, 2) for t, v in r["displace"]["mean_decoded"].items()}, flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    del Bc
    torch.cuda.empty_cache()
