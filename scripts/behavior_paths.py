"""The paper's central claim under behavior: along a steering path, do the predictor's forecasts pass smoothly
through intermediate values (spline) or jump / become incoherent (linear)?

Context-only predictor protocol. For each test clip, K = 11 waypoints t in [0, 1]:
  direction: from the clip's own θ to θ + D (D = 90 and 180, the spline going the increasing-θ way)
  speed / acceleration: from the lowest to the highest steering target
Paths (edit after block L, multiplied by the strength tuned in behavior_final for that method and layer):
  subspace_linear  multi-probe subspace steering, clamp coordinates interpolated linearly (Part 1 method)
  chord            PCA-64 component replaced by the straight line between the two curve points (paper's linear baseline)
  spline           PCA-64 component replaced by the curve point at the interpolated intrinsic coordinate (paper)
  shift            pooled move along the curve: x + P[s(u(t)) - s(u_hat)]
  token_spline     per-position curves, displacement to the interpolated coordinate
  token_chord      per-position straight line between the two endpoints' curve points (token-level linear baseline)
Metrics: % of interior waypoints whose forecast is within tolerance of the ideal intermediate value
(θ + D t; for speed and acceleration, interpolation in the curve's intrinsic coordinate), and for
direction the mean resultant length of decoded angles (1 = coherent, 0 = no preferred direction).
Writes outputs/results/behavior_paths_<dataset>.json.
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
from src.manifold import Curve, coord, fit_manifold, uncoord
from src.model import load_encoder
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL
from src.steer import clamp_coords, target_vec

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--layers", default="24,12")
ap.add_argument("--n-test", type=int, default=64)
ap.add_argument("--only", default=None, help="comma list of paths; merge into the existing json")
ap.add_argument("--spans", default=None, help="direction spans, e.g. 180")
ap.add_argument("--tag", default="", help="output file suffix")
args = ap.parse_args()
name = args.dataset
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}[name]
TARGETS = {"direction": None, "speed": [0.5, 3.5], "acceleration": [1.0, 10.0]}[name]
K_BASIS = json.loads((OUT / "results" / "manifolds.json").read_text())[name]["K"]
FINAL = json.loads((OUT / "results" / f"behavior_final_{name}.json").read_text())
BS, NW = 8, 11
TS = np.linspace(0, 1, NW)
err = (lambda a, b: circ_err(a, b)) if name == "direction" else (lambda a, b: np.abs(a - b))
dec = (lambda p, X: angle_deg(p.predict(X))) if name == "direction" else (lambda p, X: p.predict(X)[:, 0])

model = load_encoder()
df = load_manifest(name)
lab = df[LABEL[name]].values
y = targets(name, df)
idx = split_idx(name)
tr, va, te = idx["train"], idx["val"], idx["test"][: args.n_test]
readout = fit_readout(model, name, df, y, tr, va)
Fp = load_features(name, "vjepa2_ctx").astype(np.float64)
res = {"dataset": name, "tolerance": TOL, "waypoints": TS.tolist(), "layers": {}}
out_path = OUT / "results" / f"behavior_paths_{name}{args.tag}.json"
PATHS = args.only.split(",") if args.only else ["subspace_linear", "chord", "spline", "shift", "token_spline", "token_chord"]
NEED_TOKEN = any(m.startswith("token") for m in PATHS)
if args.only and out_path.exists():
    res = json.loads(out_path.read_text())
spans = ([float(v) for v in args.spans.split(",")] if args.spans else [90.0, 180.0]) if name == "direction" else [None]
for L in [int(v) for v in args.layers.split(",")]:
    sc = {m: FINAL["layers"][str(L)][m]["best_scale_on_val"] for m in ("pooled_subspace", "pooled_spline", "pooled_shift", "token_spline")}
    X = Fp[:, L]
    inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, 10, eval_splits=("val",))
    man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", K_BASIS)
    pool_probe = fit_ridge(X[va], y[va], alpha=1.0)
    u_tr = coord(name, lab[tr])
    lo, hi = (0.0, 2 * np.pi) if name == "direction" else (u_tr.min(), u_tr.max())
    curve = Curve(name, "lsq", K_BASIS, lo, hi)
    Apinv = torch.from_numpy(np.linalg.pinv(curve.design(u_tr))).float().cuda()
    Bc = None
    for s in (range(0, len(tr), BS) if NEED_TOKEN else []):
        h = encode_to(model, df, tr[s:s + BS], L).float()
        c = torch.einsum("kb,bnd->knd", Apinv[:, s:s + len(h)], h)
        Bc = c if Bc is None else Bc + c
    S = lambda u: torch.einsum("mk,knd->mnd", torch.from_numpy(curve.design(np.atleast_1d(u))).float().cuda(), Bc)
    rL = res["layers"].setdefault(str(L), {})
    rL.setdefault("scales", {}).update(sc)
    for D in spans:
        key = f"span{int(D)}" if D is not None else "range"
        dec_paths = {m: [] for m in PATHS}
        ideal_all = []
        for s in range(0, len(te), BS):
            rows = te[s:s + BS]
            h = encode_to(model, df, rows, L).float()
            xL = h.mean(1).double().cpu().numpy()
            Z = man.project(xL)
            if name == "direction":
                v0 = lab[rows]; v1 = (lab[rows] + D) % 360
                ideal = (v0[:, None] + D * TS[None]) % 360                       # [b, NW]
                u_path = np.radians(ideal)
            else:
                v0 = np.full(len(rows), TARGETS[0]); v1 = np.full(len(rows), TARGETS[1])
                u0, u1 = coord(name, TARGETS[0]), coord(name, TARGETS[1])
                u_path = np.broadcast_to(u0 + (u1 - u0) * TS[None], (len(rows), NW))
                ideal = uncoord(name, u_path)
            ideal_all.append(ideal)
            v_hat = dec(pool_probe, xL)
            if name != "direction":
                v_hat = np.clip(v_hat, lab.min(), lab.max())
            if NEED_TOKEN:
                S_hat = S(coord(name, v_hat))
                S0 = torch.cat([S(u_path[i, 0]) for i in range(len(rows))])
                S1 = torch.cat([S(u_path[i, -1]) for i in range(len(rows))])
            Zhat = man.curve(coord(name, v_hat))
            # clamp coordinates at each end (per clip for direction)
            V = inlp.Q[:, :10 * (2 if name == "direction" else 1)]
            c0 = np.stack([clamp_coords(inlp, 10, target_vec(name, float(v)))[1] for v in v0])
            c1 = np.stack([clamp_coords(inlp, 10, target_vec(name, float(v)))[1] for v in v1])
            P0 = np.stack([man.point(float(v))[0] for v in v0]); P1 = np.stack([man.point(float(v))[0] for v in v1])
            z = inlp.z(xL)
            per = {m: [] for m in dec_paths}
            for j, t in enumerate(TS):
                zc = z - (z @ V) @ V.T + ((1 - t) * c0 + t * c1) @ V.T
                d_sub = zc * inlp.sd + inlp.mu - xL
                d_chord = man.lift((1 - t) * P0 + t * P1) - man.lift(Z)
                d_spl = man.lift(np.stack([man.curve(u_path[i, j])[0] for i in range(len(rows))])) - man.lift(Z)
                d_shf = (man.curve(u_path[:, j]) - Zhat) @ man.P.T
                if NEED_TOKEN:
                    d_tok = torch.cat([S(u_path[i, j]) for i in range(len(rows))]) - S_hat
                    d_tch = (1 - t) * S0 + t * S1 - S_hat
                for m, d, scl in (("subspace_linear", torch.from_numpy(d_sub).float().cuda()[:, None], sc["pooled_subspace"]),
                                  ("chord", torch.from_numpy(d_chord).float().cuda()[:, None], sc["pooled_spline"]),
                                  ("spline", torch.from_numpy(d_spl).float().cuda()[:, None], sc["pooled_spline"]),
                                  ("shift", torch.from_numpy(d_shf).float().cuda()[:, None], sc["pooled_shift"]),
                                  ("token_spline", d_tok if NEED_TOKEN else None, sc["token_spline"]),
                                  ("token_chord", d_tch if NEED_TOKEN else None, sc["token_spline"])):
                    if m not in per:
                        continue
                    per[m].append(dec(readout, forecast_from(model, h + scl * d, L)))
            for m in dec_paths:
                dec_paths[m].append(np.stack(per[m], 1))                          # [b, NW]
        ideal_all = np.concatenate(ideal_all)
        r = rL.setdefault(key, {})
        r["ideal"] = ideal_all.round(2).tolist()
        for m, v in dec_paths.items():
            v = np.concatenate(v)
            e = err(v, ideal_all)
            r[m] = {"interior_on_path_pct": float((e[:, 1:-1] <= TOL).mean() * 100),
                    "endpoint_on_target_pct": float((e[:, -1] <= TOL).mean() * 100),
                    "on_path_pct_by_waypoint": ((e <= TOL).mean(0) * 100).tolist(),
                    "decoded": v.round(2).tolist()}
            if name == "direction":
                r[m]["coherence_by_waypoint"] = np.abs(np.exp(1j * np.radians(v - ideal_all)).mean(0)).tolist()
            else:
                r[m]["mean_decoded_by_waypoint"] = v.mean(0).tolist()
        print(f"L{L} {key}", {m: (round(v['interior_on_path_pct'], 1), round(v['endpoint_on_target_pct'], 1)) for m, v in r.items() if m != "ideal"}, flush=True)
        out_path.write_text(json.dumps(res, indent=1))
    Bc = None
    torch.cuda.empty_cache()
