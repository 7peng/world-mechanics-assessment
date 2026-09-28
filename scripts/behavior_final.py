"""Final behavioral comparison of steering methods (context-only predictor protocol, src/behavior.py).

Methods (edit to the context-encoder residual stream after block L, multiplied by a strength s):
  pooled_subspace  Part 1 multi-probe subspace steering on the pooled activation (INLP N = 10), delta added to every token
  pooled_spline    Part 2 pooled manifold: replace PCA-64 component with the curve point, delta added to every token
  pooled_shift     Part 2 pooled manifold, move along the curve: x + P[s(target) - s(v_hat)], delta added to every token
  token_subspace   the same clamp per token, with a shared token-level INLP (N = 10) fit on individual tokens
  token_spline     per-position manifold: h_p += s_p(target) - s_p(u_hat), u_hat decoded by a pooled probe
Strength s in {0.5, 1, 2, 4, 8}, chosen per (method, layer) on 64 VAL clips by % on target; reported on
96 TEST clips. Metrics: % of forecasts within tolerance of the target; for speed and acceleration also
% of forecasts whose direction stays within 15 deg of the clip's true direction (clean rate reported).
Writes outputs/results/behavior_final_<dataset>.json.
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
from src.manifold import Curve, coord, fit_manifold
from src.model import load_encoder
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL
from src.steer import clamp_coords, steer, target_vec

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--layers", default="12,24")
ap.add_argument("--n-val", type=int, default=64)
ap.add_argument("--n-test", type=int, default=96)
ap.add_argument("--only", default=None, help="comma list of methods; merge into the existing json")
args = ap.parse_args()
name = args.dataset
LAYERS = [int(v) for v in args.layers.split(",")]
SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
METHODS = args.only.split(",") if args.only else ["pooled_subspace", "pooled_spline", "pooled_shift", "token_subspace", "token_spline"]
NEED_TOKEN = any(m.startswith("token") for m in METHODS)
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}[name]
TARGETS = {"direction": [0.0, 90.0, 180.0, 270.0], "speed": [0.5, 1.5, 2.5, 3.5], "acceleration": [1.0, 4.0, 7.0, 10.0]}[name]
K = json.loads((OUT / "results" / "manifolds.json").read_text())[name]["K"]
BS, N_PROBES, TPC = 8, 10, 192
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
Fp = load_features(name, "vjepa2_ctx").astype(np.float64)
rng = np.random.default_rng(0)
res = {"dataset": name, "tolerance": TOL, "targets": TARGETS, "scales": SCALES, "layers": {}}
out_path = OUT / "results" / f"behavior_final_{name}.json"
if args.only:
    res = json.loads(out_path.read_text())

# clean rates on the test clips
clean_f = np.concatenate([forecast_from(model, encode_to(model, df, te[s:s + BS], 0), 0) for s in range(0, len(te), BS)])
res["clean"] = {"on_target_pct": float(np.mean([(err(dec(readout, clean_f), float(t)) <= TOL).mean() * 100 for t in TARGETS])),
                "on_original_pct": float((err(dec(readout, clean_f), lab[te]) <= TOL).mean() * 100)}
if th_readout is not None:
    res["clean"]["direction_kept_pct"] = float((circ_err(angle_deg(th_readout.predict(clean_f)), df.theta_degrees.values[te]) <= 15).mean() * 100)
print("clean", res["clean"], flush=True)

for L in LAYERS:
    X = Fp[:, L]
    inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, N_PROBES, eval_splits=("val",))
    man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", K)
    pool_probe = fit_ridge(X[va], y[va], alpha=1.0)
    # token-level INLP on sampled tokens of train clips (+ val tokens for alpha)
    toks, tys = [], []
    for s in (range(0, 320, BS) if NEED_TOKEN else []):
        rows = tr[s:s + BS]
        h = encode_to(model, df, rows, L)
        sel = torch.from_numpy(np.stack([rng.choice(1024, TPC, replace=False) for _ in rows])).cuda()
        toks.append(torch.gather(h, 1, sel[..., None].expand(-1, -1, h.shape[-1])).float().cpu().numpy().reshape(-1, h.shape[-1]))
        tys.append(np.repeat(y[rows], TPC, axis=0))
    if NEED_TOKEN:
        Xt, yt = np.concatenate(toks), np.concatenate(tys)
        ntr = int(0.8 * len(Xt))
        tinlp = run_inlp(name, Xt, yt, {"train": np.arange(ntr), "val": np.arange(ntr, len(Xt))}, N_PROBES, eval_splits=("val",))
        del Xt, toks
    # per-position curves
    u_tr = coord(name, lab[tr])
    lo, hi = (0.0, 2 * np.pi) if name == "direction" else (u_tr.min(), u_tr.max())
    curve = Curve(name, "lsq", K, lo, hi)
    Apinv = torch.from_numpy(np.linalg.pinv(curve.design(u_tr))).float().cuda()
    Bc = None
    for s in (range(0, len(tr), BS) if NEED_TOKEN else []):
        h = encode_to(model, df, tr[s:s + BS], L).float()
        c = torch.einsum("kb,bnd->knd", Apinv[:, s:s + len(h)], h)
        Bc = c if Bc is None else Bc + c
    S = lambda u: torch.einsum("mk,knd->mnd", torch.from_numpy(curve.design(np.atleast_1d(u))).float().cuda(), Bc)

    def deltas(h, t):
        """h [b, 1024, D] float cuda -> dict method -> delta [b, 1024, D] (or [b, 1, D])."""
        xL = h.mean(1).double().cpu().numpy()
        V, c = clamp_coords(inlp, N_PROBES, target_vec(name, float(t)))
        d = {"pooled_subspace": torch.from_numpy(steer(inlp.z(xL), V, c) * inlp.sd + inlp.mu - xL).float().cuda()[:, None],
             "pooled_spline": torch.from_numpy(man.steer(xL, float(t)) - xL).float().cuda()[:, None]}
        v_hat = dec(pool_probe, xL)
        if name != "direction":
            v_hat = np.clip(v_hat, lab.min(), lab.max())
        d["pooled_shift"] = torch.from_numpy(man.shift(xL, float(t), v_hat) - xL).float().cuda()[:, None]
        if not NEED_TOKEN:
            return d
        H = h.double().cpu().numpy().reshape(-1, h.shape[-1])
        Vt, ct = clamp_coords(tinlp, N_PROBES, target_vec(name, float(t)))
        Z = tinlp.z(H)
        d["token_subspace"] = torch.from_numpy((steer(Z, Vt, ct) * tinlp.sd + tinlp.mu - H).reshape(h.shape)).float().cuda()
        d["token_spline"] = S(coord(name, float(t))) - S(coord(name, v_hat))
        return d

    def evaluate(rows):
        """-> on_target[method][scale] list of bool arrays, kept[method][scale]"""
        on = {m: {sc: [] for sc in SCALES} for m in METHODS}
        kept = {m: {sc: [] for sc in SCALES} for m in METHODS}
        th_true = df.theta_degrees.values
        for s in range(0, len(rows), BS):
            r = rows[s:s + BS]
            h = encode_to(model, df, r, L).float()
            for t in TARGETS:
                D = deltas(h, t)
                for m in METHODS:
                    for sc in SCALES:
                        f = forecast_from(model, h + sc * D[m], L)
                        on[m][sc].append(err(dec(readout, f), float(t)) <= TOL)
                        if th_readout is not None:
                            kept[m][sc].append(circ_err(angle_deg(th_readout.predict(f)), th_true[r]) <= 15)
        return on, kept

    on_v, _ = evaluate(vsel)
    best = {m: max(SCALES, key=lambda sc: np.concatenate(on_v[m][sc]).mean()) for m in METHODS}
    on_t, kept_t = evaluate(te)
    r = res["layers"].setdefault(str(L), {})
    for m in METHODS:
        r[m] = {"best_scale_on_val": best[m],
                "val_on_target_by_scale": {str(sc): float(np.concatenate(on_v[m][sc]).mean() * 100) for sc in SCALES},
                "test_on_target_by_scale": {str(sc): float(np.concatenate(on_t[m][sc]).mean() * 100) for sc in SCALES},
                "on_target_pct": float(np.concatenate(on_t[m][best[m]]).mean() * 100)}
        if th_readout is not None:
            r[m]["direction_kept_pct"] = float(np.concatenate(kept_t[m][best[m]]).mean() * 100)
            r[m]["test_direction_kept_by_scale"] = {str(sc): float(np.concatenate(kept_t[m][sc]).mean() * 100) for sc in SCALES}
    print(f"L{L}", {m: (r[m]["best_scale_on_val"], round(r[m]["on_target_pct"], 1), round(r[m].get("direction_kept_pct", -1), 1)) for m in METHODS}, flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    Bc = None
    torch.cuda.empty_cache()
