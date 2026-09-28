"""Fixes for the two spline failures, same-layer read-out at layer 12 (as f5).
  spline          baseline: replace PCA-64 with s(target) (B-spline clamps u to the fitted range)
  spline_extrap   same, but the curve is extended linearly beyond its ends (end slope)
  spline_disp     move along the curve: x + s(target) - s(u_hat), u_hat decoded by a val-fit pooled probe
  spline_2d_shift shift along the clip's own curve on the joint manifold: x + s(target, θ_hat) - s(u_hat, θ_hat)
  spline_2d       joint manifold s(u, θ) (B-spline in u x Fourier in θ, fit on all train clips);
                  replace PCA-64 with s(target, θ_hat), θ_hat decoded by a val-fit pooled direction probe
Metrics as f5: % on target (seen / unseen / outside range) and % with direction intact (seen).
Speed and acceleration only (direction has no range ends and no second variable here).
Writes outputs/results/spline_fixes.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import BSpline

from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import coord, fit_manifold
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL, load_splits

L = 12
TOL = {"speed": 0.375, "acceleration": 0.975}
KB = {n: json.loads((OUT / "results" / "manifolds.json").read_text())[n]["K"] for n in TOL}
ALPHA = {n: json.loads((OUT / "results" / "probe_layers.json").read_text())[n]["raw"]["vjepa2"][L]["alpha"] for n in TOL}


def bspline_design(u, lo, hi, K, extrap):
    t = np.r_[[lo] * 4, np.linspace(lo, hi, K + 2)[1:-1], [hi] * 4]
    u = np.atleast_1d(np.asarray(u, float))
    D = BSpline.design_matrix(np.clip(u, lo, hi), t, 3).toarray()
    if extrap:
        eps = 1e-4 * (hi - lo)
        for edge, sgn in ((lo, 1), (hi, -1)):
            m = (u < lo) if edge == lo else (u > hi)
            if m.any():
                d0 = BSpline.design_matrix(np.array([edge]), t, 3).toarray()
                d1 = BSpline.design_matrix(np.array([edge + sgn * eps]), t, 3).toarray()
                slope = (d1 - d0) / (sgn * eps)
                D[m] = d0 + (u[m] - edge)[:, None] * slope
    return D


res = {"layer": L}
for name in TOL:
    df = load_manifest(name)
    lab = df[LABEL[name]].values
    y = targets(name, df)
    yth = targets("direction", df)
    X = load_features(name)[:, L].astype(np.float64)
    sp = load_splits()[name]
    res[name] = {}
    for cond, split, key, tvals in (("seen", "primary", "test", np.linspace(0.25, 4.0, 8) if name == "speed" else np.linspace(0.25, 10.0, 8)),
                                    ("unseen", "value_heldout", "heldout_values", np.array(sp["heldout_values"])),
                                    ("outside", "value_heldout_ends", "heldout_values", np.array(sp["heldout_values_ends"]))):
        idx = split_idx(name, split)
        tr, va, st = idx["train"], idx["val"], idx[key]
        man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", KB[name])
        probe = fit_ridge(X[va], y[va], alpha=ALPHA[name])
        thp = fit_ridge(X[va], yth[va], alpha=ALPHA[name])
        Z = man.project(X[tr])
        u_tr = coord(name, lab[tr]); lo, hi = u_tr.min(), u_tr.max()
        # extrapolating 1-D curve
        A1 = bspline_design(u_tr, lo, hi, KB[name], True)
        B1 = np.linalg.lstsq(A1, Z, rcond=None)[0]
        s1 = lambda u: bspline_design(u, lo, hi, KB[name], True) @ B1
        # 2-D curve: B-spline in u  x  [1, sin θ, cos θ, sin 2θ, cos 2θ]
        th_tr = np.radians(df.theta_degrees.values[tr])
        four = lambda th: np.column_stack([np.ones_like(th), np.sin(th), np.cos(th), np.sin(2 * th), np.cos(2 * th)])
        A2 = lambda u, th: (bspline_design(u, lo, hi, KB[name], True)[:, :, None] * four(np.atleast_1d(th))[:, None, :]).reshape(len(np.atleast_1d(u)), -1)
        B2 = np.linalg.lstsq(A2(u_tr, th_tr), Z, rcond=None)[0]
        Xs = X[st]
        Zs = man.project(Xs)
        th_true = df.theta_degrees.values[st]
        u_hat = coord(name, np.clip(probe.predict(Xs)[:, 0], lab.min(), lab.max()))
        th_hat = np.radians(angle_deg(thp.predict(Xs)))
        out = {m: {"on": [], "kept": []} for m in ("spline", "spline_extrap", "spline_disp", "spline_2d_shift", "spline_2d")}
        for t in tvals:
            ut = coord(name, float(t))
            edits = {"spline": man.steer(Xs, float(t)),
                     "spline_extrap": Xs - man.lift(Zs) + man.lift(np.broadcast_to(s1(ut), Zs.shape)),
                     "spline_disp": Xs + man.lift(s1(np.full(len(Xs), ut)) - s1(u_hat)) - man.mu,
                     "spline_2d_shift": Xs + man.lift((A2(np.full(len(Xs), ut), th_hat) - A2(u_hat, th_hat)) @ B2) - man.mu,
                     "spline_2d": Xs - man.lift(Zs) + man.lift(A2(np.full(len(Xs), ut), th_hat) @ B2)}
            for m, Xe in edits.items():
                out[m]["on"].append(np.abs(probe.predict(Xe)[:, 0] - t) <= TOL[name])
                out[m]["kept"].append(circ_err(angle_deg(thp.predict(Xe)), th_true) <= 15)
        res[name][cond] = {m: {"on_target_pct": float(np.concatenate(v["on"]).mean() * 100),
                               "direction_kept_pct": float(np.concatenate(v["kept"]).mean() * 100)} for m, v in out.items()}
        print(name, cond, {m: (round(v["on_target_pct"], 1), round(v["direction_kept_pct"], 1)) for m, v in res[name][cond].items()}, flush=True)
(OUT / "results" / "spline_fixes.json").write_text(json.dumps(res, indent=1))
