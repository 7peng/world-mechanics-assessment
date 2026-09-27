"""Checks quoted in the write-up that are not produced by the main experiments. Saves
outputs/results/sanity_checks.json:
  - label vs trajectory: px per metre, direction error of the fitted trajectory
  - frame exits in the direction set; distance travelled per label value (speed, acceleration)
  - displacement confound: layer-12 acceleration probe applied to constant-velocity (speed-set) clips
  - probe sanity: 5-fold stratified CV and shuffled-label control at layers 1, 9, 12
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.probes import fit_ridge, metrics, r2, targets

LABEL = {"direction": "theta_degrees", "speed": "speed_mps", "acceleration": "acceleration_mps2"}
T = 15 / 24  # time between first and last frame (s)
res = {"time_window_s": T}

# ---- labels vs measured trajectories
for name in DATASETS:
    df = load_manifest(name)
    z = np.load(OUT / "baselines" / f"centroids_{name}.npz")
    c, a = z["centroid"], z["area"]
    full = np.median(a)
    ok = (a >= 0.9 * full).all(1)
    t = np.arange(16) / 24
    X = np.c_[np.ones(16), t, 0.5 * t ** 2]
    coef = np.linalg.lstsq(X, np.c_[c[ok, :, 0].T, -c[ok, :, 1].T], rcond=None)[0]
    n = ok.sum()
    vx, vy, ax_, ay = coef[1, :n], coef[1, n:], coef[2, :n], coef[2, n:]
    th = np.degrees(np.arctan2(ay + vy, ax_ + vx)) % 360
    dth = np.abs((th - df.theta_degrees.values[ok] + 180) % 360 - 180)
    r = {"clips_fully_visible": int(n), "clips_with_empty_frame": int((a == 0).any(1).sum()),
         "clips_with_partial_frame": int((a < 0.9 * full).any(1).sum()), "n_clips": len(df),
         "median_direction_error_deg": float(np.median(dth))}
    if name != "direction":
        lab = df[LABEL[name]].values[ok]
        fit = np.hypot(vx, vy) if name == "speed" else np.hypot(ax_, ay)
        r["px_per_m_median"] = float(np.median(fit / lab))
        disp = np.linalg.norm(c[:, -1] - c[:, 0], axis=1)
        v = np.round(df[LABEL[name]].values, 4)
        u = np.unique(v)
        r["label_values"] = u.tolist()
        r["mean_displacement_px"] = [float(disp[v == x].mean()) for x in u]
        r["label_range"] = [float(u.min()), float(u.max())]
    res[name] = r
    print(name, {k: v for k, v in r.items() if k not in ("label_values", "mean_displacement_px")})

# ---- displacement confound
L = 12
da, ds = load_manifest("acceleration"), load_manifest("speed")
Fa, Fs = load_features("acceleration")[:, L].astype(float), load_features("speed")[:, L].astype(float)
ia = split_idx("acceleration")
ya = targets("acceleration", da)
p = fit_ridge(Fa[ia["train"]], ya[ia["train"]], Fa[ia["val"]], ya[ia["val"]])
pred = p.predict(Fs)[:, 0]
cs = np.load(OUT / "baselines" / "centroids_speed.npz")["centroid"]
dsp = np.linalg.norm(cs[:, -1] - cs[:, 0], axis=1)
px_m = res["speed"]["px_per_m_median"]
a_equiv = 2 * (dsp / px_m) / T ** 2          # acceleration from rest covering the same distance
res["confound"] = {"layer": L, "mean_pred_accel_on_velocity_clips": float(pred.mean()),
                   "corr_with_displacement": float(np.corrcoef(pred, dsp)[0, 1]),
                   "slope_vs_equivalent_accel": float(np.polyfit(a_equiv, pred, 1)[0])}
print("confound", res["confound"])

# ---- probe sanity: 5-fold CV and shuffled labels
rng = np.random.default_rng(0)
res["cv"] = {}
for name in DATASETS:
    df = load_manifest(name)
    y, idx = targets(name, df), split_idx(name)
    F = load_features(name)
    lab = np.unique(np.round(df[LABEL[name]].values, 4), return_inverse=True)[1]
    key = "circ_mae_deg" if name == "direction" else "mae"
    res["cv"][name] = {}
    for layer in (1, 9, 12):
        X = F[:, layer].astype(np.float64)
        ys = y[rng.permutation(len(y))]
        q = fit_ridge(X[idx["train"]], ys[idx["train"]], X[idx["val"]], ys[idx["val"]])
        sh = r2(ys[idx["test"]], q.predict(X[idx["test"]]))
        cv = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
            tr = rng.permutation(tr)
            v, t_ = tr[: len(tr) // 5], tr[len(tr) // 5:]
            q = fit_ridge(X[t_], y[t_], X[v], y[v])
            cv.append(metrics(name, y[te], q.predict(X[te])))
        res["cv"][name][layer] = {"shuffled_label_r2": float(sh),
                                  "cv_r2_mean": float(np.mean([c_["r2"] for c_ in cv])),
                                  "cv_r2_std": float(np.std([c_["r2"] for c_ in cv])),
                                  f"cv_{key}": float(np.mean([c_[key] for c_ in cv]))}
    print(name, res["cv"][name])

(OUT / "results" / "sanity_checks.json").write_text(json.dumps(res, indent=1))
