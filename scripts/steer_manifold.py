"""Part 2: manifold (spline) steering vs the Part 1 subspace clamp, same-layer read-out at layer 12.

Methods (all edit the pooled layer-12 activation of a clip):
  clamp      Part 1: replace coordinates in the span of the first N INLP probes so all probes read the target
  spline     replace the PCA-64 component with the curve point s(target); keep the orthogonal residual (paper Eq. 2)
  spline_span replace only the component in the curve's own span (top singular vectors of the curve samples,
             2K dims for direction, K+3 otherwise); everything else, including the other variables, kept
  spline_d   add the curve displacement s(target) - s(nearest point on curve to the clip) to the clip
  chord      paper's linear baseline: replace the PCA-64 component with the per-value centroid of the target
             (the straight-line path's endpoint; same endpoint as an interpolating spline)
Conditions (which label values the manifold / INLP saw):
  seen       primary split: fit on train, steer test clips, all 64 values as targets (8 of them)
  heldout    value_heldout split: fit on 48 values; steer withheld-value clips to the 16 withheld targets
  ends       value_heldout_ends split: fit on the middle values; targets are the withheld extremes
Read-out: ridge probe fit on the val clips of the same split (never used for fitting); for speed and
acceleration also a θ probe (off-target drift). Nearest-point-on-curve decoding is reported too.
Saves outputs/results/steer_manifold.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.inlp import run_inlp
from src.manifold import fit_manifold, uncoord
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL, load_splits
from src.steer import clamp_coords, steer, target_vec

L, N_PCS, N_CLAMP = 12, 64, 10
KBEST = {n: json.loads((OUT / "results" / "manifolds.json").read_text())[n]["K"] for n in DATASETS}
ALPHA = {n: json.loads((OUT / "results" / "probe_layers.json").read_text())[n]["raw"]["vjepa2"][L]["alpha"] for n in DATASETS}


def err(name, a, b):
    return circ_err(a, b) if name == "direction" else np.abs(a - b)


def decode(name, probe, X):
    y = probe.predict(X)
    return angle_deg(y) if name == "direction" else y[:, 0]


res = {"layer": L, "n_pcs": N_PCS, "n_clamp": N_CLAMP, "K": KBEST}
for name in DATASETS:
    df = load_manifest(name)
    lab = df[LABEL[name]].values
    y = targets(name, df)
    X = load_features(name)[:, L].astype(np.float64)
    sp = load_splits()[name]
    res[name] = {}
    for cond, split, steer_key, tvals in (
            ("seen", "primary", "test", None),
            ("heldout", "value_heldout", "heldout_values", np.array(sp["heldout_values"])),
            ("ends", "value_heldout_ends", "heldout_values", np.array(sp["heldout_values_ends"]))):
        idx = split_idx(name, split)
        tr, va, st = idx["train"], idx["val"], idx[steer_key]
        if tvals is None:
            tvals = {"direction": np.arange(0, 360, 45.0), "speed": np.linspace(0.25, 4.0, 8),
                     "acceleration": np.linspace(0.25, 10.0, 8)}[name]
        # fits on train only
        man = fit_manifold(name, X[tr], lab[tr], N_PCS, "lsq", KBEST[name])
        man_i = fit_manifold(name, X[tr], lab[tr], N_PCS, "interp", 0)
        inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, N_CLAMP, eval_splits=("val",))
        # read-out on val only
        probe = fit_ridge(X[va], y[va], alpha=ALPHA[name])
        th_probe = None if name == "direction" else fit_ridge(X[va], targets("direction", df)[va], alpha=ALPHA[name])
        Xs = X[st]
        truth = lab[st]
        th_true = df.theta_degrees.values[st]
        r = res[name][cond] = {"n_steered": int(len(st)), "targets": [float(t) for t in tvals],
                               "probe_floor": float(err(name, decode(name, probe, Xs), truth).mean()),
                               "methods": {}}
        if th_probe is not None:
            r["theta_floor"] = float(circ_err(decode("direction", th_probe, Xs), th_true).mean())
        for method in ("clamp", "spline", "spline_span", "spline_interp", "spline_d", "chord"):
            to_t, to_o, th_d, near = [], [], [], []
            for t in tvals:
                if method == "clamp":
                    V, c = clamp_coords(inlp, N_CLAMP, target_vec(name, float(t)))
                    Xe = steer(inlp.z(Xs), V, c) * inlp.sd + inlp.mu
                elif method in ("spline", "spline_interp"):
                    Xe = (man if method == "spline" else man_i).steer(Xs, float(t))
                elif method == "spline_span":
                    Xe = man.steer_span(Xs, float(t))
                elif method == "spline_d":
                    u0, _ = man.nearest_u(man.project(Xs))
                    Xe = Xs + man.lift(man.point(float(t)) - man.curve(u0)) - man.mu
                else:  # chord endpoint = centroid of the nearest seen value
                    j = np.argmin(np.abs(man.values - t))
                    Xe = Xs - man.lift(man.project(Xs)) + man.lift(man.C[j][None])
                d = decode(name, probe, Xe)
                to_t.append(err(name, d, float(t)).mean())
                to_o.append(err(name, d, truth).mean())
                u_hat, _ = man.nearest_u(man.project(Xe))
                near.append(err(name, uncoord(name, u_hat), float(t)).mean())
                if th_probe is not None:
                    th_d.append(circ_err(decode("direction", th_probe, Xe), th_true).mean())
            r["methods"][method] = {"to_target": float(np.mean(to_t)), "to_original": float(np.mean(to_o)),
                                    "nearest_point_to_target": float(np.mean(near))}
            if th_d:
                r["methods"][method]["theta_drift"] = float(np.mean(th_d))
        print(f"{name:12s} {cond:8s} floor {r['probe_floor']:.3g}  " +
              "  ".join(f"{m}:{v['to_target']:.3g}" for m, v in r["methods"].items()) +
              (f"  | θ drift clamp {r['methods']['clamp']['theta_drift']:.1f} spline {r['methods']['spline']['theta_drift']:.1f} span {r['methods']['spline_span']['theta_drift']:.1f} (floor {r['theta_floor']:.1f})"
               if th_probe is not None else ""), flush=True)
(OUT / "results" / "steer_manifold.json").write_text(json.dumps(res, indent=1))
