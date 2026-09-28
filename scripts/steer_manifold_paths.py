"""Part 2: path quality and downstream read-out for manifold vs linear steering.

(a) Paths. For each test clip and each of several (source-value, target-value) pairs, K = 21 waypoints
    from source to target: along the curve (u interpolated in intrinsic coordinate; shorter arc for
    direction), along the chord between the two curve endpoints in PCA space, and along the clamp
    (linear interpolation of the clamp coordinates between the clamp solutions for source and target).
    Waypoint activations are read out with the val-fit probe. Reported per method: mean absolute deviation
    of the decoded value from the ideal (linear in u) path; fraction of monotone steps; and off-manifold
    distance (distance of the PCA projection to the nearest curve point) averaged over waypoints.
(b) Downstream. Layer-12 edit (clamp N=10; spline replace) added to every token, blocks 13-24 run,
    read out with val-fit probes at layers 12, 16, 20, 24. Subset of 64 test clips, 4 targets.
Saves outputs/results/steer_manifold_paths.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.data import DATASETS, OUT, load_manifest, read_video
from src.features import load_features, split_idx
from src.inlp import run_inlp
from src.manifold import coord, fit_manifold, uncoord
from src.model import load_encoder, preprocess, run_from, run_to
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL
from src.steer import clamp_coords, steer, target_vec

L, N_PCS, N_CLAMP, K = 12, 64, 10, 21
KBEST = {n: json.loads((OUT / "results" / "manifolds.json").read_text())[n]["K"] for n in DATASETS}
ALPHA = {n: json.loads((OUT / "results" / "probe_layers.json").read_text())[n]["raw"]["vjepa2"][L]["alpha"] for n in DATASETS}
PAIRS = {"direction": [(0, 90), (45, 225), (300, 60), (180, 270)],
         "speed": [(0.5, 3.5), (3.5, 0.5), (1.0, 2.0), (0.25, 4.0)],
         "acceleration": [(1.0, 9.0), (9.0, 1.0), (2.0, 5.0), (0.25, 10.0)]}
TARGETS_DS = {"direction": [0, 90, 180, 270], "speed": [0.5, 1.5, 2.5, 3.5], "acceleration": [1, 4, 7, 10]}
err = lambda name, a, b: circ_err(a, b) if name == "direction" else np.abs(a - b)
dec = lambda name, p, X: angle_deg(p.predict(X)) if name == "direction" else p.predict(X)[:, 0]


def interp_u(name, u0, u1, t):
    if name == "direction":
        d = (u1 - u0 + np.pi) % (2 * np.pi) - np.pi          # shorter arc
        return (u0 + t * d) % (2 * np.pi)
    return u0 + t * (u1 - u0)


res = {"layer": L, "K": K}
model = None
for name in DATASETS:
    df = load_manifest(name)
    lab = df[LABEL[name]].values
    y = targets(name, df)
    F = load_features(name).astype(np.float64)
    X = F[:, L]
    idx = split_idx(name)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    man = fit_manifold(name, X[tr], lab[tr], N_PCS, "lsq", KBEST[name])
    inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, N_CLAMP, eval_splits=("val",))
    probe = fit_ridge(X[va], y[va], alpha=ALPHA[name])
    Xs = X[te]
    r = res[name] = {"paths": {}, "downstream": {}}
    # ---------------- (a) paths
    ts = np.linspace(0, 1, K)
    acc = {m: {"dev": [], "mono": [], "off": []} for m in ("curve", "chord", "clamp")}
    for v0, v1 in PAIRS[name]:
        u0, u1 = coord(name, v0), coord(name, v1)
        ideal = uncoord(name, interp_u(name, u0, u1, ts))                              # [K]
        P0, P1 = man.point(v0)[0], man.point(v1)[0]
        V0, c0 = clamp_coords(inlp, N_CLAMP, target_vec(name, float(v0)))
        V1, c1 = clamp_coords(inlp, N_CLAMP, target_vec(name, float(v1)))
        Z = man.project(Xs)
        for m in acc:
            decoded, offd = [], []
            for t, u_t in zip(ts, interp_u(name, u0, u1, ts)):
                if m == "curve":
                    T = man.curve(u_t)
                    Xe = Xs - man.lift(Z) + man.lift(np.broadcast_to(T, Z.shape))
                elif m == "chord":
                    T = (1 - t) * P0 + t * P1
                    Xe = Xs - man.lift(Z) + man.lift(np.broadcast_to(T[None], Z.shape))
                else:
                    c = (1 - t) * c0 + t * c1                                          # same V for both (V0 == V1)
                    Xe = steer(inlp.z(Xs), V0, c) * inlp.sd + inlp.mu
                decoded.append(dec(name, probe, Xe).mean() if name != "direction" else
                               np.degrees(np.angle(np.exp(1j * np.radians(dec(name, probe, Xe))).mean())) % 360)
                _, dmin = man.nearest_u(man.project(Xe))
                offd.append(dmin.mean())
            decoded = np.array(decoded)
            acc[m]["dev"].append(err(name, decoded, ideal).mean())
            step = np.array([(err(name, decoded[i + 1], decoded[i]) if False else
                              ((decoded[i + 1] - decoded[i] + 180) % 360 - 180) if name == "direction" else decoded[i + 1] - decoded[i])
                             for i in range(K - 1)])
            sgn = np.sign((u1 - u0 + np.pi) % (2 * np.pi) - np.pi) if name == "direction" else np.sign(v1 - v0)
            acc[m]["mono"].append(float((np.sign(step) == sgn).mean()))
            acc[m]["off"].append(float(np.mean(offd)))
            r["paths"].setdefault("decoded_example", {})[f"{m}:{v0}->{v1}"] = decoded.tolist()
        r["paths"].setdefault("ideal_example", {})[f"{v0}->{v1}"] = ideal.tolist()
    for m in acc:
        r["paths"][m] = {k: float(np.mean(v)) for k, v in acc[m].items()}
    r["paths"]["unsteered_off_manifold"] = float(man.nearest_u(man.project(Xs))[1].mean())
    print(name, "paths:", {m: {k: round(v, 3) for k, v in r["paths"][m].items()} for m in acc},
          "unsteered off", round(r["paths"]["unsteered_off_manifold"], 2), flush=True)
    # ---------------- (b) downstream
    if model is None:
        model = load_encoder()
    layers = [12, 16, 20, 24]
    probes_l = {l: fit_ridge(F[va, l], y[va], alpha=ALPHA[name]) for l in layers}
    sub = te[:64]
    out = {m: {l: [] for l in layers} for m in ("clamp", "spline")}
    for s in range(0, len(sub), 4):
        rows = sub[s:s + 4]
        x = preprocess(np.stack([read_video(df.video[i]) for i in rows])).cuda()
        h = run_to(model, x, L)
        xL = h.mean(1).double().cpu().numpy()
        for t in TARGETS_DS[name]:
            V, c = clamp_coords(inlp, N_CLAMP, target_vec(name, float(t)))
            edits = {"clamp": steer(inlp.z(xL), V, c) * inlp.sd + inlp.mu - xL,
                     "spline": man.steer(xL, float(t)) - xL}
            for m, D in edits.items():
                pooled = run_from(model, h + torch.from_numpy(D).float().cuda()[:, None], L).double().cpu().numpy()
                for li, l in enumerate(layers):
                    j = l - L
                    out[m][l].append(err(name, dec(name, probes_l[l], pooled[:, j]), float(t)))
    for m in out:
        r["downstream"][m] = {str(l): float(np.concatenate(out[m][l]).mean()) for l in layers}
    print(name, "downstream:", r["downstream"], flush=True)
(OUT / "results" / "steer_manifold_paths.json").write_text(json.dumps(res, indent=1))
