"""Experiment 3b: multi-probe subspace steering propagated through the network.

Stronger held-out test than the paper's same-layer read-out:
  1. Steering subspace + probes: INLP on *train* pooled activations at layer L (Experiment 2).
  2. For each *test* clip and target, compute the pooled steered vector x* (Eq. 8) and add
     delta = x* - x to *every token* of the layer-L residual stream (pooled mean becomes x*,
     token-level structure is preserved).
  3. Run blocks L+1..24 of the frozen encoder, mean-pool each layer.
  4. Read out with ridge probes fit on clean *val* activations at each layer (never seen by steering).
Controls: matched-norm random delta (random subspace of equal dim), unsteered pass.
Off-target: θ read-out (speed/acceleration sets, where θ varies).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tqdm import tqdm

from src.data import OUT, load_manifest, read_video
from src.features import load_features, split_idx
from src.model import load_encoder, preprocess, run_from, run_to
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.steer import clamp_coords, target_vec

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--n-list", default="1,3,5,10,20,50")
ap.add_argument("--clips-per-batch", type=int, default=2)
args = ap.parse_args()
name = args.dataset

L = json.loads((OUT / "results" / "inlp.json").read_text())["layer"]
N_LIST = [int(n) for n in args.n_list.split(",")]
N_RANDOM = 20
TARGETS = {"direction": np.arange(0, 360, 45.0),
           "speed": np.linspace(0.25, 4.0, 8), "acceleration": np.linspace(0.25, 10.0, 8)}[name]
rng = np.random.default_rng(0)

df = load_manifest(name)
y, idx = targets(name, df), split_idx(name)
F = load_features(name).astype(np.float64)
inlp = pickle.load(open(OUT / "inlp" / f"inlp_{name}_L{L}.pkl", "rb"))
alphas = [m["alpha"] for m in json.loads((OUT / "results" / "probe_layers.json").read_text())[name]["raw"]["vjepa2"]]
layers = list(range(L, 25))
evalp = {l: fit_ridge(F[idx["val"], l], y[idx["val"]], alpha=alphas[l]) for l in layers}
thetap = None
if name != "direction":
    yth = targets("direction", df)
    thetap = {l: fit_ridge(F[idx["val"], l], yth[idx["val"]], alpha=alphas[l]) for l in layers}

# steering deltas in raw feature space, per condition: (kind, N, target) -> function of pooled x
conds = [("clean", 0, None)] + [("steer", N, t) for N in N_LIST for t in TARGETS] + \
        [("random", N_RANDOM, t) for t in TARGETS]
coords = {(N, t): clamp_coords(inlp, N, target_vec(name, t)) for N in N_LIST for t in TARGETS}
Rb, _ = np.linalg.qr(rng.standard_normal((F.shape[2], N_RANDOM * (2 if name == "direction" else 1))))


def deltas(x: np.ndarray, kind, N, t) -> np.ndarray:
    """x: pooled raw [B, d] at layer L -> raw delta [B, d]."""
    if kind == "clean":
        return np.zeros_like(x)
    z = inlp.z(x)
    V, c = coords[(N if kind == "steer" else N_RANDOM, t)]
    dz = -(z @ V) @ V.T + c[None] @ V.T
    if kind == "random":
        g = rng.standard_normal((len(z), Rb.shape[1])) @ Rb.T
        dz = g * (np.linalg.norm(dz, axis=1) / np.linalg.norm(g, axis=1))[:, None]
    return dz * inlp.sd


model = load_encoder()
test = idx["test"]
pooled = np.zeros((len(conds), len(test), len(layers), F.shape[2]), np.float32)
for s in tqdm(range(0, len(test), args.clips_per_batch), desc=name):
    rows = test[s : s + args.clips_per_batch]
    x = preprocess(np.stack([read_video(df.video[i]) for i in rows])).cuda()
    h = run_to(model, x, L)                                     # [B, T, D]
    xL = h.mean(1).double().cpu().numpy()
    for ci, (kind, N, t) in enumerate(conds):
        d = torch.from_numpy(deltas(xL, kind, N, t)).float().cuda()
        pooled[ci, s : s + len(rows)] = run_from(model, h + d[:, None, :], L).cpu().numpy()

# ---- read-outs
yt = y[test]
truth = angle_deg(yt) if name == "direction" else yt[:, 0]
dec = lambda p, X: angle_deg(p.predict(X)) if name == "direction" else p.predict(X)[:, 0]
err = (lambda a, b: circ_err(a, b)) if name == "direction" else (lambda a, b: np.abs(a - b))
out = {"layer": L, "layers": layers, "targets": TARGETS.tolist(), "n_list": N_LIST, "n_random": N_RANDOM,
       "conditions": []}
for ci, (kind, N, t) in enumerate(conds):
    row = {"kind": kind, "N": N, "target": t, "to_target": [], "to_truth": [], "decoded_mean": [],
           "delta_norm_final": []}
    if thetap:
        row["theta_mae"] = []
    for li, l in enumerate(layers):
        X = pooled[ci, :, li].astype(np.float64)
        d = dec(evalp[l], X)
        row["to_truth"].append(float(err(d, truth).mean()))
        if t is not None:
            row["to_target"].append(float(err(d, t).mean()))
            row["decoded_mean"].append(float(t + ((d - t + 180) % 360 - 180).mean()) if name == "direction"
                                       else float(d.mean()))
        if thetap:
            row["theta_mae"].append(float(circ_err(angle_deg(thetap[l].predict(X)),
                                                   df.theta_degrees.values[test]).mean()))
    out["conditions"].append(row)

(OUT / "results").mkdir(exist_ok=True)
(OUT / "results" / f"steer_propagate_{name}.json").write_text(json.dumps(out, indent=1))
clean = out["conditions"][0]
print(f"clean to-truth @L{L}: {clean['to_truth'][0]:.3f}, @24: {clean['to_truth'][-1]:.3f}")
for N in N_LIST:
    rs = [c for c in out["conditions"] if c["kind"] == "steer" and c["N"] == N]
    tt = np.mean([c["to_target"] for c in rs], 0)
    print(f"N={N:3d} to-target @L{L} {tt[0]:.3f}  @L{L+4} {tt[4]:.3f}  @24 {tt[-1]:.3f}"
          + (f"  θ MAE @24 {np.mean([c['theta_mae'][-1] for c in rs]):.1f}" if thetap else ""))
rs = [c for c in out["conditions"] if c["kind"] == "random"]
print(f"random(N={N_RANDOM}) to-target @24 {np.mean([c['to_target'][-1] for c in rs]):.3f}")
