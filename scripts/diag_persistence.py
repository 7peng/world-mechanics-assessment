"""Diagnostic for Experiment 3b: after steering at layer L, is the injected delta still present
in the residual stream downstream (just not read), or is it cancelled?

For a subset of test clips, direction/speed/acceleration, N=20, all targets:
  retained(l)  = <p_steer(l) - p_clean(l), Δ> / ||Δ||²   (fraction of Δ still along its own direction)
  layerL_probe(l): the layer-L val-fit probe applied to downstream steered features (error to target)
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.data import DATASETS, OUT, load_manifest, read_video
from src.features import load_features, split_idx
from src.model import load_encoder, preprocess, run_from, run_to
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.steer import clamp_coords, target_vec

L = json.loads((OUT / "results" / "inlp.json").read_text())["layer"]
N, n_clips = 20, 32
TARGETS = {"direction": [0, 90, 180, 270], "speed": [0.5, 2.0, 3.5], "acceleration": [1.0, 5.0, 9.0]}
model = load_encoder()
out = {}
for name in DATASETS:
    df = load_manifest(name)
    y, idx = targets(name, df), split_idx(name)
    F = load_features(name).astype(np.float64)
    inlp = pickle.load(open(OUT / "inlp" / f"inlp_{name}_L{L}.pkl", "rb"))
    alpha = json.loads((OUT / "results" / "probe_layers.json").read_text())[name]["raw"]["vjepa2"][L]["alpha"]
    pL = fit_ridge(F[idx["val"], L], y[idx["val"]], alpha=alpha)
    rows = idx["test"][:n_clips]
    ret, perr = [], []
    for s in range(0, n_clips, 4):
        x = preprocess(np.stack([read_video(df.video[i]) for i in rows[s:s + 4]])).cuda()
        h = run_to(model, x, L)
        clean = run_from(model, h, L).double().cpu().numpy()          # [B, 25-L, d]
        xL = h.mean(1).double().cpu().numpy()
        for t in TARGETS[name]:
            V, c = clamp_coords(inlp, N, target_vec(name, t))
            z = inlp.z(xL)
            D = (-(z @ V) @ V.T + c[None] @ V.T) * inlp.sd                # raw delta [B, d]
            st = run_from(model, h + torch.from_numpy(D).float().cuda()[:, None], L).double().cpu().numpy()
            diff = st - clean
            ret.append(np.einsum("bld,bd->bl", diff, D) / (D ** 2).sum(1)[:, None])
            pr = pL.predict(st.reshape(-1, st.shape[-1])).reshape(len(D), -1, y.shape[1])
            d = angle_deg(pr.reshape(-1, 2)).reshape(len(D), -1) if name == "direction" else pr[..., 0]
            perr.append(circ_err(d, t) if name == "direction" else np.abs(d - t))
    out[name] = {"layers": list(range(L, 25)),
                 "retained_frac": np.concatenate(ret).mean(0).tolist(),
                 "layerL_probe_err": np.concatenate(perr).mean(0).tolist()}
    print(name, "retained:", " ".join(f"{v:.2f}" for v in out[name]["retained_frac"]))
    print(name, "layer-L probe err:", " ".join(f"{v:.3g}" for v in out[name]["layerL_probe_err"]))
(OUT / "results" / "diag_persistence.json").write_text(json.dumps(out, indent=1))
