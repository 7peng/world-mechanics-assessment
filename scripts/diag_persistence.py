"""Diagnostic for Experiment 3b: after steering at layer L, how much of the injected delta stays
along its own direction downstream, compared with baselines (random Δ of equal norm, natural
clip-to-clip difference)?

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
    ret, perr, ret_rand, ret_nat, perr_clean = [], [], [], [], []
    rng = np.random.default_rng(0)
    other = idx["val"]  # donor clips for the natural-difference baseline (disjoint from the test clips)
    for s in range(0, n_clips, 4):
        x = preprocess(np.stack([read_video(df.video[i]) for i in rows[s:s + 4]])).cuda()
        h = run_to(model, x, L)
        clean = run_from(model, h, L).double().cpu().numpy()          # [B, 25-L, d]
        xL = h.mean(1).double().cpu().numpy()
        # baseline: layer-L probe applied to the *unsteered* downstream features (error to the true label)
        prc = pL.predict(clean.reshape(-1, clean.shape[-1])).reshape(len(xL), -1, y.shape[1])
        tr_ = y[rows[s:s + 4]]
        if name == "direction":
            perr_clean.append(circ_err(angle_deg(prc.reshape(-1, 2)).reshape(len(xL), -1), angle_deg(tr_)[:, None]))
        else:
            perr_clean.append(np.abs(prc[..., 0] - tr_[:, :1]))
        # baseline: natural clip-to-clip difference, Δ = x_other - x at layer L (other clip from val)
        donors = rng.choice(other, len(xL), replace=False)
        Dn = F[donors, L] - xL
        stn = run_from(model, h + torch.from_numpy(Dn).float().cuda()[:, None], L).double().cpu().numpy()
        ret_nat.append(np.einsum("bld,bd->bl", stn - clean, Dn) / (Dn ** 2).sum(1)[:, None])
        for t in TARGETS[name]:
            V, c = clamp_coords(inlp, N, target_vec(name, t))
            z = inlp.z(xL)
            D = (-(z @ V) @ V.T + c[None] @ V.T) * inlp.sd                # raw delta [B, d]
            st = run_from(model, h + torch.from_numpy(D).float().cuda()[:, None], L).double().cpu().numpy()
            diff = st - clean
            ret.append(np.einsum("bld,bd->bl", diff, D) / (D ** 2).sum(1)[:, None])
            # baseline: isotropic random raw-space Δ with the same norm
            G = rng.standard_normal(D.shape)
            G *= (np.linalg.norm(D, axis=1) / np.linalg.norm(G, axis=1))[:, None]
            stg = run_from(model, h + torch.from_numpy(G).float().cuda()[:, None], L).double().cpu().numpy()
            ret_rand.append(np.einsum("bld,bd->bl", stg - clean, G) / (G ** 2).sum(1)[:, None])
            pr = pL.predict(st.reshape(-1, st.shape[-1])).reshape(len(D), -1, y.shape[1])
            d = angle_deg(pr.reshape(-1, 2)).reshape(len(D), -1) if name == "direction" else pr[..., 0]
            perr.append(circ_err(d, t) if name == "direction" else np.abs(d - t))
    out[name] = {"layers": list(range(L, 25)),
                 "retained_frac": np.concatenate(ret).mean(0).tolist(),
                 "layerL_probe_err": np.concatenate(perr).mean(0).tolist(),
                 "retained_frac_random": np.concatenate(ret_rand).mean(0).tolist(),
                 "retained_frac_natural": np.concatenate(ret_nat).mean(0).tolist(),
                 "layerL_probe_err_clean": np.concatenate(perr_clean).mean(0).tolist()}
    print(name, "retained:", " ".join(f"{v:.2f}" for v in out[name]["retained_frac"]))
    print(name, "retained (random Δ):", " ".join(f"{v:.2f}" for v in out[name]["retained_frac_random"]))
    print(name, "retained (natural Δ):", " ".join(f"{v:.2f}" for v in out[name]["retained_frac_natural"]))
    print(name, "layer-L probe err:", " ".join(f"{v:.3g}" for v in out[name]["layerL_probe_err"]))
    print(name, "layer-L probe err, unsteered:", " ".join(f"{v:.3g}" for v in out[name]["layerL_probe_err_clean"]))
(OUT / "results" / "diag_persistence.json").write_text(json.dumps(out, indent=1))
