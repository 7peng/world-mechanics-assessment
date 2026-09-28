"""Behavioral steering test via the V-JEPA 2 predictor.

Behavior = the predictor's forecast of the last 4 time steps (tokens 1024-2047) from the first 4
(tokens 0-1023). Readout = ridge from the mean predicted target token to the variable, fit on
predictions for clean TRAIN clips (never on steered data). Steering edits layer 12 of the encoder
(added to every token; context and target tokens alike, since the whole clip passes through the
encoder), the encoder runs to the end, and only the context tokens reach the predictor.

Conditions: unsteered; multi-probe subspace steering (N = 10); spline steering (replace PCA-64);
random edit with the norm of the subspace edit. Targets: 4 per variable. Test clips: 96.
Metric: % of clips whose forecast is decoded within tolerance of the target (direction ±15°,
speed ±0.375 m/s, acceleration ±0.975 m/s²), and the same for the clip's original label.
Saves outputs/results/steer_behavior.json.
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
from src.manifold import fit_manifold
from src.model import load_encoder, preprocess, run_from_tokens, run_to
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL
from src.steer import clamp_coords, steer, target_vec

L, N_CLAMP, N_PCS = 12, 10, 64
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}
TARGETS = {"direction": [0.0, 90.0, 180.0, 270.0], "speed": [0.5, 1.5, 2.5, 3.5], "acceleration": [1.0, 4.0, 7.0, 10.0]}
KBEST = {n: json.loads((OUT / "results" / "manifolds.json").read_text())[n]["K"] for n in DATASETS}
N_TRAIN, N_TEST, BS = 400, 96, 8
err = lambda n, a, b: circ_err(a, b) if n == "direction" else np.abs(a - b)
dec = lambda n, p, X: angle_deg(p.predict(X)) if n == "direction" else p.predict(X)[:, 0]
model = load_encoder()
CTX = torch.arange(0, 1024, device="cuda")
TGT = torch.arange(1024, 2048, device="cuda")


@torch.no_grad()
def forecast(h_L, delta=None):
    """h_L: layer-L residual [B, 2048, D]; delta: pooled raw edit [B, D] or None. Returns mean predicted
    target token [B, D]."""
    if delta is not None:
        h_L = h_L + torch.from_numpy(delta).float().cuda()[:, None, :]
    enc = model.encoder.layernorm(run_from_tokens(model, h_L, L))
    B = enc.shape[0]
    pred = model.predictor(enc, context_mask=[CTX[None].repeat(B, 1)], target_mask=[TGT[None].repeat(B, 1)]).last_hidden_state
    return pred.mean(1).double().cpu().numpy()


res = {"layer": L, "tolerance": TOL, "targets": TARGETS}
rng = np.random.default_rng(0)
for name in DATASETS:
    df = load_manifest(name)
    lab = df[LABEL[name]].values
    y = targets(name, df)
    X = load_features(name)[:, L].astype(np.float64)
    idx = split_idx(name)
    tr, va, te = idx["train"], idx["val"], idx["test"][:N_TEST]
    man = fit_manifold(name, X[tr], lab[tr], N_PCS, "lsq", KBEST[name])
    inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, N_CLAMP, eval_splits=("val",))
    # behavior readout: fit on clean forecasts of train clips
    Ftr = []
    for s in range(0, N_TRAIN, BS):
        rows = tr[s:s + BS]
        x = preprocess(np.stack([read_video(df.video[i]) for i in rows])).cuda()
        Ftr.append(forecast(run_to(model, x, L)))
    Ftr = np.concatenate(Ftr)
    n_in = int(0.8 * len(Ftr))
    readout = fit_ridge(Ftr[:n_in], y[tr[:n_in]], Ftr[n_in:], y[tr[n_in:N_TRAIN]])
    truth = lab[te]
    out = {c: {"to_target": [], "to_original": []} for c in ("unsteered", "subspace", "spline", "random")}
    for s in range(0, len(te), BS):
        rows = te[s:s + BS]
        x = preprocess(np.stack([read_video(df.video[i]) for i in rows])).cuda()
        h = run_to(model, x, L)
        xL = h.mean(1).double().cpu().numpy()
        d0 = dec(name, readout, forecast(h))
        out["unsteered"]["to_original"].append(err(name, d0, truth[s:s + BS]))
        for t in TARGETS[name]:
            V, c = clamp_coords(inlp, N_CLAMP, target_vec(name, float(t)))
            d_sub = steer(inlp.z(xL), V, c) * inlp.sd + inlp.mu - xL
            d_spl = man.steer(xL, float(t)) - xL
            g = rng.standard_normal(d_sub.shape)
            d_rnd = g * (np.linalg.norm(d_sub, axis=1) / np.linalg.norm(g, axis=1))[:, None]
            out["unsteered"]["to_target"].append(err(name, d0, float(t)))
            for cname, D in (("subspace", d_sub), ("spline", d_spl), ("random", d_rnd)):
                d = dec(name, readout, forecast(h, D))
                out[cname]["to_target"].append(err(name, d, float(t)))
                out[cname]["to_original"].append(err(name, d, truth[s:s + BS]))
    r = res[name] = {}
    for cname, v in out.items():
        tt = np.concatenate(v["to_target"]); to = np.concatenate(v["to_original"])
        r[cname] = {"within_target_pct": float((tt <= TOL[name]).mean() * 100), "within_original_pct": float((to <= TOL[name]).mean() * 100),
                    "err_to_target": float(tt.mean()), "err_to_original": float(to.mean())}
    print(name, {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 1) for kk, vv in v.items()}) for k, v in r.items()}, flush=True)
(OUT / "results" / "steer_behavior.json").write_text(json.dumps(res, indent=1))

