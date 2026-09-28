"""Token-level subspace steering, judged by the predictor's forecast (context-only protocol, src/behavior.py).

Instead of one pooled delta added to all tokens, each token is edited individually: a shared token-level
probe (same weights at every position) is fit on individual residual-stream tokens of train clips at
layer L, INLP gives N probe directions, and every token of a test clip is clamped so all N token probes
read the target (Part 1 Eq. 8, per token). Then the encoder finishes and the predictor forecasts.
Readout as in behavior_sweep (ridge on clean train forecasts).
Also reported: the same token clamp but restricted to the context tokens (the only ones the predictor
receives at the end). Writes outputs/results/behavior_token_<dataset>.json incrementally.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.behavior import encode_to, fit_readout, forecast_from
from src.data import OUT, load_manifest, read_video
from src.features import split_idx
from src.inlp import run_inlp
from src.model import load_encoder, preprocess
from src.probes import angle_deg, circ_err, targets
from src.splits import LABEL
from src.steer import clamp_coords, steer, target_vec

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--layers", default="4,8,12,16,20,24")
ap.add_argument("--n-probes", type=int, default=10)
ap.add_argument("--n-train-clips", type=int, default=320)
ap.add_argument("--tokens-per-clip", type=int, default=192)
ap.add_argument("--n-test", type=int, default=64)
args = ap.parse_args()
name = args.dataset
LAYERS = [int(v) for v in args.layers.split(",")]
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}[name]
TARGETS = {"direction": [0.0, 90.0, 180.0, 270.0], "speed": [0.5, 1.5, 2.5, 3.5], "acceleration": [1.0, 4.0, 7.0, 10.0]}[name]
BS = 8
err = (lambda a, b: circ_err(a, b)) if name == "direction" else (lambda a, b: np.abs(a - b))
dec = (lambda p, X: angle_deg(p.predict(X))) if name == "direction" else (lambda p, X: p.predict(X)[:, 0])

model = load_encoder()
df = load_manifest(name)
lab = df[LABEL[name]].values
y = targets(name, df)
idx = split_idx(name)
tr, va, te = idx["train"], idx["val"], idx["test"][: args.n_test]
readout = fit_readout(model, name, df, y, tr, va)
rng = np.random.default_rng(0)

# token features for all requested layers in one pass over a subset of train (and val) clips
rows_fit = np.r_[tr[: args.n_train_clips], va[: args.n_train_clips // 4]]
n_fit_tr = min(args.n_train_clips, len(tr))
tok = {L: [] for L in LAYERS}
for s in range(0, len(rows_fit), BS):
    rows = rows_fit[s:s + BS]
    x = preprocess(np.stack([read_video(df.video[i])[:8] for i in rows])).cuda()
    with torch.no_grad():
        hs = model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True).hidden_states
    sel = torch.from_numpy(np.stack([rng.choice(1024, args.tokens_per_clip, replace=False) for _ in rows])).cuda()
    for L in LAYERS:
        tok[L].append(torch.gather(hs[L], 1, sel[..., None].expand(-1, -1, hs[L].shape[-1])).float().cpu().numpy())
res = {"dataset": name, "tolerance": TOL, "layers": {}}
out_path = OUT / "results" / f"behavior_token_ctx_{name}.json"
for L in LAYERS:
    T = np.concatenate(tok[L])                                  # [n_clips, tpc, D]
    n_clips = T.shape[0]
    Xt = T.reshape(-1, T.shape[-1])
    yt = np.repeat(y[rows_fit[:n_clips]], args.tokens_per_clip, axis=0)
    ntr = n_fit_tr * args.tokens_per_clip
    inlp = run_inlp(name, Xt, yt, {"train": np.arange(ntr), "val": np.arange(ntr, len(Xt))}, args.n_probes, eval_splits=("val",))
    tok_r2 = inlp.val[0]["r2"]
    dec_all = {t: [] for t in TARGETS}
    dec_ctx = {t: [] for t in TARGETS}
    for s in range(0, len(te), BS):
        rows = te[s:s + BS]
        h = encode_to(model, df, rows, L)                       # [B, 2048, D]
        H = h.double().cpu().numpy()
        B = H.shape[0]
        Z = inlp.z(H.reshape(-1, H.shape[-1]))
        for t in TARGETS:
            V, c = clamp_coords(inlp, args.n_probes, target_vec(name, float(t)))
            He = (steer(Z, V, c) * inlp.sd + inlp.mu).reshape(H.shape)
            he = torch.from_numpy(He).float().cuda()
            dec_all[t].append(dec(readout, forecast_from(model, he, L)))
            # half-strength edit (interpolate halfway to the clamped value) for a dose check
            hm = 0.5 * (h + he)
            dec_ctx[t].append(dec(readout, forecast_from(model, hm, L)))
    def score(d):
        per = {str(t): float((err(np.concatenate(v), float(t)) <= TOL).mean() * 100) for t, v in d.items()}
        return {"within_pct": float(np.mean(list(per.values()))), "per_target": per,
                "mean_decoded": {str(t): (float(np.degrees(np.angle(np.exp(1j * np.radians(np.concatenate(v))).mean())) % 360)
                                          if name == "direction" else float(np.concatenate(v).mean())) for t, v in d.items()}}
    res["layers"][str(L)] = {"token_probe_val_r2": float(tok_r2), "all_tokens": score(dec_all), "half_strength": score(dec_ctx)}
    print(f"L{L} token-probe R² {tok_r2:.3f} | token clamp {res['layers'][str(L)]['all_tokens']['within_pct']:.1f}% "
          f"| half {res['layers'][str(L)]['half_strength']['within_pct']:.1f}%  means {res['layers'][str(L)]['all_tokens']['mean_decoded']}", flush=True)
    out_path.write_text(json.dumps(res, indent=1))
