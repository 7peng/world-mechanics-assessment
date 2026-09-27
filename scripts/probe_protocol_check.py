"""Why does direction decode at layer 1 for us but only at layer ~8 in Joseph et al.?
Tests probe-protocol and data-regime differences on cached features.

Probes (all: 5-fold stratified CV; hyper-parameters chosen on one fixed inner 80/20 split of the
training folds per fold, shared by all layers; reported on the held-out fold):
  ridge_z        : ours (z-scored features, closed-form ridge)
  ridge_raw      : closed-form ridge on centered raw features (alpha grid scaled to the features)
  gd_paper       : paper App. B: linear probe trained by AdamW, lr {1e-4..5e-3} x wd {0.01..0.8},
                   raw features, 100 epochs, batch 32
  gd_paper_z     : same on z-scored features
  gd_paper_long  : gd_paper with 1000 epochs (tests whether gd_paper is limited by its training budget)
Data regimes (direction set):
  full      : all 1500 clips
  paperlike : constant-velocity clips at the 8 directions 0,45,..,315 (their grid); 96 clips
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.data import OUT, load_manifest
from src.features import load_features
from src.gd_probe import predict, to_dev, train_linear
from src.probes import fit_ridge, metrics, targets

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WDS = [0.01, 0.1, 0.4, 0.8]
CFG_LR = [lr for lr in LRS for _ in WDS]
CFG_WD = [wd for _ in LRS for wd in WDS]
PROBES = ["ridge_z", "ridge_raw", "gd_paper", "gd_paper_z", "gd_paper_long"]


def gd_all_layers(Xtr, ytr, Xva, yva, Xte, epochs, seed):
    """Train 20 configs x all layers at once; per layer pick config by inner-val MSE.
    X*: [L, n, d] numpy. Returns test predictions [L, n_te, k] and chosen config per layer."""
    W, b = train_linear(to_dev(Xtr), to_dev(ytr), CFG_LR, CFG_WD, epochs, decoupled=True, seed=seed)
    with torch.no_grad():
        va_mse = ((predict(to_dev(Xva), W, b) - to_dev(yva)) ** 2).mean((2, 3))      # [L, C]
        va_mse = torch.nan_to_num(va_mse, nan=float("inf"))
        best = va_mse.argmin(1)                                                       # [L]
        pte = predict(to_dev(Xte), W, b)                                              # [L, C, n, k]
        pte = pte[torch.arange(len(best)), best]
    return pte.cpu().numpy(), best.cpu().numpy()


def run(name, F, y, strata, layers, seed=0):
    rng = np.random.default_rng(seed)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(F[:, 0], strata))
    inner = []
    for tr, te in folds:  # one fixed inner split per fold, shared by every layer and probe
        p = rng.permutation(tr)
        inner.append((p[len(p) // 5:], p[: len(p) // 5], te))
    acc = {p: [[] for _ in layers] for p in PROBES}      # per layer, per fold metrics
    chosen = {p: [[] for _ in layers] for p in PROBES}
    for fi, (tr, va, te) in enumerate(inner):
        Xl = F[:, layers].astype(np.float64)              # [N, L, d]
        for li, L in enumerate(layers):
            X = Xl[:, li]
            q = fit_ridge(X[tr], y[tr], X[va], y[va])
            acc["ridge_z"][li].append(metrics(name, y[te], q.predict(X[te]))); chosen["ridge_z"][li].append(q.alpha)
            # raw features: scale the alpha grid by the mean eigenvalue of the (centered) Gram matrix
            scale = float(((X[tr] - X[tr].mean(0)) ** 2).sum() / X.shape[1])
            q = fit_ridge(X[tr], y[tr], X[va], y[va], standardize=False, alphas=scale * np.logspace(-8, 4, 25))
            acc["ridge_raw"][li].append(metrics(name, y[te], q.predict(X[te])))
            chosen["ridge_raw"][li].append(q.alpha / scale)
        XT = Xl.transpose(1, 0, 2)                         # [L, N, d]
        mu, sd = XT[:, tr].mean(1, keepdims=True), XT[:, tr].std(1, keepdims=True) + 1e-6
        for p, Xin, ep in (("gd_paper", XT, 100), ("gd_paper_z", (XT - mu) / sd, 100), ("gd_paper_long", XT, 1000)):
            pte, best = gd_all_layers(Xin[:, tr], y[tr], Xin[:, va], y[va], Xin[:, te], ep, seed=fi)
            for li in range(len(layers)):
                acc[p][li].append(metrics(name, y[te], pte[li]))
                chosen[p][li].append(int(best[li]))
    out = {}
    for p in PROBES:
        out[p] = [{"r2": float(np.mean([a["r2"] for a in acc[p][li]])),
                   "r2_std": float(np.std([a["r2"] for a in acc[p][li]])),
                   "circ_mae_deg": float(np.mean([a["circ_mae_deg"] for a in acc[p][li]])),
                   "chosen": chosen[p][li]} for li in range(len(layers))]
    for li, L in enumerate(layers):
        print(L, " ".join(f"{p}: {out[p][li]['r2']:.3f}" for p in PROBES), flush=True)
    return out


if __name__ == "__main__":
    name = "direction"
    df = load_manifest(name)
    F, y = load_features(name), targets(name, df)
    th = np.round(df.theta_degrees.values, 4)
    layers = [1, 2, 4, 6, 8, 9, 10, 12, 16, 20, 24]
    res = {"layers": layers, "probes": PROBES}
    print("== paperlike: constant velocity, 8 directions")
    m = (df.motion.values == "velocity") & np.isin(th, np.arange(0, 360, 45.0))
    print("   clips:", m.sum())
    res["paperlike"] = run(name, F[m], y[m], np.unique(th[m], return_inverse=True)[1], layers)
    print("== full direction set")
    res["full"] = run(name, F, y, np.unique(th, return_inverse=True)[1], layers)
    (OUT / "results" / "probe_protocol_check.json").write_text(json.dumps(res, indent=1))
