"""Why does direction decode at layer 1 for us but only at layer ~8 in Joseph et al.?
Tests probe-protocol and data-regime differences on cached features.

Probes:
  ridge_z   : ours (z-scored features, closed-form ridge, alpha on inner val)
  ridge_raw : closed-form ridge on centered raw features
  gd_paper  : paper App. B: linear probe trained by AdamW, lr {1e-4,3e-4,1e-3,3e-3,5e-3} x
              wd {0.01,0.1,0.4,0.8}, best config by inner val; raw features, 100 epochs, batch 32
  gd_paper_z: same on z-scored features
Data regimes (direction set):
  full      : all 1500 clips, 5-fold CV
  paperlike : constant-velocity clips at the 8 directions 0,45,..,315 (their grid), 5-fold CV
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
from src.probes import fit_ridge, metrics, targets

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WDS = [0.01, 0.1, 0.4, 0.8]


def gd_probe(Xtr, ytr, Xva, yva, epochs=100, bs=32, seed=0):
    """Train all 20 (lr, wd) configs in parallel; return predict fn of best-on-val config."""
    torch.manual_seed(seed)
    dev = "cuda"
    Xtr, ytr = torch.tensor(Xtr, dtype=torch.float32, device=dev), torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva, yva = torch.tensor(Xva, dtype=torch.float32, device=dev), torch.tensor(yva, dtype=torch.float32, device=dev)
    cfgs = [(lr, wd) for lr in LRS for wd in WDS]
    d, k = Xtr.shape[1], ytr.shape[1]
    lin = [torch.nn.Linear(d, k).to(dev) for _ in cfgs]
    opts = [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd) for m, (lr, wd) in zip(lin, cfgs)]
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            for m, o in zip(lin, opts):
                o.zero_grad()
                torch.nn.functional.mse_loss(m(Xtr[b]), ytr[b]).backward()
                o.step()
    with torch.no_grad():
        losses = [torch.nn.functional.mse_loss(m(Xva), yva).item() for m in lin]
    best = lin[int(np.nanargmin(losses))]
    return lambda X: best(torch.tensor(X, dtype=torch.float32, device=dev)).detach().cpu().numpy()


def run(name, F, y, strata, layers, probes, seed=0):
    rng = np.random.default_rng(seed)
    out = {p: [] for p in probes}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(F[:, 0], strata))
    for L in layers:
        X = F[:, L].astype(np.float64)
        acc = {p: [] for p in probes}
        for tr, te in folds:
            tr = rng.permutation(tr)
            va, tr = tr[: len(tr) // 5], tr[len(tr) // 5:]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            for p in probes:
                if p == "ridge_z":
                    q = fit_ridge(X[tr], y[tr], X[va], y[va]); pred = q.predict(X[te])
                elif p == "ridge_raw":
                    q = fit_ridge(X[tr], y[tr], X[va], y[va], standardize=False,
                                  alphas=np.logspace(-2, 8, 21)); pred = q.predict(X[te])
                else:
                    f = (lambda A: (A - mu) / sd) if p.endswith("_z") else (lambda A: A)
                    pr = gd_probe(f(X[tr]), y[tr], f(X[va]), y[va]); pred = pr(f(X[te]))
                acc[p].append(metrics(name, y[te], pred))
        for p in probes:
            out[p].append({k: float(np.mean([a[k] for a in acc[p]])) for k in acc[p][0]})
        print(L, " ".join(f"{p}: R² {out[p][-1]['r2']:.3f} / {out[p][-1]['circ_mae_deg']:.1f}°" for p in probes), flush=True)
    return out


if __name__ == "__main__":
    name = "direction"
    df = load_manifest(name)
    F, y = load_features(name), targets(name, df)
    th = np.round(df.theta_degrees.values, 4)
    layers = [1, 2, 4, 6, 8, 9, 10, 12, 16, 20, 24]
    probes = ["ridge_z", "ridge_raw", "gd_paper", "gd_paper_z"]
    res = {}
    print("== paperlike: constant velocity, 8 directions")
    m = (df.motion.values == "velocity") & np.isin(th, np.arange(0, 360, 45.0))
    print("   clips:", m.sum())
    res["paperlike"] = run(name, F[m], y[m], np.unique(th[m], return_inverse=True)[1], layers, probes)
    print("== full direction set")
    res["full"] = run(name, F, y, np.unique(th, return_inverse=True)[1], layers, probes)
    res["layers"] = layers
    (OUT / "results" / "probe_protocol_check.json").write_text(json.dumps(res, indent=1))
