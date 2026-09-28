"""Can the sawtooth of Joseph et al. Fig. 4c be reproduced, and is it an optimisation artefact?

Orthogonal probe sequence (App. C.11: Adam lr 1e-3, wd 1e-4, 100 epochs, 80/20 split) on direction
at paper layer 8 (our index 9), under several data regimes. At every step k, alongside the Adam probe
that defines the projection, a closed-form ridge probe is fit on the *same* residual X^(k). If ridge
stays high where Adam dips, the dip is a failed fit, not missing information (information cannot
return after a projection).

Regimes:
  full       all 1,500 clips
  sub392     392 random clips (paper's dataset size), 64 directions
  dir8       clips at the paper's 8 directions (0, 45, ..., 315), both motion types (~188 clips)
  dir8_fixed dir8 with the same seed for every probe ("fixed random seed")
  dir8_long  dir8 with 1,000 epochs
Saves outputs/results/sawtooth.json and outputs/figures/report/supp_sawtooth.png.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import OUT, load_manifest
from src.features import load_features
from src.gd_probe import to_dev, train_linear
from src.probes import angle_deg, circ_err, fit_ridge, targets

LAYER = 9          # paper layer 8
N_PROBES = 60
df = load_manifest("direction")
F = load_features("direction")[:, LAYER].astype(np.float32)
y = targets("direction", df)
th = np.round(df.theta_degrees.values, 4)


def acc15(yhat, ytrue):
    return float((circ_err(angle_deg(yhat), angle_deg(ytrue)) <= 15).mean() * 100)


def sequence(rows, epochs=100, fixed_seed=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.permutation(rows)
    n_tr = int(0.8 * len(rows))
    tr, te = rows[:n_tr], rows[n_tr:]
    itr, iva = tr[: int(0.8 * n_tr)], tr[int(0.8 * n_tr):]          # inner split for ridge alpha only
    X = to_dev(F[rows])
    Y = to_dev(y[rows])
    trI, teI = torch.arange(n_tr, device="cuda"), torch.arange(n_tr, len(rows), device="cuda")
    itrI, ivaI = trI[: len(itr)], trI[len(itr):]
    out = {"adam": [], "ridge": [], "n_train": int(n_tr), "n_test": int(len(te))}
    for k in range(N_PROBES):
        W, b = train_linear(X[None, trI], Y[trI], [1e-3], [1e-4], epochs, decoupled=False,
                            seed=1000 if fixed_seed else 1000 + k)
        W, b = W[0, 0], b[0, 0]
        pred = (X[teI] @ W + b).cpu().numpy()
        out["adam"].append(acc15(pred, y[te]))
        Xn = X.cpu().numpy().astype(np.float64)
        r = fit_ridge(Xn[itrI.cpu().numpy()], y[itr], Xn[ivaI.cpu().numpy()], y[iva])
        out["ridge"].append(acc15(r.predict(Xn[teI.cpu().numpy()]), y[te]))
        Q, _ = torch.linalg.qr(W)                                      # project out the Adam probe's directions
        X = X - (X @ Q) @ Q.T
    return out


rng = np.random.default_rng(0)
all_rows = np.arange(len(df))
dir8 = np.nonzero(np.isin(th, np.arange(0, 360, 45.0)))[0]
regimes = {
    "full": dict(rows=all_rows),
    "sub392": dict(rows=rng.choice(all_rows, 392, replace=False)),
    "dir8": dict(rows=dir8),
    "dir8_fixed": dict(rows=dir8, fixed_seed=True),
    "dir8_long": dict(rows=dir8, epochs=1000),
}
res = {"layer_index": LAYER, "paper_layer": LAYER - 1, "n_probes": N_PROBES}
for name, kw in regimes.items():
    res[name] = sequence(**kw)
    a, r = np.array(res[name]["adam"]), np.array(res[name]["ridge"])
    d = np.diff(a)
    print(f"{name:11s} n_train={res[name]['n_train']:4d}  adam first 20: {' '.join(f'{v:.0f}' for v in a[:20])}")
    print(f"{'':11s} ridge first 20: {' '.join(f'{v:.0f}' for v in r[:20])}   "
          f"adam dips>30 followed by rise>20: {int(((d[:-1] < -30) & (d[1:] > 20)).sum())}   "
          f"ridge below adam anywhere: {int((r < a - 5).sum())}", flush=True)
(OUT / "results" / "sawtooth.json").write_text(json.dumps(res, indent=1))

# ---- figure
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False})
show = ["full", "dir8", "dir8_fixed", "dir8_long"]
fig, axes = plt.subplots(1, len(show), figsize=(13, 3.2), sharey=True)
for a, name in zip(axes, show):
    a.plot(range(1, N_PROBES + 1), res[name]["adam"], color="#1f77b4", lw=1.4)
    a.plot(range(1, N_PROBES + 1), res[name]["ridge"], color="#ff7f0e", lw=1.4)
    a.set_title(f"{name} (n_train = {res[name]['n_train']})"); a.set_xlabel("probe number"); a.set_ylim(0, 102)
axes[0].set_ylabel("accuracy within 15° (%)")
axes[-1].legend(handles=[plt.Line2D([], [], color="#1f77b4", label="Adam probe (defines the projection)"),
                         plt.Line2D([], [], color="#ff7f0e", label="ridge probe on the same residual")],
                loc="center left", bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "supp_sawtooth.png", dpi=150, bbox_inches="tight", pad_inches=0.1,
            bbox_extra_artists=[axes[-1].get_legend()] + fig.get_default_bbox_extra_artists())
