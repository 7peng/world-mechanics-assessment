"""Which protocol variant produces the period-2 sawtooth of Joseph et al. Fig. 4c?

Direction, paper layer 8 (our index 9), clips at the paper's 8 directions (~188 clips, 80/20).
Each variant produces a sequence of accuracy-within-15° values; we report lag-1 autocorrelation
(strongly negative = alternation), mean of odd and even entries, and the first 16 values.

Variants (probe = Adam as in App. C.11, or closed-form ridge):
  baseline        standard sequence: fit, record test accuracy, project out Q_k from train and test
  log_twice       per probe, record accuracy before and after projecting its own Q_k out of the test features
  test_unproj     project train only; test features stay unprojected
  zscore_mismatch probe fit on z-scored features; projection applied in raw feature space with the z-space W
  row_alternate   treat the sin row and the cos row of W_k as separate 1-dim probes, removed alternately;
                  accuracy uses the current 2-output probe
Saves outputs/results/sawtooth_mechanisms.json and outputs/figures/report/supp_sawtooth_mech.png.
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

LAYER, N_PROBES = 9, 40
df = load_manifest("direction")
F = load_features("direction")[:, LAYER].astype(np.float64)
y = targets("direction", df)
th = np.round(df.theta_degrees.values, 4)
rows = np.random.default_rng(0).permutation(np.nonzero(np.isin(th, np.arange(0, 360, 45.0)))[0])
n_tr = int(0.8 * len(rows))
tr, te = rows[:n_tr], rows[n_tr:]
acc15 = lambda yhat, yt: float((circ_err(angle_deg(yhat), angle_deg(yt)) <= 15).mean() * 100)


def fit(Xtr, ytr, kind, k):
    """returns W [d,2], b [2] acting on the features given."""
    if kind == "adam":
        W, b = train_linear(to_dev(Xtr)[None], to_dev(ytr), [1e-3], [1e-4], 100, decoupled=False, seed=1000 + k)
        return W[0, 0].double().cpu().numpy(), b[0, 0].double().cpu().numpy()
    itr, iva = np.arange(len(Xtr))[: int(0.8 * len(Xtr))], np.arange(len(Xtr))[int(0.8 * len(Xtr)):]
    r = fit_ridge(Xtr[itr], ytr[itr], Xtr[iva], ytr[iva])
    return r.W_raw, r.b_raw


def orth(W):
    Q, _ = np.linalg.qr(W)
    return Q


def run(variant, kind):
    Xtr, Xte = F[tr].copy(), F[te].copy()
    out = []
    for k in range(N_PROBES):
        if variant == "zscore_mismatch":
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            W, b = fit((Xtr - mu) / sd, y[tr], kind, k)
            out.append(acc15(((Xte - mu) / sd) @ W + b, y[te]))
            Q = orth(W)                      # W lives in z-space; applied to raw space (the mismatch)
            Xtr, Xte = Xtr - (Xtr @ Q) @ Q.T, Xte - (Xte @ Q) @ Q.T
            continue
        W, b = fit(Xtr, y[tr], kind, k)
        out.append(acc15(Xte @ W + b, y[te]))
        if variant == "row_alternate":
            Q = orth(W[:, [k % 2]])
        else:
            Q = orth(W)
        if variant == "log_twice":
            out.append(acc15((Xte - (Xte @ Q) @ Q.T) @ W + b, y[te]))
        Xtr = Xtr - (Xtr @ Q) @ Q.T
        if variant != "test_unproj":
            Xte = Xte - (Xte @ Q) @ Q.T
    return out


def lag1(a):
    a = np.asarray(a, float)
    a = a - a.mean()
    return float((a[:-1] * a[1:]).sum() / max((a * a).sum(), 1e-12))


res = {"n_train": int(n_tr), "n_test": int(len(te)), "layer_index": LAYER}
variants = ["baseline", "log_twice", "test_unproj", "zscore_mismatch", "row_alternate"]
for v in variants:
    for kind in ("adam", "ridge"):
        a = run(v, kind)
        res[f"{v}/{kind}"] = a
        print(f"{v:16s} {kind:5s} lag1={lag1(a):+.2f}  odd mean={np.mean(a[0::2]):5.1f}  even mean={np.mean(a[1::2]):5.1f}  "
              f"first 16: {' '.join(f'{x:.0f}' for x in a[:16])}", flush=True)
(OUT / "results" / "sawtooth_mechanisms.json").write_text(json.dumps(res, indent=1))

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False})
fig, axes = plt.subplots(1, len(variants), figsize=(15, 3), sharey=True)
for a, v in zip(axes, variants):
    for kind, c in (("adam", "#1f77b4"), ("ridge", "#ff7f0e")):
        s = res[f"{v}/{kind}"]
        a.plot(range(1, len(s) + 1), s, color=c, lw=1.3)
    a.set_title(v); a.set_xlabel("entry"); a.set_ylim(0, 102)
axes[0].set_ylabel("accuracy within 15° (%)")
axes[-1].legend(handles=[plt.Line2D([], [], color="#1f77b4", label="Adam probe"), plt.Line2D([], [], color="#ff7f0e", label="ridge probe")],
                loc="center left", bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "supp_sawtooth_mech.png", dpi=150, bbox_inches="tight", pad_inches=0.1,
            bbox_extra_artists=[axes[-1].get_legend()] + fig.get_default_bbox_extra_artists())
