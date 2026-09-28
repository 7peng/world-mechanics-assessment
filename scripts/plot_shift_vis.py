"""(a) manifold_extrapolation.png: curve fit on the middle values only, clamped at its ends (before) vs extended
    linearly past the ends (after), with the withheld end centroids. Top-2 PCs of all centroids.
(b) shift_vs_replace.png: real low-speed test clips steered to a high target speed. x = position along the
    speed curve's main axis; y = the axis carrying direction (pooled sin θ probe weights in PCA space,
    orthogonalised against x). Naive replace collapses all clips to one point; shifting moves them together."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import BSpline

from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import coord, fit_manifold, uncoord
from src.probes import fit_ridge, targets
from src.splits import LABEL, load_splits

L = 12
KB = {n: json.loads((OUT / "results" / "manifolds.json").read_text())[n]["K"] for n in ("speed", "acceleration")}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.color": "#555", "ytick.color": "#555",
                     "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})


def design(u, lo, hi, K, extrap):
    t = np.r_[[lo] * 4, np.linspace(lo, hi, K + 2)[1:-1], [hi] * 4]
    u = np.atleast_1d(np.asarray(u, float))
    D = BSpline.design_matrix(np.clip(u, lo, hi), t, 3).toarray()
    if extrap:
        eps = 1e-4 * (hi - lo)
        for edge, sg in ((lo, 1), (hi, -1)):
            m = (u < lo) if edge == lo else (u > hi)
            if m.any():
                d0 = BSpline.design_matrix(np.array([edge]), t, 3).toarray()
                d1 = BSpline.design_matrix(np.array([edge + sg * eps]), t, 3).toarray()
                D[m] = d0 + (u[m] - edge)[:, None] * (d1 - d0) / (sg * eps)
    return D


# ---------------- (a) extrapolation
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
for a, name in zip(axes, ("speed", "acceleration")):
    df = load_manifest(name); lab = df[LABEL[name]].values
    X = load_features(name)[:, L].astype(np.float64)
    idx = split_idx(name, "value_heldout_ends")
    tr = idx["train"]
    man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", KB[name])
    Z = man.project(X[tr]); u_tr = coord(name, lab[tr]); lo, hi = u_tr.min(), u_tr.max()
    B = np.linalg.lstsq(design(u_tr, lo, hi, KB[name], True), Z, rcond=None)[0]
    # centroids of all values (train + withheld ends), projected with this PCA
    allv = np.unique(np.round(lab, 6))
    Za = man.project(X)
    C = np.stack([Za[np.round(lab, 6) == v].mean(0) for v in allv])
    held = np.isin(allv, np.round(load_splits()[name]["heldout_values_ends"], 6))
    V2 = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][:2].T
    pr = lambda P: (P - C.mean(0)) @ V2
    uall = np.linspace(coord(name, allv.min()), coord(name, allv.max()), 400)
    inside = (uall >= lo) & (uall <= hi)
    a.plot(*pr(design(uall, lo, hi, KB[name], True) @ B)[~inside & (uall < lo)].T, color="#e8702a", lw=2, ls=(0, (3, 2)))
    a.plot(*pr(design(uall, lo, hi, KB[name], True) @ B)[~inside & (uall > hi)].T, color="#e8702a", lw=2, ls=(0, (3, 2)))
    a.plot(*pr(design(uall[inside], lo, hi, KB[name], False) @ B).T, color="#222", lw=2)
    ends = pr(design(np.array([lo, hi]), lo, hi, KB[name], False) @ B)
    a.scatter(*ends.T, s=60, facecolor="white", edgecolor="#222", lw=1.5, zorder=4)
    sc = a.scatter(*pr(C[~held]).T, c=allv[~held], cmap="viridis", vmin=allv.min(), vmax=allv.max(), s=16, zorder=3)
    a.scatter(*pr(C[held]).T, c=allv[held], cmap="viridis", vmin=allv.min(), vmax=allv.max(), s=34, marker="D", edgecolor="#222", lw=0.8, zorder=3)
    a.set_title(name); a.set_xticks([]); a.set_yticks([]); a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02).set_label("m/s" if name == "speed" else "m/s²")
fig.legend(handles=[Line2D([], [], color="#222", lw=2, label="curve fit on the middle values"),
                    Line2D([], [], ls="none", marker="o", mfc="white", mec="#222", ms=8, label="naive: targets past the ends map here"),
                    Line2D([], [], color="#e8702a", lw=2, ls=(0, (3, 2)), label="fix: linear extension"),
                    Line2D([], [], ls="none", marker="D", mfc="#bbb", mec="#222", ms=6, label="centroids of withheld end values")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "manifold_extrapolation.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# ---------------- (b) shift vs replace
name = "speed"
df = load_manifest(name); lab = df[LABEL[name]].values
X = load_features(name)[:, L].astype(np.float64)
idx = split_idx(name)
tr, va, te = idx["train"], idx["val"], idx["test"]
man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", KB[name])
yth = targets("direction", df)
thp = fit_ridge(man.project(X[va]), yth[va], alpha=1.0)             # direction probe in PCA-64 coords
sp = fit_ridge(X[va], targets(name, df)[va], alpha=1.0)
us = man.u_grid(400); curve = man.curve(us)
ax1 = curve[-1] - curve[0]; ax1 /= np.linalg.norm(ax1)
wd = thp.W_raw[:, 0]; wd = wd - (wd @ ax1) * ax1; wd /= np.linalg.norm(wd)
P2 = np.stack([ax1, wd], 1)
rng = np.random.default_rng(3)
rows = te[lab[te] < 0.9]; rows = rng.choice(rows, 28, replace=False)
T = 3.0
Z0 = man.project(X[rows])
Zr = np.broadcast_to(man.point(T), Z0.shape)
u_hat = coord(name, np.clip(sp.predict(X[rows])[:, 0], lab.min(), lab.max()))
Zs = Z0 + man.point(T) - man.curve(u_hat)
cen = curve.mean(0)
p = lambda Zz: (Zz - cen) @ P2
th = df.theta_degrees.values[rows]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True, sharey=True)
for a, Zn, title in ((axes[0], Zr, "naive: replace with the curve point"), (axes[1], Zs, "fix: shift along the curve")):
    a.plot(*p(curve).T, color="#222", lw=2, zorder=2)
    a0, a1 = p(Z0), p(Zn)
    for (x0, y0), (x1, y1) in zip(a0, a1):
        a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.7, shrinkA=3, shrinkB=3), zorder=1)
    a.scatter(*a0.T, c=th, cmap="twilight", vmin=0, vmax=360, s=26, edgecolor="#333", lw=0.4, zorder=3)
    sc = a.scatter(*a1.T, c=th, cmap="twilight", vmin=0, vmax=360, s=40, marker="s", edgecolor="#333", lw=0.4, zorder=4)
    a.scatter(*p(man.point(T)).T, s=90, marker="*", color="#e8702a", edgecolor="#222", lw=0.6, zorder=5)
    a.set_title(title); a.set_xlabel("along the speed curve  →  faster"); a.set_xticks([]); a.set_yticks([])
axes[0].set_ylabel("direction-carrying axis")
fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02).set_label("clip's direction θ (°)")
fig.legend(handles=[Line2D([], [], color="#222", lw=2, label="speed curve"),
                    Line2D([], [], ls="none", marker="o", mfc="#ccc", mec="#333", label="slow clips before steering"),
                    Line2D([], [], ls="none", marker="s", mfc="#ccc", mec="#333", label="after steering to 3 m/s"),
                    Line2D([], [], ls="none", marker="*", mfc="#e8702a", mec="#222", ms=11, label="curve point for 3 m/s")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.45, -0.1))
fig.savefig(OUT / "figures" / "report" / "shift_vs_replace.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
