"""Manifold visualisation: (top) 3-D view in the top-3 PCs of the centroid cloud with test clips as
faint points; (bottom) each of the top-3 PCA coordinates of the *activation* PCA as a function of the
label, clips + curve. Output: outputs/figures/report/f4_manifolds.png"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import coord
from src.splits import LABEL

L = 12
CM = {"direction": "twilight", "speed": "viridis", "acceleration": "viridis"}
UNIT = {"direction": "θ (°)", "speed": "speed (m/s)", "acceleration": "acceleration (m/s²)"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False})
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
for j, name in enumerate(DATASETS):
    m = pickle.load(open(OUT / "manifolds" / f"{name}_L{L}_lsq.pkl", "rb"))
    df = load_manifest(name)
    idx = split_idx(name)
    lab = df[LABEL[name]].values
    Zt = m.project(load_features(name)[idx["test"], L].astype(float))
    pc = m.C - m.C.mean(0)
    _, _, V = np.linalg.svd(pc, full_matrices=False)
    B = V[:3].T
    us = m.u_grid(400)
    curve = (m.curve(us) - m.C.mean(0)) @ B
    a = fig.add_subplot(gs[0, j], projection="3d")
    a.set_facecolor("white")
    zt = (Zt - m.C.mean(0)) @ B
    a.scatter(*zt.T, c=lab[idx["test"]], cmap=CM[name], s=6, alpha=0.5, lw=0)
    a.plot(*curve.T, color="#222", lw=1.6)
    a.scatter(*(pc @ B).T, c=m.values, cmap=CM[name], s=18, edgecolor="#333", lw=0.4, zorder=5)
    a.set_title(name); a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
    a.set_xlabel("PC1", labelpad=-8); a.set_ylabel("PC2", labelpad=-8); a.set_zlabel("PC3", labelpad=-8)
    a.view_init(elev={"direction": 55, "speed": 30, "acceleration": 30}[name], azim={"direction": -60, "speed": -70, "acceleration": -70}[name])
    lim = np.abs(zt).max() * 0.85
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
    a.xaxis.pane.set_facecolor("#EAEAF2"); a.yaxis.pane.set_facecolor("#EAEAF2"); a.zaxis.pane.set_facecolor("#EAEAF2")
    for ax_ in (a.xaxis, a.yaxis, a.zaxis):
        ax_.pane.set_edgecolor("white"); ax_._axinfo["grid"]["color"] = "white"
    # bottom: activation-PCA coordinates 1..3 vs label
    b = fig.add_subplot(gs[1, j])
    x = lab[idx["test"]]
    xs = np.degrees(us) if name == "direction" else (np.exp(us) if name == "speed" else us)
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        b.scatter(x, Zt[:, k], s=3, color=cols[k], alpha=0.18, lw=0)
        b.plot(xs, m.curve(us)[:, k], color=cols[k], lw=1.6, label=f"activation PC{k + 1}")
    b.set_xlabel(UNIT[name])
    if name == "speed":
        b.set_xscale("log"); b.set_xticks([0.25, 0.5, 1, 2, 4]); b.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); b.minorticks_off()
    if j == 0:
        b.set_ylabel("PCA coordinate")
    if j == 2:
        b.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f4_manifolds.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
