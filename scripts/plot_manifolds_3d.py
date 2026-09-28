"""3-D view of the layer-12 manifolds: per-value centroids, the paper's interpolating cubic, and the
least-squares curve used for steering, in the top-3 PCs of the centroid cloud. -> manifolds_3d.png"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.data import DATASETS, OUT

CM = {"direction": "twilight", "speed": "viridis", "acceleration": "viridis"}
UNIT = {"direction": "θ (°)", "speed": "m/s", "acceleration": "m/s²"}
VIEW = {"direction": (35, -60), "speed": (25, -70), "acceleration": (25, -70)}
plt.rcParams.update({"figure.facecolor": "white", "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False})
fig = plt.figure(figsize=(14, 4.6))
for j, name in enumerate(DATASETS):
    m = pickle.load(open(OUT / "manifolds" / f"{name}_L12_lsq.pkl", "rb"))
    mi = pickle.load(open(OUT / "manifolds" / f"{name}_L12_interp.pkl", "rb"))
    c0 = m.C.mean(0)
    B = np.linalg.svd(m.C - c0, full_matrices=False)[2][:3].T
    us = m.u_grid(600)
    a = fig.add_subplot(1, 3, j + 1, projection="3d")
    a.plot(*((mi.curve(us) - c0) @ B).T, color="#aaaaaa", lw=1)
    a.plot(*((m.curve(us) - c0) @ B).T, color="#222222", lw=2)
    sc = a.scatter(*((m.C - c0) @ B).T, c=m.values, cmap=CM[name], s=22, edgecolor="#333", lw=0.4, depthshade=False)
    a.view_init(*VIEW[name])
    a.set_title(name)
    a.set_xticklabels([]); a.set_yticklabels([]); a.set_zticklabels([])
    a.set_xlabel("PC1", labelpad=-10); a.set_ylabel("PC2", labelpad=-10); a.set_zlabel("PC3", labelpad=-10)
    for ax_ in (a.xaxis, a.yaxis, a.zaxis):
        ax_.pane.set_facecolor("#EAEAF2"); ax_.pane.set_edgecolor("white"); ax_._axinfo["grid"]["color"] = "white"
    cb = fig.colorbar(sc, ax=a, fraction=0.03, pad=0.02, shrink=0.7)
    cb.set_label(UNIT[name], fontsize=8.5)
fig.legend(handles=[Line2D([], [], color="#222222", lw=2, label="least-squares curve (used for steering)"),
                    Line2D([], [], color="#aaaaaa", lw=1, label="interpolating cubic through centroids (paper)"),
                    Line2D([], [], ls="none", marker="o", color="#777", label="centroid of one label value")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "manifolds_3d.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
