"""rings.png: how speed and direction share the layer-12 representation, and what each steering method does.
For 4 speeds, each clip's offset from the speed curve, shown in the 2-D plane that encodes direction
(the two direction-probe axes), placed at that speed's position along the curve. Colour = clip's direction.
Left: naive replace sends slow clips to the centre of the fast ring (direction lost).
Right: shift along the curve moves them to the same place on the fast ring (direction kept)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import coord, fit_manifold
from src.probes import fit_ridge, targets
from src.splits import LABEL

L, name = 12, "speed"
K = json.loads((OUT / "results" / "manifolds.json").read_text())[name]["K"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 10,
                     "axes.titlesize": 11.5, "legend.frameon": False})
df = load_manifest(name); lab = df[LABEL[name]].values; th = df.theta_degrees.values
X = load_features(name)[:, L].astype(np.float64)
idx = split_idx(name); tr, va = idx["train"], idx["val"]
rows = np.r_[tr, va, idx["test"]]
man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", K)
thp = fit_ridge(man.project(X[va]), targets("direction", df)[va], alpha=1.0)
D = np.linalg.qr(thp.W_raw)[0]                                   # 64 x 2 plane that encodes direction
Z = man.project(X[rows]); u = coord(name, lab[rows])
R = (Z - man.curve(u)) @ D                                       # offset from the curve, in the direction plane
R /= R.std()
SPEEDS = [0.5, 1.2, 2.2, 3.5]
GAP = 7.0
xs = {v: i * GAP for i, v in enumerate(SPEEDS)}
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True, gridspec_kw={"wspace": 0.04})
rng = np.random.default_rng(1)
for a, mode in zip(axes, ("naive", "shift along curve")):
    a.plot([xs[SPEEDS[0]], xs[SPEEDS[-1]]], [0, 0], color="#222", lw=1.5, zorder=1)
    for v in SPEEDS:
        m = np.abs(lab[rows] - v) <= 0.12 * v
        a.scatter(xs[v] + R[m, 0], R[m, 1], c=th[rows][m], cmap="twilight", vmin=0, vmax=360, s=14, lw=0, alpha=0.85, zorder=2)
        a.scatter([xs[v]], [0], s=30, color="#222", zorder=3)
        a.text(xs[v], -2.55, f"{v} m/s", ha="center", fontsize=10)
    # steer 10 slow clips to the fastest speed
    src = np.nonzero(np.abs(lab[rows] - SPEEDS[0]) <= 0.12 * SPEEDS[0])[0]
    pick = src[np.argsort(th[rows][src])][:: max(1, len(src) // 10)][:10]
    for i in pick:
        x0, y0 = xs[SPEEDS[0]] + R[i, 0], R[i, 1]
        x1, y1 = (xs[SPEEDS[-1]], 0.0) if mode == "naive" else (xs[SPEEDS[-1]] + R[i, 0], R[i, 1])
        a.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle="-|>", color=plt.get_cmap("twilight")(th[rows][i] / 360), lw=1.4,
                                   connectionstyle="arc3,rad=-0.15", shrinkA=2, shrinkB=2), zorder=4)
        a.scatter([x0], [y0], s=36, color=plt.get_cmap("twilight")(th[rows][i] / 360), edgecolor="#222", lw=0.8, zorder=5)
    a.set_title(mode); a.set_xticks([]); a.set_yticks([])
    a.set_xlim(xs[SPEEDS[0]] - 3.8, xs[SPEEDS[-1]] + 3.8); a.set_ylim(-2.8, 2.5)
axes[0].set_ylabel("direction plane")
sm = plt.cm.ScalarMappable(cmap="twilight", norm=plt.Normalize(0, 360))
fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01).set_label("clip's direction θ (°)")
fig.legend(handles=[Line2D([], [], color="#222", lw=1.5, marker="o", ms=5, label="speed curve (one point per speed)"),
                    Line2D([], [], ls="none", marker="o", mfc="#bbb", mec="#bbb", ms=5, label="clips: offset from the curve in the direction plane"),
                    Line2D([], [], color="#777", lw=1.4, label="steering 0.5 → 3.5 m/s")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.45, -0.07))
fig.savefig(OUT / "figures" / "report" / "rings.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
