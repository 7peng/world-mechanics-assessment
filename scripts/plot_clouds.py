"""clouds.png: the speed representation as one cloud sliding along a curve.
(a) train clips in 2-D (x = along the speed curve, y = largest spread across the curve), coloured by speed,
    with a 2-sigma ellipse for 5 speed bins and the curve through their centres.
(b) the same ellipses moved to a common centre: same shape at every speed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import coord, fit_manifold
from src.splits import LABEL

L, name = 12, "speed"
K = json.loads((OUT / "results" / "manifolds.json").read_text())[name]["K"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
                     "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})
df = load_manifest(name); lab = df[LABEL[name]].values
X = load_features(name)[:, L].astype(np.float64)
tr = split_idx(name)["train"]
man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", K)
Z = man.project(X[tr]); u = coord(name, lab[tr])
curve = man.curve(man.u_grid(400))
ax1 = curve[-1] - curve[0]; ax1 /= np.linalg.norm(ax1)
R = Z - man.curve(u)
R2 = R - np.outer(R @ ax1, ax1)
ax2 = np.linalg.svd(R2, full_matrices=False)[2][0]
P = np.stack([ax1, ax2], 1)
c0 = curve.mean(0)
pz = (Z - c0) @ P
edges = np.quantile(lab[tr], np.linspace(0, 1, 6))
shapes = []
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(lab.min(), lab.max())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
a, b = axes
pres = ((man.curve(u) + R) - c0) @ P
a.scatter(*pz.T, c=lab[tr], cmap=cmap, norm=norm, s=6, alpha=0.3, lw=0)
a.plot(*((curve - c0) @ P).T, color="#222", lw=2)
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    m = (lab[tr] >= lo_) & (lab[tr] <= hi_)
    # cloud shape from each clip's offset from the curve at its own speed (removes within-bin speed spread)
    uc = np.median(u[m])
    pts = ((man.curve(np.full(m.sum(), uc)) + R[m]) - c0) @ P
    mu = pts.mean(0); C = np.cov(pts.T)
    shapes.append(2 * np.sqrt(np.linalg.eigvalsh(C)))
    ev, evec = np.linalg.eigh(C); ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    col = cmap(norm(lab[tr][m].mean()))
    for aa, cen in ((a, mu), (b, np.zeros(2))):
        aa.add_patch(Ellipse(cen, 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=ang, fill=False, ec=col, lw=2))
    a.scatter(*mu, color=col, s=40, edgecolor="#222", zorder=4)
a.set_title("clips (dots) and the spread around the curve at 5 speeds (2σ)")
a.set_xlabel("along the speed curve  →  faster"); a.set_ylabel("across the curve"); a.set_xticks([]); a.set_yticks([])
lim = max(max(sh) for sh in shapes) * 1.15
b.set_xlim(-lim, lim); b.set_ylim(-lim, lim); b.set_aspect("equal")
b.set_title("the same clouds, re-centred"); b.set_xticks([]); b.set_yticks([])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02).set_label("speed (m/s)")
fig.savefig(OUT / "figures" / "report" / "clouds.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
