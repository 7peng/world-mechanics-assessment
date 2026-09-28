"""Speed and direction in the layer-12 representation, and what each steering method does.
Each clip = its offset from the speed curve, in the 2-D plane a direction probe reads. Colour = direction.
rings.png     4 speeds side by side; left naive steering, right shift along curve (0.5 -> 3.5 m/s)
rings_3d.png  all clips: x = position along the speed curve, (y, z) = direction plane -> a tube"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import coord, fit_manifold
from src.probes import fit_ridge, targets
from src.splits import LABEL

L, name = 12, "speed"
K = json.loads((OUT / "results" / "manifolds.json").read_text())[name]["K"]
CM = plt.get_cmap("twilight_shifted")
INK, SOFT = "#2b2b2b", "#f3f3f6"
plt.rcParams.update({"figure.facecolor": "white", "font.size": 10.5, "axes.titlesize": 12, "legend.frameon": False})
df = load_manifest(name); lab = df[LABEL[name]].values; th = df.theta_degrees.values
X = load_features(name)[:, L].astype(np.float64)
idx = split_idx(name); tr, va = idx["train"], idx["val"]
rows = np.r_[tr, va, idx["test"]]
man = fit_manifold(name, X[tr], lab[tr], 64, "lsq", K)
thp = fit_ridge(man.project(X[va]), targets("direction", df)[va], alpha=1.0)
D = np.linalg.qr(thp.W_raw)[0]
Z = man.project(X[rows]); u = coord(name, lab[rows]); v_all = lab[rows]; t_all = th[rows]
R = (Z - man.curve(u)) @ D
R /= R.std()
# orient so that θ = 0° points up-ish consistently (rotate plane so mean offset of θ≈90° clips is +y)
ang = np.arctan2(*(R[np.abs(((t_all - 90) + 180) % 360 - 180) < 20].mean(0)[::-1]))
rot = np.array([[np.cos(np.pi / 2 - ang), -np.sin(np.pi / 2 - ang)], [np.sin(np.pi / 2 - ang), np.cos(np.pi / 2 - ang)]])
R = R @ rot.T
col = lambda t: CM(np.asarray(t) / 360.0)

# ---------------- 2-D
SPEEDS = [0.5, 1.2, 2.2, 3.5]
GAP = 5.2
xs = {v: i * GAP for i, v in enumerate(SPEEDS)}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True, gridspec_kw={"wspace": 0.03})
src = np.nonzero(np.abs(v_all - SPEEDS[0]) <= 0.12 * SPEEDS[0])[0]
order = src[np.argsort(t_all[src])]
pick = order[np.linspace(0, len(order) - 1, 8).astype(int)]
for a, mode in zip(axes, ("naive", "shift along curve")):
    a.set_facecolor(SOFT)
    a.plot([xs[SPEEDS[0]], xs[SPEEDS[-1]]], [0, 0], color=INK, lw=1.2, zorder=1)
    for v in SPEEDS:
        m = np.abs(v_all - v) <= 0.12 * v
        a.scatter(xs[v] + R[m, 0], R[m, 1], c=col(t_all[m]), s=11, lw=0, alpha=0.9, zorder=2)
        a.scatter([xs[v]], [0], s=22, color=INK, zorder=3)
        a.text(xs[v], -2.45, f"{v} m/s", ha="center", fontsize=10, color=INK)
    for i in pick:
        x0, y0 = xs[SPEEDS[0]] + R[i, 0], R[i, 1]
        x1, y1 = (xs[SPEEDS[-1]], 0.0) if mode == "naive" else (xs[SPEEDS[-1]] + R[i, 0], R[i, 1])
        a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5",
                   color=col(t_all[i]), lw=1.6, alpha=0.95, shrinkA=3, shrinkB=2), zorder=4)
        a.scatter([x0], [y0], s=46, color=col(t_all[i]), edgecolor=INK, lw=0.9, zorder=5)
    a.set_title(mode, color=INK)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_xlim(xs[SPEEDS[0]] - 2.8, xs[SPEEDS[-1]] + 2.8); a.set_ylim(-2.75, 2.4)
axes[0].set_ylabel("direction plane", color=INK)
fig.text(0.5, 0.01, "position along the speed curve  →", ha="center", color=INK)
sm = plt.cm.ScalarMappable(cmap=CM, norm=plt.Normalize(0, 360))
cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.01, ticks=[0, 90, 180, 270, 360])
cb.set_label("direction θ (°)"); cb.outline.set_visible(False)
fig.savefig(OUT / "figures" / "report" / "rings.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# ---------------- 3-D tube (depth cues: far segments thinner, fainter, drawn first)
curve = man.curve(man.u_grid(400))
ax1 = curve[-1] - curve[0]; ax1 /= np.linalg.norm(ax1)
pos = (man.curve(u) - curve[0]) @ ax1
pos = pos / pos.max() * 10
bins = np.quantile(pos, np.linspace(0, 1, 9))
TB = np.arange(0, 360, 22.5)
ELEV, AZIM, ASPECT, LIM = 22, -38, (2.4, 1, 1), 2.2
xr = (pos.min(), pos.max())
view = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)), np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)), np.sin(np.radians(ELEV))])


def depth(P):
    P = np.atleast_2d(P)
    n = np.c_[(P[:, 0] - xr[0]) / (xr[1] - xr[0]) * ASPECT[0], (P[:, 1] + LIM) / (2 * LIM), (P[:, 2] + LIM) / (2 * LIM)]
    return n @ view


# reference depth range: the box corners
corners = np.array([[x, y, z] for x in xr for y in (-LIM, LIM) for z in (-LIM, LIM)])
dmin, dmax = depth(corners).min(), depth(corners).max()
near = lambda P: np.clip((depth(P).mean() - dmin) / (dmax - dmin), 0, 1)       # 0 far, 1 near

segments = []                                                                   # (nearness, xs, ys, zs, color, lw)
for lo_, hi_ in zip(bins[:-1], bins[1:]):
    m = (pos >= lo_) & (pos <= hi_)
    ring = []
    for tb in TB:
        mm = m & (np.abs(((t_all - tb) + 180) % 360 - 180) <= 11.25)
        if mm.sum() >= 2:
            ring.append((pos[m].mean(), *R[mm].mean(0), tb))
    ring = np.array(ring + ring[:1])
    for a0, a1 in zip(ring[:-1], ring[1:]):
        seg = np.stack([a0[:3], a1[:3]])
        segments.append((near(seg), seg, col(a0[3]), "ring"))
fig = plt.figure(figsize=(12, 5.2))
for k, mode in enumerate(("naive", "shift along curve")):
    a = fig.add_subplot(1, 2, k + 1, projection="3d", computed_zorder=False)
    segs = list(segments)
    ax_line = np.c_[np.linspace(*xr, 40), np.zeros(40), np.zeros(40)]
    for j in range(39):
        segs.append((near(ax_line[j:j + 2]), ax_line[j:j + 2], INK, "axis"))
    x_end = pos[np.abs(v_all - 3.5) <= 0.1].mean()
    ends = []
    for i in pick:
        ts = np.linspace(0, 1, 12)
        if mode == "naive":
            P = np.c_[pos[i] + ts * (x_end - pos[i]), R[i, 0] * (1 - ts), R[i, 1] * (1 - ts)]
        else:
            P = np.c_[pos[i] + ts * (x_end - pos[i]), np.full(len(ts), R[i, 0]), np.full(len(ts), R[i, 1])]
        for j in range(len(P) - 1):
            segs.append((near(P[j:j + 2]), P[j:j + 2], col(t_all[i]), "path"))
        ends.append((near(P[0]), P[0], col(t_all[i]), "o"))
        ends.append((near(P[-1]), P[-1], col(t_all[i]), "s"))
    # faint clip cloud, depth-faded
    dn = np.clip((depth(np.c_[pos, R]) - dmin) / (dmax - dmin), 0, 1)
    cc = col(t_all).copy(); cc[:, 3] = 0.05 + 0.25 * dn
    a.scatter(pos, R[:, 0], R[:, 1], c=cc, s=2, lw=0, depthshade=False, zorder=0)
    items = sorted(segs + ends, key=lambda z: z[0])
    for zi, (nr, P, c, kind) in enumerate(items):
        if kind in ("o", "s"):
            a.scatter(*P, s=18 + 30 * nr, color=c, marker=kind, edgecolor=INK, lw=0.8, alpha=0.4 + 0.6 * nr, depthshade=False, zorder=zi + 1)
            continue
        lw = {"ring": 0.8 + 2.4 * nr, "axis": 0.6 + 1.4 * nr, "path": 0.8 + 2.4 * nr}[kind]
        al = 0.2 + 0.8 * nr
        a.plot(*P.T, color=c, lw=lw, alpha=al, solid_capstyle="butt" if kind == "path" else "round", zorder=zi + 1)
    a.set_title(mode, color=INK)
    a.set_xlabel("along speed curve →", labelpad=-6); a.set_ylabel(""); a.set_zlabel("")
    a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
    for ax_ in (a.xaxis, a.yaxis, a.zaxis):
        ax_.pane.set_facecolor(SOFT); ax_.pane.set_edgecolor("white"); ax_._axinfo["grid"]["color"] = "white"
    a.view_init(elev=ELEV, azim=AZIM); a.set_box_aspect(ASPECT)
    a.set_xlim(*xr); a.set_ylim(-LIM, LIM); a.set_zlim(-LIM, LIM)
sm = plt.cm.ScalarMappable(cmap=CM, norm=plt.Normalize(0, 360))
cb = fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.02, ticks=[0, 90, 180, 270, 360]); cb.set_label("direction θ (°)"); cb.outline.set_visible(False)
fig.savefig(OUT / "figures" / "report" / "rings_3d.png", dpi=170, bbox_inches="tight", pad_inches=0.15)
