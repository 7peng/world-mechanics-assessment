"""Schematic of the token-swap experiment behind f7 (f7_explainer.png).
Tokens at layer L as a point cloud: the token average is where the cloud sits (×), the token pattern is
its shape (tokens minus their average). Dot colour = whose pattern, × colour = whose average.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import OUT

BLUE, ORANGE, INK = "#1f77b4", "#ff7f0e", "#333333"
plt.rcParams.update({"font.size": 10.5, "figure.facecolor": "white", "axes.facecolor": "#EAEAF2"})

rng = np.random.default_rng(3)
t = rng.uniform(-1, 1, 40)
shape = {"orig": np.c_[t, 0.25 * np.sin(3 * t)] + 0.05 * rng.standard_normal((40, 2)),            # horizontal wave
         "donor": np.c_[0.3 * np.cos(np.pi * t), 0.8 * t] + 0.05 * rng.standard_normal((40, 2))}   # vertical arc
shape = {k: v - v.mean(0) for k, v in shape.items()}
center = {"orig": np.array([-0.9, -0.5]), "donor": np.array([0.9, 0.6])}
col = {"orig": BLUE, "donor": ORANGE}

panels = [("original clip", "orig", "orig"), ("donor clip", "donor", "donor"),
          ("donor pattern swapped in", "donor", "orig"), ("donor average swapped in", "orig", "donor")]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for a, (title, pat, avg) in zip(axes, panels):
    for k in ("orig", "donor"):                                            # faint reference positions of both averages
        a.scatter(*center[k], marker="x", s=40, color=col[k], alpha=0.25, lw=1.5)
    P = shape[pat] + center[avg]
    a.scatter(P[:, 0], P[:, 1], s=14, color=col[pat], lw=0)
    a.scatter(*center[avg], marker="x", s=90, color=col[avg], lw=2.5)
    a.set_title(title); a.set_xlim(-2.2, 2.2); a.set_ylim(-1.8, 1.8); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for sp in a.spines.values():
        sp.set_visible(False)
fig.text(0.5, -0.02, "dots = tokens (their shape = token pattern)      × = token average (what pooled probes read and pooled steering moves)",
         ha="center", va="top", color="#555", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f7_explainer.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
print("ok")
