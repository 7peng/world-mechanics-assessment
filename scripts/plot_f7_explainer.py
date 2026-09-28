"""Schematic of the token-swap experiment behind f7 (f7_explainer.png).
Tokens drawn as a grid: tint = token average, dark trail = token pattern (tokens minus their average).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data import OUT

INK, MUTED = "#333333", "#777777"
TINT = {"orig": "#b9cde5", "donor": "#fbd3a8"}
TRAIL = {"orig": [(3, c) for c in range(1, 5)], "donor": [(r, 3) for r in range(5, 1, -1)]}
N, S = 6, 1.0
plt.rcParams.update({"font.size": 10.5, "figure.facecolor": "white"})


def grid(ax, x, y, tint, trail):
    c = S / N
    for i in range(N):
        for j in range(N):
            ax.add_patch(Rectangle((x + j * c, y + (N - 1 - i) * c), c, c, fc=tint, ec="white", lw=1.2))
    for k, (i, j) in enumerate(trail):
        ax.add_patch(Rectangle((x + j * c, y + (N - 1 - i) * c), c, c, fc=INK, alpha=0.3 + 0.7 * k / (len(trail) - 1), ec="white", lw=1.2))


fig, ax = plt.subplots(figsize=(10, 2.6))
ax.set_xlim(0, 10); ax.set_ylim(-0.75, 1.6); ax.axis("off")
panels = [(0.2, "original", "orig", "orig", None),
          (1.7, "donor", "donor", "donor", None),
          (4.0, "all donor", "donor", "donor", "follows donor\nat every layer"),
          (6.0, "donor pattern", "orig", "donor", "follows donor\nat layers 0–8"),
          (8.0, "donor average", "donor", "orig", "never follows\ndonor (layers 0–20)")]
for x, lab, tint, trail, res in panels:
    grid(ax, x, 0, TINT[tint], TRAIL[trail])
    ax.text(x + S / 2, S + 0.12, lab, ha="center", va="bottom", color=INK)
    if res:
        ax.text(x + S / 2, -0.12, res, ha="center", va="top", color=INK, fontsize=9.5)
ax.plot([3.3, 3.3], [-0.6, 1.45], color="#cccccc", lw=1)
ax.text(1.45, -0.2, "tint = token average\ntrail = token pattern", ha="center", va="top", color=MUTED, fontsize=9)
fig.savefig(OUT / "figures" / "report" / "f7_explainer.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
print("ok")
