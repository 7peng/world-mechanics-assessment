"""f5_manifold_steering.png from outputs/results/steer_manifold_acc.json (same-layer read-out, layer 12)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from src.data import OUT

VARS = ["direction", "speed", "acceleration"]
BLUE, ORANGE, GRAY, DARK = "#1f77b4", "#ff7f0e", "#9a9a9a", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.color": "#555", "ytick.color": "#555",
                     "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})
res = json.loads((OUT / "results" / "steer_manifold_acc.json").read_text())
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True, gridspec_kw={"width_ratios": [3, 3, 3, 2.4], "wspace": 0.1})
w = 0.34
x = np.arange(3)
for a, n in zip(axes[:3], VARS):
    for i, (m, col) in enumerate((("subspace", BLUE), ("spline", ORANGE))):
        a.bar(x + (i - 0.5) * w, [res[n][c][m]["target_within"] for c in ("seen", "heldout", "ends")], w * 0.92, color=col)
    a.axhline(res["chance"][n], color=GRAY, lw=1, ls=(0, (2, 2)))
    a.set_xticks(x); a.set_xticklabels(["seen", "unseen", "outside\nrange"])
    a.set_title(f"steering {n}")
    a.grid(axis="x", visible=False)
a = axes[3]
x2 = np.arange(2)
for i, (m, col) in enumerate((("subspace", BLUE), ("spline", ORANGE))):
    a.bar(x2 + (i - 0.5) * w, [res[n]["seen"][m]["theta_kept_within"] for n in ("speed", "acceleration")], w * 0.92, color=col)
for xi, n in zip(x2, ("speed", "acceleration")):
    a.hlines(res[n]["seen"]["theta_unsteered_within"], xi - 0.4, xi + 0.4, color=DARK, lw=1.3)
a.axhline(res["chance"]["direction"], color=GRAY, lw=1, ls=(0, (2, 2)))
a.set_xticks(x2); a.set_xticklabels(["steering\nspeed", "steering\nacceleration"])
a.set_title("direction left intact")
a.grid(axis="x", visible=False)
axes[0].set_ylim(0, 105); axes[0].set_yticks([0, 25, 50, 75, 100])
axes[0].set_ylabel("% of clips")
fig.legend(handles=[Patch(color=BLUE, label="subspace steering"), Patch(color=ORANGE, label="spline steering"),
                    Line2D([], [], color=DARK, lw=1.3, label="before steering"), Line2D([], [], color=GRAY, ls=(0, (2, 2)), label="chance")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.14))
fig.savefig(OUT / "figures" / "report" / "f5_manifold_steering.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
