"""f5b_spline_fixed.png: subspace steering vs spline (paper's replace) vs spline fixed
(move along the curve + linear extrapolation). Same-layer read-out, layer 12."""
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

BLUE, ORANGE, RED, GRAY, DARK = "#1f77b4", "#ffbb78", "#ff7f0e", "#9a9a9a", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.color": "#555", "ytick.color": "#555",
                     "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})
A = json.loads((OUT / "results" / "steer_manifold_acc.json").read_text())
F = json.loads((OUT / "results" / "spline_fixes.json").read_text())
CONDS = [("seen", "seen", "seen"), ("heldout", "unseen", "unseen"), ("ends", "outside", "outside\nrange")]
M = [("subspace", BLUE, "subspace steering"), ("spline", ORANGE, "spline, replace (paper)"), ("spline_disp", RED, "spline, move along curve (fixed)")]


def val(n, ca, cf, m, key):
    if m == "subspace":
        return A[n][ca]["subspace"]["target_within" if key == "on" else "theta_kept_within"]
    return F[n][cf][m]["on_target_pct" if key == "on" else "direction_kept_pct"]


fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True, gridspec_kw={"wspace": 0.1})
w = 0.27
x = np.arange(3)
for a, (n, key, title) in zip(axes, [("speed", "on", "speed: on target"), ("acceleration", "on", "acceleration: on target"),
                                     ("speed", "kept", "speed: direction intact"), ("acceleration", "kept", "acceleration: direction intact")]):
    for i, (m, col, _) in enumerate(M):
        a.bar(x + (i - 1) * w, [val(n, ca, cf, m, key) for ca, cf, _ in CONDS], w * 0.92, color=col)
    if key == "kept":
        a.axhline(A[n]["seen"]["theta_unsteered_within"], color=DARK, lw=1.3)
        a.axhline(A["chance"]["direction"], color=GRAY, lw=1, ls=(0, (2, 2)))
    else:
        a.axhline(A["chance"][n], color=GRAY, lw=1, ls=(0, (2, 2)))
    a.set_xticks(x); a.set_xticklabels([c for _, _, c in CONDS]); a.set_title(title); a.grid(axis="x", visible=False)
axes[0].set_ylim(0, 105); axes[0].set_yticks([0, 25, 50, 75, 100]); axes[0].set_ylabel("% of clips")
fig.legend(handles=[Patch(color=c, label=l) for _, c, l in M] + [Line2D([], [], color=DARK, lw=1.3, label="before steering"),
                                                                Line2D([], [], color=GRAY, ls=(0, (2, 2)), label="chance")],
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.14))
fig.savefig(OUT / "figures" / "report" / "f5b_spline_fixed.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
