"""f5b_spline_fixed.png: subspace steering vs spline (paper's replace) vs spline fixed (move along the
curve + linear extrapolation). Same-layer read-out, layer 12. Averaged over speed and acceleration;
'inside range' averages seen and unseen target values."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from src.data import OUT

BLUE, LIGHT, ORANGE, DARK = "#3a78b5", "#f2c9a0", "#e8702a", "#444444"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.color": "#333", "ytick.color": "#555",
                     "font.size": 11, "axes.titlesize": 12, "legend.frameon": False})
A = json.loads((OUT / "results" / "steer_manifold_acc.json").read_text())
F = json.loads((OUT / "results" / "spline_fixes.json").read_text())
M = [("subspace", BLUE, "subspace steering"), ("spline", LIGHT, "spline, naive (replace)"), ("spline_disp", ORANGE, "spline, shift along curve")]
GROUPS = {"inside": [("seen", "seen"), ("heldout", "unseen")], "outside": [("ends", "outside")]}


def val(m, key, group):
    v = []
    for n in ("speed", "acceleration"):
        for ca, cf in GROUPS[group]:
            if m == "subspace":
                v.append(A[n][ca]["subspace"]["target_within" if key == "on" else "theta_kept_within"])
            else:
                v.append(F[n][cf][m]["on_target_pct" if key == "on" else "direction_kept_pct"])
    return float(np.mean(v))


fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.06})
w = 0.25
x = np.arange(2) * 0.95
for a, key, title in ((axes[0], "on", "successes"), (axes[1], "kept", "direction retention")):
    for i, (m, col, _) in enumerate(M):
        vals = [val(m, key, g) for g in ("inside", "outside")]
        bars = a.bar(x + (i - 1) * w, vals, w * 0.88, color=col, zorder=3)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=DARK)
    a.set_xticks(x); a.set_xticklabels(["interpolation", "extrapolation"])
    a.set_title(title, pad=10); a.grid(axis="x", visible=False); a.tick_params(axis="x", labelsize=11, pad=6)
axes[0].set_ylim(0, 112); axes[0].set_yticks([0, 25, 50, 75, 100]); axes[0].set_yticklabels(["0", "25", "50", "75", "100%"]); axes[0].set_ylabel("steered clips")
fig.legend(handles=[Patch(color=c, label=l) for _, c, l in M], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09), handlelength=1.2, columnspacing=1.8)
fig.savefig(OUT / "figures" / "report" / "f5b_spline_fixed.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
