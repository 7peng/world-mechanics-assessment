"""Part 2 figure, f5_manifold_steering.png. Two panels, two methods.
Left:  steering speed. x = how far the target is missed (relative to the unsteered probe's own error);
       y = how much direction is disturbed. Ideal is the bottom-left corner.
Right: steering speed to every label value, spline fit on the middle values only: error per target.
       Shaded region = targets outside the fitted range.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.data import OUT

BLUE, ORANGE, DARK = "#1f77b4", "#ff7f0e", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False})
S = json.loads((OUT / "results" / "steer_manifold.json").read_text())
E = json.loads((OUT / "results" / "steer_ends_by_target.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

# left: target error vs side effect, speed and acceleration (seen + held-out conditions), one marker per case
a = axes[0]
for m, col, mk in (("clamp", BLUE, "o"), ("spline", ORANGE, "s")):
    for n in ("speed", "acceleration"):
        for cond in ("seen", "heldout"):
            r = S[n][cond]
            a.scatter(r["methods"][m]["to_target"] / r["probe_floor"], r["methods"][m]["theta_drift"], color=col, marker=mk, s=45, zorder=3)
a.axhline(S["speed"]["seen"]["theta_floor"], color=DARK, ls=(0, (2, 2)), lw=1.1)
a.text(0.02, S["speed"]["seen"]["theta_floor"] + 2, "direction error before steering", fontsize=8, color=DARK)
a.axvline(1, color=DARK, ls=(0, (2, 2)), lw=1.1)
a.text(1.02, 50, "probe's\nown error", fontsize=8, color=DARK)
a.axhline(90, color="#999", lw=0.8); a.text(0.02, 91, "chance", fontsize=8, color="#777")
a.set_xlim(0, 1.5); a.set_ylim(0, 100)
a.set_xlabel("error to target (speed or acceleration), relative to the probe's own error")
a.set_ylabel("direction error after steering (°)")
a.set_title("both methods hit the target; only the spline erases direction")

# right: error per target value, spline fit on the middle of the range only
a = axes[1]
rows = E["speed"]
t = np.array([r[0] for r in rows]); e1 = np.array([r[1] for r in rows]); e2 = np.array([r[2] for r in rows]); held = np.array([r[3] for r in rows])
lo, hi = t[~held].min(), t[~held].max()
a.axvspan(t.min() - 0.05, lo, color="#d9d9d9", zorder=0); a.axvspan(hi, t.max() + 0.05, color="#d9d9d9", zorder=0)
a.plot(t, e1, color=BLUE, lw=1.6); a.plot(t, e2, color=ORANGE, lw=1.6)
a.axhline(S["speed"]["ends"]["probe_floor"], color=DARK, ls=(0, (2, 2)), lw=1.1)
a.text(1.2, S["speed"]["ends"]["probe_floor"] + 0.01, "probe's own error", fontsize=8, color=DARK)
a.text((t.min() + lo) / 2, 0.56, "not in\nspline fit", ha="center", fontsize=8, color="#555")
a.text((hi + t.max()) / 2, 0.56, "not in\nspline fit", ha="center", fontsize=8, color="#555")
a.set_xlim(t.min() - 0.05, t.max() + 0.05); a.set_ylim(0, 0.6)
a.set_xlabel("target speed (m/s)"); a.set_ylabel("error to target (m/s)")
a.set_title("spline fit on the middle of the range: it fails outside it")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=5, label="multi-probe subspace steering (Part 1)"),
                    Line2D([], [], color=ORANGE, lw=5, label="spline steering (Part 2)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f5_manifold_steering.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
