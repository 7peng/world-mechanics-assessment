"""Part 2 figure, f5_manifold_steering.png. Two methods (multi-probe subspace steering vs spline steering).
Panels 1-3: error to target in real units per variable, by condition (seen / held-out values / range ends).
Panel 4:    direction read-out error while steering speed or acceleration (off-target drift).
Panel 5:    how far steering-path waypoints sit from the data manifold, relative to unsteered clips.
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

VARS = ["direction", "speed", "acceleration"]
UNIT = {"direction": "error to target (°)", "speed": "error to target (m/s)", "acceleration": "error to target (m/s²)"}
BLUE, ORANGE, DARK = "#1f77b4", "#ff7f0e", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False})
S = json.loads((OUT / "results" / "steer_manifold.json").read_text())
P = json.loads((OUT / "results" / "steer_manifold_paths.json").read_text())
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 0.9, 0.9]})
w = 0.36
x = np.arange(3)
for a, n in zip(axes[:3], VARS):
    for i, (m, col) in enumerate((("clamp", BLUE), ("spline", ORANGE))):
        a.bar(x + (i - 0.5) * w, [S[n][c]["methods"][m]["to_target"] for c in ("seen", "heldout", "ends")], w, color=col)
    for xi, c in zip(x, ("seen", "heldout", "ends")):
        a.hlines(S[n][c]["probe_floor"], xi - 0.45, xi + 0.45, color=DARK, ls=(0, (2, 2)), lw=1.2)
    a.set_xticks(x); a.set_xticklabels(["seen\nvalues", "held-out\nvalues", "range\nends"]); a.set_ylabel(UNIT[n]); a.set_title(n)
    a.set_ylim(0, {"direction": 8, "speed": 0.5, "acceleration": 1.2}[n])
axes[0].text(2.45, S["direction"]["ends"]["probe_floor"] + 0.15, "unsteered probe error", fontsize=8, color=DARK, ha="right", va="bottom")

a = axes[3]
x2 = np.arange(2)
for i, (m, col) in enumerate((("clamp", BLUE), ("spline", ORANGE))):
    a.bar(x2 + (i - 0.5) * w, [S[n]["seen"]["methods"][m]["theta_drift"] for n in ("speed", "acceleration")], w, color=col)
for xi, n in zip(x2, ("speed", "acceleration")):
    a.hlines(S[n]["seen"]["theta_floor"], xi - 0.45, xi + 0.45, color=DARK, ls=(0, (2, 2)), lw=1.2)
a.axhline(90, color="#999", lw=0.8); a.text(1.45, 91, "chance", fontsize=8, color="#777", ha="right", va="bottom")
a.set_xticks(x2); a.set_xticklabels(["while steering\nspeed", "while steering\nacceleration"]); a.set_ylim(0, 100)
a.set_ylabel("direction error (°)"); a.set_title("side effect on direction")

a = axes[4]
for i, (m, col) in enumerate((("clamp", BLUE), ("curve", ORANGE))):
    vals = [100 * P[n]["paths"][m]["off"] / P[n]["paths"]["unsteered_off_manifold"] for n in VARS]
    a.bar(x + (i - 0.5) * w, vals, w, color=col)
    if m == "curve":
        for xi, v in zip(x, vals):
            a.text(xi + 0.5 * w, 2, f"{v:.2f}%", ha="center", va="bottom", fontsize=8, color=ORANGE)
a.axhline(100, color=DARK, ls=(0, (2, 2)), lw=1.2); a.text(2.45, 101, "unsteered clips", fontsize=8, color=DARK, ha="right", va="bottom")
a.set_xticks(x); a.set_xticklabels(VARS); a.set_ylim(0, 120)
a.set_ylabel("distance from manifold\n(% of unsteered clips)"); a.set_title("do steered activations lie on the manifold?")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=7, label="multi-probe subspace steering (Part 1)"),
                    Line2D([], [], color=ORANGE, lw=7, label="spline steering (Part 2)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f5_manifold_steering.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
