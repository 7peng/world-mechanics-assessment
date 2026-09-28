"""Part 2 figure: f5_manifold_steering.png.
Row 1: error to target by condition (seen / held-out values / range ends) for clamp, spline replace,
       interpolating spline, chord; probe floor as a line.
Row 2: off-target θ drift when steering speed / acceleration (floor, clamp, spline replace, spline span-only);
       decoded value along a source->target path for curve / chord / clamp against the ideal (one pair per variable).
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
UNIT = {"direction": "circular MAE (°)", "speed": "MAE (m/s)", "acceleration": "MAE (m/s²)"}
BLUE, ORANGE, GREEN, RED, GRAY, DARK = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False,
                     "lines.linewidth": 1.5})
S = json.loads((OUT / "results" / "steer_manifold.json").read_text())
P = json.loads((OUT / "results" / "steer_manifold_paths.json").read_text())
METH = [("clamp", BLUE, "subspace clamp (Part 1)"), ("spline", ORANGE, "spline, replace PCA-64"),
        ("spline_interp", RED, "interpolating spline (paper)"), ("chord", GREEN, "chord to nearest centroid")]
CONDS = [("seen", "seen"), ("heldout", "held-out\nvalues"), ("ends", "range\nends")]

fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
for j, n in enumerate(VARS):
    a = axes[0, j]
    x = np.arange(len(CONDS))
    w = 0.19
    for i, (m, c, lab) in enumerate(METH):
        vals = [S[n][cond]["methods"][m]["to_target"] for cond, _ in CONDS]
        a.bar(x + (i - 1.5) * w, vals, w, color=c)
    floors = [S[n][cond]["probe_floor"] for cond, _ in CONDS]
    for xi, f in zip(x, floors):
        a.hlines(f, xi - 0.42, xi + 0.42, color=DARK, ls=(0, (2, 2)), lw=1.2)
    a.set_xticks(x); a.set_xticklabels([c for _, c in CONDS]); a.set_title(n); a.set_ylabel(UNIT[n] + " to target")
    top = {"direction": 20, "speed": 0.6, "acceleration": 2.0}[n]
    a.set_ylim(0, top)
    for i, (m, c, lab) in enumerate(METH):          # clipped bars get their value printed
        for k, (cond, _) in enumerate(CONDS):
            v = S[n][cond]["methods"][m]["to_target"]
            if v > top:
                a.text(x[k] + (i - 1.5) * w, top * 0.97, f"{v:.1f}", ha="center", va="top", fontsize=7, color="white", rotation=90)
axes[0, 2].legend(handles=[Line2D([], [], color=c, lw=6, label=l) for _, c, l in METH] + [Line2D([], [], color=DARK, ls=(0, (2, 2)), label="unsteered probe error")],
                  loc="center left", bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

# off-target drift
a = axes[1, 0]
x = np.arange(2)
items = [("theta_floor", DARK, "unsteered"), ("clamp", BLUE, "clamp"), ("spline", ORANGE, "spline, replace PCA-64"), ("spline_span", "#c49c00", "spline, curve span only")]
w = 0.2
for i, (m, c, lab) in enumerate(items):
    vals = [S[n]["seen"]["theta_floor"] if m == "theta_floor" else S[n]["seen"]["methods"][m]["theta_drift"] for n in ("speed", "acceleration")]
    a.bar(x + (i - 1.5) * w, vals, w, color=c)
a.set_xticks(x); a.set_xticklabels(["steering speed", "steering acceleration"]); a.set_ylabel("θ read-out error (°)")
a.set_title("off-target drift"); a.set_ylim(0, 100)
a.legend(handles=[Line2D([], [], color=c, lw=6, label=l) for _, c, l in items], loc="upper left", fontsize=8)

# paths
pairs = {"direction": "0->90", "speed": "0.5->3.5", "acceleration": "1.0->9.0"}
for j, n in enumerate(["speed", "direction"]):
    a = axes[1, j + 1]
    k = pairs[n]
    ideal = P[n]["paths"]["ideal_example"][k]
    ts = np.linspace(0, 1, len(ideal))
    a.plot(ts, ideal, color=DARK, ls=(0, (2, 2)), lw=1.2)
    for m, c in (("curve", ORANGE), ("chord", GREEN), ("clamp", BLUE)):
        a.plot(ts, P[n]["paths"]["decoded_example"][f"{m}:{k}"], color=c)
    a.set_xlabel("path position t"); a.set_ylabel({"direction": "decoded θ (°)", "speed": "decoded speed (m/s)"}[n])
    a.set_title(f"{n}: path {k.replace('->', ' → ')}")
    if n == "speed":
        a.set_yscale("log"); a.set_yticks([0.5, 1, 2, 4]); a.set_yticklabels(["0.5", "1", "2", "4"]); a.minorticks_off()
axes[1, 2].legend(handles=[Line2D([], [], color=ORANGE, label="along the curve"), Line2D([], [], color=GREEN, label="chord in PCA space"),
                           Line2D([], [], color=BLUE, label="clamp coordinates"), Line2D([], [], color=DARK, ls=(0, (2, 2)), label="ideal (linear in u)")],
                  loc="center left", bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f5_manifold_steering.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
