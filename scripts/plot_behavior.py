"""f6_behavior.png from outputs/results/steer_behavior.json."""
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
BLUE, ORANGE, GRAY, DARK = "#1f77b4", "#ff7f0e", "#7f7f7f", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False})
res = json.loads((OUT / "results" / "steer_behavior.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
x = np.arange(3); w = 0.26
conds = (("subspace", BLUE), ("spline", ORANGE), ("random", GRAY))
for a, key, title in ((axes[0], "within_target_pct", "forecast moved to the steering target"),
                      (axes[1], "within_original_pct", "forecast still shows the original value")):
    for i, (c, col) in enumerate(conds):
        a.bar(x + (i - 1) * w, [res[n][c][key] for n in VARS], w, color=col)
    for xi, n in zip(x, VARS):
        a.hlines(res[n]["unsteered"][key], xi - 0.42, xi + 0.42, color=DARK, ls=(0, (2, 2)), lw=1.2)
    a.set_xticks(x); a.set_xticklabels(VARS); a.set_title(title); a.set_ylim(0, 102)
axes[0].set_ylabel("% of clips (higher is better)")
axes[0].text(2.42, res["acceleration"]["unsteered"]["within_target_pct"] + 2, "no steering", fontsize=8, color=DARK, ha="right")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="multi-probe subspace steering"), Line2D([], [], color=ORANGE, lw=6, label="spline steering"),
                    Line2D([], [], color=GRAY, lw=6, label="random edit, same size")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), borderaxespad=0)
fig.text(0.5, -0.03, "behavior = predictor forecast of frames 9-16 from frames 1-8; edit at encoder layer 12; tolerance ±15°, ±0.375 m/s, ±0.975 m/s²",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f6_behavior.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
