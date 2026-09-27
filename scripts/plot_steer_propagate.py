"""Figure for Experiment 3b: does steering at layer L survive the remaining blocks?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import DATASETS, OUT
from src.plotting import COLORS, INK2, NEUTRAL, plt

units = {"direction": "circular MAE to target (°)", "speed": "MAE to target (m/s)",
         "acceleration": "MAE to target (m/s²)"}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
summary = {}
for j, name in enumerate(DATASETS):
    r = json.loads((OUT / "results" / f"steer_propagate_{name}.json").read_text())
    layers, c = r["layers"], COLORS[name]
    conds = r["conditions"]
    clean = conds[0]
    ax = axes[0, j]
    shades = np.linspace(0.3, 1.0, len(r["n_list"]))
    summary[name] = {}
    for N, a in zip(r["n_list"], shades):
        rs = [x for x in conds if x["kind"] == "steer" and x["N"] == N]
        tt = np.mean([x["to_target"] for x in rs], 0)
        summary[name][N] = {"at_L": tt[0], "at_24": tt[-1]}
        ax.plot(layers, tt, "-o", ms=3, color=c, alpha=a, label=f"steer, N={N}")
    rs = [x for x in conds if x["kind"] == "random"]
    ax.plot(layers, np.mean([x["to_target"] for x in rs], 0), color=NEUTRAL, label=f"random, matched norm (N={r['n_random']})")
    ax.plot(layers, clean["to_truth"], color=INK2, ls="--", lw=1.5, label="unsteered probe error (floor)")
    ax.set_title(f"{name}: steer at layer {r['layer']}, read out downstream")
    ax.set_xlabel("read-out layer"); ax.set_ylabel(units[name])
    if j == 0:
        ax.legend(fontsize=8)
    # dose-response at final layer
    ax = axes[1, j]
    key = "within_22.5" if name == "direction" else "decoded_mean"
    for N, a in zip(r["n_list"], shades):
        rs = sorted([x for x in conds if x["kind"] == "steer" and x["N"] == N], key=lambda x: x["target"])
        ax.plot([x["target"] for x in rs], [x[key][-1] for x in rs], "-o", ms=3, color=c, alpha=a, label=f"N={N}")
    tg = r["targets"]
    if name == "direction":
        ax.axhline(45 / 360, color=INK2, ls="--", lw=1, label="chance")
        ax.set_ylim(0, 1.02); ax.set_ylabel("fraction within ±22.5° of target, layer 24")
    else:
        ax.plot(tg, tg, color=INK2, ls="--", lw=1, label="ideal")
        ax.set_ylabel("mean decoded value at layer 24")
    ax.set_xlabel("target")
    if name != "direction":
        rs = [x for x in conds if x["kind"] == "steer" and x["N"] == 20]
        th = np.mean([x["theta_mae"][-1] for x in rs])
        ax.text(0.02, 0.97, f"off-target θ error @24, N=20: {th:.1f}° (clean {clean['theta_mae'][-1]:.1f}°)",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    if j == 0:
        ax.legend(fontsize=8)
fig.suptitle("Subspace steering propagated through the frozen encoder (held-out test clips, val-fit read-out probes)",
             x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "figures" / "steer_propagate.png", dpi=130)
print(json.dumps(summary, indent=1))
