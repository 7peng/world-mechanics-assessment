"""Behavior figures (context-only predictor protocol).
f6_behavior_layers.png  % of forecasts on target vs the layer where the edit is applied (unit strength), 4 methods
f7_mean_vs_pattern.png  % of forecasts that follow a donor clip when only its token average / token pattern is swapped in
f8_behavior_final.png   best strength per method (chosen on val), test: % on target and % with direction kept
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
BLUE, ORANGE, GRAY, DARK, GREEN = "#1f77b4", "#ff7f0e", "#7f7f7f", "#333333", "#2ca02c"
DASH, DOT = (0, (4, 2)), (0, (1, 1.6))
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False,
                     "lines.linewidth": 1.6})
R = lambda f: json.loads((OUT / "results" / f).read_text())
FIG = OUT / "figures" / "report"


def legend_below(fig, handles, ncol, y=-0.08):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), borderaxespad=0)


# ---------------- f6: by layer
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, n in zip(axes, VARS):
    sw, tk, ts = R(f"behavior_sweep_ctx_{n}.json"), R(f"behavior_token_ctx_{n}.json"), R(f"behavior_token_spline_{n}.json")
    Ls = sorted(int(l) for l in sw["layers"])
    a.plot(Ls, [sw["layers"][str(l)]["subspace"]["within_pct"] for l in Ls], color=BLUE)
    a.plot(Ls, [sw["layers"][str(l)]["spline"]["within_pct"] for l in Ls], color=ORANGE)
    Lt = sorted(int(l) for l in tk["layers"])
    a.plot(Lt, [tk["layers"][str(l)]["all_tokens"]["within_pct"] for l in Lt], color=BLUE, ls=DASH)
    Lp = sorted(int(l) for l in ts["layers"])
    a.plot(Lp, [ts["layers"][str(l)]["displace"]["within_pct"] for l in Lp], color=ORANGE, ls=DASH)
    a.axhline(sw["clean"]["within_target_pct"], color=GRAY, ls=DOT)
    a.axhline(sw["donor"]["12"]["within_pct"], color=GREEN, lw=1.2)
    a.set_title(f"steering {n}"); a.set_xlabel("encoder layer where the edit is applied"); a.set_xticks([4, 8, 12, 16, 20, 24])
    a.set_ylim(0, 105)
axes[0].set_ylabel("% of forecasts on target (higher is better)")
axes[0].text(4, R("behavior_sweep_ctx_direction.json")["clean"]["within_target_pct"] + 2, "no steering", fontsize=8, color=GRAY)
axes[0].text(4, 101, "swap in a target clip's tokens (ceiling)", fontsize=8, color=GREEN, va="top")
legend_below(fig, [Line2D([], [], color=BLUE, label="subspace steering, pooled edit"), Line2D([], [], color=ORANGE, label="spline steering, pooled edit"),
                   Line2D([], [], color=BLUE, ls=DASH, label="subspace steering, per-token edit"), Line2D([], [], color=ORANGE, ls=DASH, label="spline steering, per-token edit")], 4)
fig.tight_layout()
fig.savefig(FIG / "f6_behavior_layers.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

# ---------------- f7: mean vs pattern
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, n in zip(axes, VARS):
    d = R(f"behavior_decompose_{n}.json")
    Ls = sorted(int(l) for l in d["layers"])
    for k, col, ls in (("full", GREEN, "-"), ("pattern", BLUE, "-"), ("mean", ORANGE, "-")):
        a.plot(Ls, [d["layers"][str(l)][k]["follows_donor_pct"] for l in Ls], color=col, ls=ls)
    a.set_title(n); a.set_xlabel("encoder layer where tokens are swapped"); a.set_xticks([0, 4, 8, 12, 16, 20, 24]); a.set_ylim(0, 105)
axes[0].set_ylabel("% of forecasts showing the donor's motion")
legend_below(fig, [Line2D([], [], color=GREEN, label="all tokens from the donor"),
                   Line2D([], [], color=BLUE, label="donor's token pattern, original token average"),
                   Line2D([], [], color=ORANGE, label="donor's token average, original token pattern")], 3)
fig.tight_layout()
fig.savefig(FIG / "f7_mean_vs_pattern.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

# ---------------- f8: final comparison (if available)
if all((OUT / "results" / f"behavior_final_{n}.json").exists() for n in VARS):
    M = [("pooled_subspace", BLUE, None, "subspace, pooled"), ("pooled_spline", ORANGE, None, "spline, pooled"),
         ("token_subspace", BLUE, "//", "subspace, per-token"), ("token_spline", ORANGE, "//", "spline, per-token")]
    Fs = {n: R(f"behavior_final_{n}.json") for n in VARS}
    L = "24" if all("24" in Fs[n]["layers"] for n in VARS) else sorted(Fs["speed"]["layers"])[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    a = axes[0]
    x = np.arange(3); w = 0.2
    for i, (m, col, hatch, lab) in enumerate(M):
        a.bar(x + (i - 1.5) * w, [Fs[n]["layers"][L][m]["on_target_pct"] for n in VARS], w, color=col, hatch=hatch, edgecolor="white", lw=0)
    for xi, n in zip(x, VARS):
        a.hlines(Fs[n]["clean"]["on_target_pct"], xi - 0.45, xi + 0.45, color=GRAY, ls=DOT)
    a.set_xticks(x); a.set_xticklabels([f"steering {n}" for n in VARS]); a.set_ylim(0, 105)
    a.set_ylabel("% of forecasts on target (higher is better)"); a.set_title(f"reaches the target (edit at layer {L}, strength tuned on val)")
    a = axes[1]
    x = np.arange(2)
    for i, (m, col, hatch, lab) in enumerate(M):
        a.bar(x + (i - 1.5) * w, [Fs[n]["layers"][L][m]["direction_kept_pct"] for n in ("speed", "acceleration")], w, color=col, hatch=hatch, edgecolor="white", lw=0)
    for xi, n in zip(x, ("speed", "acceleration")):
        a.hlines(Fs[n]["clean"]["direction_kept_pct"], xi - 0.45, xi + 0.45, color=DARK, ls=DASH)
    a.set_xticks(x); a.set_xticklabels(["steering speed", "steering acceleration"]); a.set_ylim(0, 105)
    a.set_ylabel("% of forecasts with direction intact"); a.set_title("leaves direction intact")
    legend_below(fig, [plt.Rectangle((0, 0), 1, 1, fc=c, hatch=h, ec="white", lw=0.8, label=l) for _, c, h, l in M]
                 + [Line2D([], [], color=GRAY, ls=DOT, label="no steering"), Line2D([], [], color=DARK, ls=DASH, label="direction intact before steering")], 3, y=-0.14)
    fig.tight_layout()
    fig.savefig(FIG / "f8_behavior_final.png", dpi=150, bbox_inches="tight", pad_inches=0.1)

# ---------------- f9: paths, direction +180°
if (OUT / "results" / "behavior_paths_direction.json").exists():
    P = R("behavior_paths_direction.json")
    ts = np.array(P["waypoints"])
    M9 = [("subspace_linear", BLUE, "-", "linear path, subspace steering (Part 1)"), ("chord", BLUE, DASH, "linear path, straight chord (paper's baseline)"),
          ("spline", ORANGE, "-", "spline path, pooled (paper)"), ("token_spline", ORANGE, DASH, "spline path, per-token")]
    Ls = [l for l in ("24", "12") if l in P["layers"] and "span180" in P["layers"][l]]
    fig, axes = plt.subplots(1, len(Ls), figsize=(5.2 * len(Ls), 3.6), sharey=True, squeeze=False)
    for a, L in zip(axes[0], Ls):
        for m, col, ls, lab in M9:
            a.plot(ts, P["layers"][L]["span180"][m]["on_path_pct_by_waypoint"], color=col, ls=ls)
        a.set_ylim(0, 105); a.set_xlabel("position along the steering path (0 = start, 1 = target)")
        a.set_title(f"turning direction by 180°, edit at layer {L}")
    axes[0][0].set_ylabel("% of forecasts at the intended\nintermediate direction (±15°)")
    legend_below(fig, [Line2D([], [], color=c, ls=ls, label=l) for _, c, ls, l in M9], 2, y=-0.2)
    fig.tight_layout()
    fig.savefig(FIG / "f9_behavior_paths.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
print("ok")
