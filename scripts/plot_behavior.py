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
LIGHT_ORANGE = "#fdc692"
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
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for j, (a, n) in enumerate(zip(axes, VARS)):
    sw, tk, ts = R(f"behavior_sweep_ctx_{n}.json"), R(f"behavior_token_ctx_{n}.json"), R(f"behavior_token_spline_{n}.json")
    Ls = sorted(int(l) for l in sw["layers"])
    a.axhline(sw["donor"]["12"]["within_pct"], color=GREEN, lw=1.2)
    a.axhline(sw["clean"]["within_target_pct"], color=GRAY, ls=DOT)
    a.plot(Ls, [sw["layers"][str(l)]["spline"]["within_pct"] for l in Ls], color=LIGHT_ORANGE)
    a.plot(Ls, [sw["layers"][str(l)]["subspace"]["within_pct"] for l in Ls], color=BLUE)
    Lsh = [l for l in Ls if "shift" in sw["layers"][str(l)]]
    if Lsh:
        a.plot(Lsh, [sw["layers"][str(l)]["shift"]["within_pct"] for l in Lsh], color=ORANGE)
    Lt = sorted(int(l) for l in tk["layers"])
    a.plot(Lt, [tk["layers"][str(l)]["all_tokens"]["within_pct"] for l in Lt], color=BLUE, ls=DASH)
    Lp = sorted(int(l) for l in ts["layers"])
    a.plot(Lp, [ts["layers"][str(l)]["displace"]["within_pct"] for l in Lp], color=ORANGE, ls=DASH)
    a.set_title(n); a.set_xticks([4, 8, 12, 16, 20, 24]); a.set_xlim(3, 25); a.set_ylim(0, 105)
    if j == 1:
        a.set_xlabel("edit layer")
axes[0].set_ylabel("% on target")
legend_below(fig, [Line2D([], [], color=BLUE, label="subspace, pooled"), Line2D([], [], color=ORANGE, label="shift along curve, pooled"),
                   Line2D([], [], color=LIGHT_ORANGE, label="naive spline, pooled"),
                   Line2D([], [], color=BLUE, ls=DASH, label="subspace, per-token"), Line2D([], [], color=ORANGE, ls=DASH, label="shift along curve, per-token"),
                   Line2D([], [], color="none", label=" "),
                   Line2D([], [], color=GREEN, lw=1.2, label="token swap (ceiling)"), Line2D([], [], color=GRAY, ls=DOT, label="no steering")], 3, y=-0.2)
fig.tight_layout()
fig.savefig(FIG / "f6_behavior_layers.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

# ---------------- f7: mean vs pattern
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for j, (a, n) in enumerate(zip(axes, VARS)):
    d = R(f"behavior_decompose_{n}.json")
    Ls = sorted(int(l) for l in d["layers"])
    for k, col, ls in (("full", GREEN, "-"), ("pattern", BLUE, "-"), ("mean", ORANGE, "-")):
        a.plot(Ls, [d["layers"][str(l)][k]["follows_donor_pct"] for l in Ls], color=col, ls=ls)
    a.set_title(n); a.set_xticks([0, 4, 8, 12, 16, 20, 24]); a.set_ylim(0, 105)
    if j == 1:
        a.set_xlabel("swap layer")
axes[0].set_ylabel("% following donor")
legend_below(fig, [Line2D([], [], color=GREEN, label="all donor tokens"),
                   Line2D([], [], color=BLUE, label="donor pattern only"),
                   Line2D([], [], color=ORANGE, label="donor average only")], 3, y=-0.1)
fig.tight_layout()
fig.savefig(FIG / "f7_mean_vs_pattern.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

# ---------------- f8: final comparison (if available)
if all((OUT / "results" / f"behavior_final_{n}.json").exists() for n in VARS):
    M = [("pooled_subspace", BLUE, None, "subspace, pooled"), ("pooled_shift", ORANGE, None, "shift along curve, pooled"),
         ("pooled_spline", LIGHT_ORANGE, None, "naive spline, pooled"),
         ("token_subspace", BLUE, "//", "subspace, per-token"), ("token_spline", ORANGE, "//", "shift along curve, per-token")]
    Fs = {n: R(f"behavior_final_{n}.json") for n in VARS}
    L = "24" if all("24" in Fs[n]["layers"] for n in VARS) else sorted(Fs["speed"]["layers"])[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    a = axes[0]
    x = np.arange(3); w = 0.17
    for i, (m, col, hatch, lab) in enumerate(M):
        if m not in Fs["speed"]["layers"][L]:
            continue
        a.bar(x + (i - 2) * w, [Fs[n]["layers"][L][m]["on_target_pct"] for n in VARS], w, color=col, hatch=hatch, edgecolor="white", lw=0)
    for xi, n in zip(x, VARS):
        a.hlines(Fs[n]["clean"]["on_target_pct"], xi - 0.45, xi + 0.45, color=GRAY, ls=DOT)
    a.set_xticks(x); a.set_xticklabels(VARS); a.set_ylim(0, 105)
    a.set_ylabel("% on target"); a.set_title("on target")
    a = axes[1]
    x = np.arange(2)
    for i, (m, col, hatch, lab) in enumerate(M):
        if m not in Fs["speed"]["layers"][L]:
            continue
        a.bar(x + (i - 2) * w, [Fs[n]["layers"][L][m]["direction_kept_pct"] for n in ("speed", "acceleration")], w, color=col, hatch=hatch, edgecolor="white", lw=0)
    for xi, n in zip(x, ("speed", "acceleration")):
        a.hlines(Fs[n]["clean"]["direction_kept_pct"], xi - 0.45, xi + 0.45, color=DARK, ls=DASH)
    a.set_xticks(x); a.set_xticklabels(["speed", "acceleration"]); a.set_ylim(0, 105)
    a.set_ylabel("% direction kept"); a.set_title("direction kept")
    legend_below(fig, [plt.Rectangle((0, 0), 1, 1, fc=c, hatch=h, ec="white", lw=0.8, label=l) for _, c, h, l in M]
                 + [Line2D([], [], color=GRAY, ls=DOT, label="no steering"), Line2D([], [], color=DARK, ls=DASH, label="direction kept, no steering")], 3, y=-0.2)
    fig.tight_layout()
    fig.savefig(FIG / "f8_behavior_final.png", dpi=150, bbox_inches="tight", pad_inches=0.1)

# ---------------- f9: paths, direction +180°
# one row per edit layer: forecast turn vs intended turn for single clips (3 methods), then % at the intended angle
DEC = {L: OUT / "results" / f"behavior_paths_direction_decoded_L{L}.json" for L in ("24", "12")}
if all(f.exists() for f in DEC.values()):
    M9 = [("subspace_linear", BLUE, "-", "straight path, subspace, pooled"), ("shift", ORANGE, "-", "curve path, shift along curve, pooled"),
          ("spline", LIGHT_ORANGE, "-", "curve path, naive spline, pooled"),
          ("token_chord", BLUE, DASH, "straight path, per-token"), ("token_spline", ORANGE, DASH, "curve path, shift along curve, per-token")]
    SHOW = [("subspace_linear", BLUE, "straight path"), ("shift", ORANGE, "curve path, pooled"), ("token_spline", ORANGE, "curve path, per-token")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for i, L in enumerate(("24", "12")):
        P = R(DEC[L].name)["layers"][L]["span180"]
        ideal = np.array(P["ideal"]); turn = np.linspace(0, 180, ideal.shape[1])
        for j, (m, col, lab) in enumerate(SHOW):
            a = axes[i, j]
            d = np.array(P[m]["decoded"])
            rel = (d - ideal[:, :1] + 90) % 360 - 90                                   # forecast turn, in [-90, 270)
            a.fill_between(turn, turn - 15, turn + 15, color="white", lw=0)
            a.plot(turn, turn, color=DARK, ls=DOT, lw=1.2)
            for k in range(min(40, len(rel))):
                a.plot(turn, rel[k], color=col, lw=0.7, alpha=0.35)
            a.set_xlim(0, 180); a.set_ylim(-90, 270); a.set_xticks([0, 90, 180]); a.set_yticks([-90, 0, 90, 180, 270])
            if i == 0:
                a.set_title(lab)
            if i == 1:
                a.set_xlabel("intended turn (°)")
            if j == 0:
                a.set_ylabel(f"layer {L}\nforecast turn (°)")
            else:
                a.set_yticklabels([])
        a = axes[i, 3]
        for m, col, ls, lab in M9:
            a.plot(turn, P[m]["on_path_pct_by_waypoint"], color=col, ls=ls)
        a.set_ylim(0, 105); a.set_xlim(0, 180); a.set_xticks([0, 90, 180])
        a.set_ylabel("% within 15° of intended")
        if i == 0:
            a.set_title("summary")
        if i == 1:
            a.set_xlabel("intended turn (°)")
    legend_below(fig, [Line2D([], [], color=DARK, ls=DOT, lw=1.2, label="ideal (±15° band)")]
                 + [Line2D([], [], color=c, ls=ls, label=l) for _, c, ls, l in M9], 3, y=-0.07)
    fig.tight_layout()
    fig.savefig(FIG / "f9_behavior_paths.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
print("ok")
