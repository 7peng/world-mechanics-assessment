"""Figure suite for the Part 1 report: three main figures (f1_probes, f2_nullspace, f3_steering)
and supplementary ones (supp_*). Outputs to outputs/figures/report/.

Style: white ground, thin lines, light spines, no grid, direct labels at line ends.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import OUT, load_manifest
from src.features import load_features, split_idx
from src.probes import fit_ridge, targets

R = lambda f: json.loads((OUT / "results" / f).read_text())
FIG = OUT / "figures" / "report"
FIG.mkdir(parents=True, exist_ok=True)
VARS = ["direction", "speed", "acceleration"]
C = {"direction": "#3b74c4", "speed": "#e0743f", "acceleration": "#2f9b78"}
DARK, GRAY, LIGHT = "#333333", "#8c8c8c", "#c8c8c8"
UNIT = {"direction": "circular MAE (°)", "speed": "MAE (m/s)", "acceleration": "MAE (m/s²)"}
YMAX_ERR = {"direction": 100, "speed": 1.5, "acceleration": 4}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.8, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.labelcolor": DARK, "text.color": DARK, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "lines.linewidth": 1.5, "lines.markersize": 3.5, "legend.frameon": False, "legend.fontsize": 8,
})


def end_labels(ax, x, items, gap=0.06):
    """items: (y, text, color). Labels right of x, spread vertically so they don't overlap."""
    lo, hi = ax.get_ylim()
    g = gap * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    pos = []
    for y, _, _ in items:                       # push up
        pos.append(y if not pos else max(y, pos[-1] + g))
    pos[-1] = min(pos[-1], hi - g / 2)          # then push down from the top edge
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - g)
    for (y, text, color), p in zip(items, pos):
        ax.annotate(text, (x, p), xytext=(4, 0), textcoords="offset points", color=color, fontsize=8, va="center")


def hline(ax, y, text, color=GRAY, ls=(0, (2, 2))):
    lo, hi = ax.get_ylim()
    if not lo <= y <= hi:
        return
    ax.axhline(y, color=color, lw=0.9, ls=ls)
    ax.text(ax.get_xlim()[1], y, " " + text, color=color, fontsize=7.5, va="center")


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=11)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------- F1 layer-wise probing
pl = R("probe_layers.json")
pp = R("paper_protocol_vjepa2.json")
patch = R("probe_patches.json")
patch_layers = patch["layers"]
patch_med = [float(np.median(patch["per_patch_r2"][str(l)])) for l in patch_layers]
fig2c = {"direction": (0.22, 0.93), "speed": (0.85, 0.96), "acceleration": (0.78, 0.90)}
fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
X = np.arange(25)
for j, n in enumerate(VARS):
    r = pl[n]["raw"]
    k = "circ_mae_deg" if n == "direction" else "mae"
    ours = [m["r2"] for m in r["vjepa2"]]
    rnd = [m["r2"] for m in r["vjepa2_random"]]
    pap = pp[n]["layerwise"]["exact_mean"]
    a = axes[0, j]
    a.set_xlim(0, 27.5); a.set_ylim(0, 1.03)
    a.plot(X, ours, color=C[n])
    a.plot(X, rnd, color=GRAY, lw=1.2)
    a.plot(X[1:], pap, color=DARK, lw=1, ls=(0, (4, 2)))
    a.plot([1, 9], fig2c[n], "o", mfc="white", mec=DARK, ms=6)
    a.set_title(n); a.set_xticks([0, 8, 16, 24])
    labels = [(ours[24], "ours, pooled", C[n]), (rnd[24], "random init", GRAY), (pap[-1], "paper protocol", DARK)]
    if n == "direction":
        a.plot(patch_layers, patch_med, color=C[n], lw=1.2, ls=(0, (1, 1.5)), marker="o", ms=2.5)
        labels.append((patch_med[-1], "ours, per-patch median", C[n]))
    end_labels(a, 24, labels)
    if j == 0:
        a.set_ylabel("test R²")
        a.annotate("paper, Fig. 2c", (1, 0.22), xytext=(8, 0), textcoords="offset points", color=DARK, fontsize=7.5, va="center")
    b = axes[1, j]
    e_ours = [m[k] for m in r["vjepa2"]]
    e_rnd = [m[k] for m in r["vjepa2_random"]]
    b.set_xlim(0, 27.5); b.set_ylim(0, {"direction": 40, "speed": 0.4, "acceleration": 1.0}[n])
    b.plot(X[1:], e_ours[1:], color=C[n])
    b.plot(X[1:], e_rnd[1:], color=GRAY, lw=1.2)
    b.set_xticks([0, 8, 16, 24]); b.set_ylabel(UNIT[n])
    hline(b, r["centroid_poly2"][k], "centroid baseline")
    hline(b, r["pixels"][k], "pixel baseline", ls=(0, (1, 2)))
    end_labels(b, 24, [(e_ours[24], "ours", C[n]), (e_rnd[24], "random init", GRAY)])
    if j == 1:
        b.set_xlabel("layer (0 = patch embedding; paper index = layer − 1)")
title(fig, "Linear probes by layer")
fig.tight_layout()
save(fig, "f1_probes.png")

# ---------------------------------------------------------------- supplementary: probe type × data
pc = R("probe_protocol_check.json")
L = pc["layers"]
sty = [("ridge_z", C["direction"], "-", "ridge (ours)"), ("gd_paper", DARK, "-", "gradient probe, 100 epochs (paper)"),
       ("gd_paper_long", DARK, (0, (1, 1.5)), "gradient probe, 1000 epochs")]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for a, reg, t in ((axes[0], "full", "full direction set, 1,500 clips"), (axes[1], "paperlike", "paper-like subset, 96 clips")):
    a.set_xlim(0, 33); a.set_ylim(0, 1.03)
    items = []
    for key, c, ls, lab in sty:
        y = [x["r2"] for x in pc[reg][key]]
        a.plot(L, y, color=c, ls=ls, marker="o", ms=2.5)
        items.append((y[-1], lab, c))
    end_labels(a, L[-1], items, gap=0.07)
    a.plot([1, 9], [0.22, 0.93], "o", mfc="white", mec=DARK, ms=6)
    a.set_title(t); a.set_xticks([1, 6, 12, 18, 24]); a.set_xlabel("layer")
axes[0].set_ylabel("direction R² (5-fold CV)")
axes[0].annotate("paper, Fig. 2c", (1, 0.22), xytext=(8, 0), textcoords="offset points", color=DARK, fontsize=7.5, va="center")
title(fig, "Direction decodability depends on probe training and data size")
fig.tight_layout()
save(fig, "supp_probe_type.png")

# ---------------------------------------------------------------- F2 nullspace probing
inl = R("inlp.json")
Lm = inl["layer"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
for a, n in zip(axes[:3], VARS):
    k = 2 if n == "direction" else 1
    m = {src: [v["r2"] for v in inl["main"][n][src]["test"]] for src in ("probe", "random", "pca")}
    xr = np.arange(len(m["probe"])) * k
    a.set_ylim(-0.05, 1.03); a.set_xlim(0, xr[-1] * 1.4)
    a.plot(xr, m["probe"], color=C[n])
    a.plot(xr, m["random"], color=GRAY, lw=1.2)
    a.plot(xr, m["pca"], color=DARK, lw=1, ls=(0, (4, 2)))
    a.set_title(f"{n}, layer {Lm}"); a.set_xlabel("dimensions removed")
    end_labels(a, xr[-1], [(m["probe"][-1], "probe directions", C[n]), (m["random"][-1], "random directions", GRAY), (m["pca"][-1], "top PCs", DARK)])
axes[0].set_ylabel("test R² of next probe")
a = axes[3]
a.set_xlim(0, 30); a.set_ylim(0, 170)
items = []
for n in VARS:
    k = 2 if n == "direction" else 1
    y = [k * (80 if x is None else x) for x in inl["sweep"][n]["n_to_0.3"]]
    a.plot(range(25), y, color=C[n], marker="o", ms=2.5)
    items.append((y[-1], n, C[n]))
end_labels(a, 24, items)
a.set_xticks([0, 8, 16, 24]); a.set_title("dims removed until val R² < 0.3"); a.set_xlabel("layer")
a.text(1.5, 162, "160 / 80 = never within budget", fontsize=7.5, color=GRAY)
title(fig, "Iterative nullspace probing")
fig.tight_layout()
save(fig, "f2_nullspace.png")

# ---------------------------------------------------------------- F3 steering: same-layer and propagated
sp = R("steer_pooled.json")
N = sp["n_list"]
pers = R("diag_persistence.json")
fig, axes = plt.subplots(2, 3, figsize=(11, 5.8))
for j, n in enumerate(VARS):
    a = axes[0, j]
    s = {"strict": [x["to_target"]["strict"] for x in sp[n]["steer"]],
         "truth": [x["to_truth"]["strict"] for x in sp[n]["steer"]],
         "random": [x["to_target"]["strict"] for x in sp[n]["random"]]}
    a.set_xscale("log"); a.set_xlim(0.9, 300); a.set_ylim(0, YMAX_ERR[n])
    a.plot(N, s["strict"], color=C[n], marker="o", ms=2.5)
    a.plot(N, s["random"], color=GRAY, lw=1.2)
    a.plot(N, s["truth"], color=C[n], lw=1, ls=(0, (1, 1.5)))
    end_labels(a, N[-1], [(s["strict"][-1], "to target", C[n]), (s["random"][-1], "random edit", GRAY), (s["truth"][-1], "to original label", C[n])])
    if n == "direction":
        st = pp["direction"]["steer"]["block_output"]["runs"]
        Np = [x["N"] for x in st[0]["steer"]]
        tt = np.array([[x["to_target"] for x in r["steer"]] for r in st])
        a.plot(Np, tt.mean(0), color=DARK, lw=1, ls=(0, (4, 2)), marker="o", ms=2.5)
        a.fill_between(Np, tt.min(0), tt.max(0), color=DARK, alpha=0.08, lw=0)
        a.plot([1, 2, 3, 5, 10, 15, 20], [77, 66, 61, 51, 24, 14, 11.9], "o", mfc="white", mec=DARK, ms=6)
        a.text(28, 40, "○  paper, Fig. 24", color=DARK, fontsize=7.5)
        a.text(28, 31, "- -  paper protocol at layer 8\n      (3 splits; band = range)", color=DARK, fontsize=7.5, va="top")
    a.set_xticks([1, 3, 10, 30, 100]); a.set_xticklabels(["1", "3", "10", "30", "100"])
    a.set_title(f"{n}: read out at layer {sp['layer']}"); a.set_ylabel(UNIT[n])
    if j == 1:
        a.set_xlabel("probes in steering subspace")
    b = axes[1, j]
    sprop = R(f"steer_propagate_{n}.json")
    c = sprop["conditions"]
    b.set_xlim(12, 27.5); b.set_ylim(0, YMAX_ERR[n])
    items = []
    for Nn, alpha in ((1, 0.35), (5, 0.6), (20, 1.0)):
        y = np.mean([x["to_target"] for x in c if x["kind"] == "steer" and x["N"] == Nn], 0)
        b.plot(sprop["layers"], y, color=C[n], alpha=alpha)
        items.append((y[-1], f"N = {Nn}", C[n]))
    rnd = np.mean([x["to_target"] for x in c if x["kind"] == "random"], 0)
    b.plot(sprop["layers"], rnd, color=GRAY, lw=1.2)
    b.plot(sprop["layers"], c[0]["to_truth"], color=DARK, lw=1, ls=(0, (1, 1.5)))
    end_labels(b, 24, items + [(rnd[-1], "random edit", GRAY), (c[0]["to_truth"][-1], "unsteered floor", DARK)])
    b.set_xticks([12, 16, 20, 24]); b.set_title(f"{n}: read out downstream"); b.set_ylabel(UNIT[n] + " to target")
    if j == 1:
        b.set_xlabel("read-out layer (edit applied at layer 12)")
title(fig, "Multi-probe subspace steering")
fig.tight_layout()
save(fig, "f3_steering.png")

# ---------------------------------------------------------------- supplementary: persistence of the edit
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for b, n in zip(axes, VARS):
    P = pers[n]
    b.set_xlim(12, 27.5); b.set_ylim(0, 1.03)
    b.plot(P["layers"], P["retained_frac"], color=C[n], marker="o", ms=2.5)
    b.plot(P["layers"], P["retained_frac_natural"], color=DARK, lw=1, ls=(0, (4, 2)))
    b.plot(P["layers"], P["retained_frac_random"], color=GRAY, lw=1.2)
    end_labels(b, 24, [(P["retained_frac"][-1], "steering edit", C[n]), (P["retained_frac_natural"][-1], "clip-to-clip difference", DARK),
                       (P["retained_frac_random"][-1], "random edit", GRAY)])
    b.set_xticks([12, 16, 20, 24]); b.set_title(n); b.set_xlabel("layer")
axes[0].set_ylabel("fraction of Δ retained along Δ")
title(fig, "How much of an injected shift survives")
fig.tight_layout()
save(fig, "supp_persistence.png")

# ---------------------------------------------------------------- supplementary: displacement confound
sc = R("sanity_checks.json")
Lc = sc["confound"]["layer"]
da, ds = load_manifest("acceleration"), load_manifest("speed")
Fa, Fs = load_features("acceleration")[:, Lc].astype(float), load_features("speed")[:, Lc].astype(float)
ia = split_idx("acceleration")
ya = targets("acceleration", da)
probe = fit_ridge(Fa[ia["train"]], ya[ia["train"]], Fa[ia["val"]], ya[ia["val"]])
pred = probe.predict(Fs)[:, 0]
cen = np.load(OUT / "baselines" / "centroids_speed.npz")["centroid"]
disp = np.linalg.norm(cen[:, -1] - cen[:, 0], axis=1)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
a = axes[0]
a.set_xlim(0, 1.3); a.set_ylim(0, 85)
items = []
for n in ["speed", "acceleration"]:
    d = sc[n]
    a.plot(np.array(d["label_values"]) / max(d["label_values"]), d["mean_displacement_px"], color=C[n], marker="o", ms=2.5)
    items.append((d["mean_displacement_px"][-1], f"{n} set", C[n]))
end_labels(a, 1.0, items)
a.axhline(22, color=GRAY, lw=0.9, ls=(0, (2, 2))); a.text(0.02, 24, "disk diameter", color=GRAY, fontsize=7.5)
a.set_xlabel("label / max label"); a.set_ylabel("distance travelled (px)"); a.set_title("displacement is fixed by the label")
b = axes[1]
b.set_xlim(0, 85); b.set_ylim(0, 14)
b.scatter(disp, pred, s=6, color=C["acceleration"], alpha=0.5, lw=0)
b.set_xlabel("distance travelled (px), constant-velocity clips"); b.set_ylabel("predicted acceleration (m/s²)")
b.set_title(f"acceleration probe on clips with zero acceleration (layer {Lc})")
b.text(3, 12.6, f"true value 0 for every clip\nr = {sc['confound']['corr_with_displacement']:.3f} with displacement", fontsize=7.5, color=DARK)
title(fig, "The acceleration probe reads displacement")
fig.tight_layout()
save(fig, "supp_confound.png")
print("wrote", sorted(p.name for p in FIG.glob("*.png")))
