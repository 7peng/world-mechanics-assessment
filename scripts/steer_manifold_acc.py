"""Part 2 accuracy metrics: % of steered clips decoded within tolerance of the target
(direction ±15°, speed/acceleration ±10% of range = 0.375 m/s, 0.975 m/s²), and % of clips whose
direction is still decoded within 15° of its true value after steering speed / acceleration.
Conditions: seen values, held-out values, outside fitted range. Methods: subspace steering (Part 1),
spline steering (Part 2). Also the unsteered probe's own within-tolerance rate and chance.
Saves outputs/results/steer_manifold_acc.json and draws f5_manifold_steering.png.
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

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.inlp import run_inlp
from src.manifold import fit_manifold
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL, load_splits
from src.steer import clamp_coords, steer, target_vec

L, N_PCS, N_CLAMP = 12, 64, 10
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}
RANGE = {"direction": 360.0, "speed": 3.75, "acceleration": 9.75}
KBEST = {n: json.loads((OUT / "results" / "manifolds.json").read_text())[n]["K"] for n in DATASETS}
ALPHA = {n: json.loads((OUT / "results" / "probe_layers.json").read_text())[n]["raw"]["vjepa2"][L]["alpha"] for n in DATASETS}
err = lambda n, a, b: circ_err(a, b) if n == "direction" else np.abs(a - b)
dec = lambda n, p, X: angle_deg(p.predict(X)) if n == "direction" else p.predict(X)[:, 0]
within = lambda n, a, b: float((err(n, a, b) <= TOL[n]).mean() * 100)
chance = {n: (2 * TOL[n] / RANGE[n]) * 100 for n in DATASETS}

RES = OUT / "results" / "steer_manifold_acc.json"
if "--plot-only" in sys.argv and RES.exists():
    res = json.loads(RES.read_text())
    DATASETS_RUN = []
else:
    res = {"tolerance": TOL, "chance": chance}
    DATASETS_RUN = DATASETS
for name in DATASETS_RUN:
    df = load_manifest(name)
    lab = df[LABEL[name]].values
    y = targets(name, df)
    X = load_features(name)[:, L].astype(np.float64)
    sp = load_splits()[name]
    res[name] = {}
    for cond, split, key, tvals in (("seen", "primary", "test", None),
                                    ("heldout", "value_heldout", "heldout_values", np.array(sp["heldout_values"])),
                                    ("ends", "value_heldout_ends", "heldout_values", np.array(sp["heldout_values_ends"]))):
        idx = split_idx(name, split)
        tr, va, st = idx["train"], idx["val"], idx[key]
        if tvals is None:
            tvals = {"direction": np.arange(0, 360, 45.0), "speed": np.linspace(0.25, 4.0, 8), "acceleration": np.linspace(0.25, 10.0, 8)}[name]
        man = fit_manifold(name, X[tr], lab[tr], N_PCS, "lsq", KBEST[name])
        inlp = run_inlp(name, X.astype(np.float32), y, {"train": tr, "val": va}, N_CLAMP, eval_splits=("val",))
        probe = fit_ridge(X[va], y[va], alpha=ALPHA[name])
        thp = None if name == "direction" else fit_ridge(X[va], targets("direction", df)[va], alpha=ALPHA[name])
        Xs, th_true = X[st], df.theta_degrees.values[st]
        r = res[name][cond] = {"unsteered_within": within(name, dec(name, probe, Xs), lab[st])}
        if thp is not None:
            r["theta_unsteered_within"] = float((circ_err(dec("direction", thp, Xs), th_true) <= 15).mean() * 100)
        for m in ("subspace", "spline"):
            hit, keep = [], []
            for t in tvals:
                if m == "subspace":
                    V, c = clamp_coords(inlp, N_CLAMP, target_vec(name, float(t)))
                    Xe = steer(inlp.z(Xs), V, c) * inlp.sd + inlp.mu
                else:
                    Xe = man.steer(Xs, float(t))
                hit.append(within(name, dec(name, probe, Xe), float(t)))
                if thp is not None:
                    keep.append(float((circ_err(dec("direction", thp, Xe), th_true) <= 15).mean() * 100))
            r[m] = {"target_within": float(np.mean(hit))}
            if keep:
                r[m]["theta_kept_within"] = float(np.mean(keep))
        print(name, cond, {k: (round(v, 1) if isinstance(v, float) else {kk: round(vv, 1) for kk, vv in v.items()}) for k, v in r.items()}, flush=True)
if DATASETS_RUN:
    RES.write_text(json.dumps(res, indent=1))

# ---------------- figure
BLUE, ORANGE, DARK = "#1f77b4", "#ff7f0e", "#333333"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False})
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, gridspec_kw={"wspace": 0.12})
w = 0.36
x = np.arange(3)
for a, n in zip(axes[:3], DATASETS):
    for i, (m, col) in enumerate((("subspace", BLUE), ("spline", ORANGE))):
        a.bar(x + (i - 0.5) * w, [res[n][c][m]["target_within"] for c in ("seen", "heldout", "ends")], w, color=col)
    a.axhline(chance[n], color="#999", lw=0.9)
    a.set_xticks(x); a.set_xticklabels(["target seen\nin training", "target value\nnever seen", "target outside\nfitted range"])
    a.set_title(f"steering {n}: on target")
axes[0].set_ylabel("% of steered clips  (higher is better)")
axes[0].set_ylim(0, 102)
axes[0].text(2.45, chance["direction"] + 1.5, "chance", fontsize=8, color="#777", ha="right")
a = axes[3]
x2 = np.arange(2)
for i, (m, col) in enumerate((("subspace", BLUE), ("spline", ORANGE))):
    a.bar(x2 + (i - 0.5) * w, [res[n]["seen"][m]["theta_kept_within"] for n in ("speed", "acceleration")], w, color=col)
for xi, n in zip(x2, ("speed", "acceleration")):
    a.hlines(res[n]["seen"]["theta_unsteered_within"], xi - 0.45, xi + 0.45, color=DARK, ls=(0, (2, 2)), lw=1.2)
a.axhline(chance["direction"], color="#999", lw=0.9)
a.text(-0.45, res["speed"]["seen"]["theta_unsteered_within"] + 1.5, "before steering", fontsize=8, color=DARK, ha="left", va="bottom")
a.set_xticks(x2); a.set_xticklabels(["while steering\nspeed", "while steering\nacceleration"])
a.set_title("direction unchanged")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="multi-probe subspace steering (Part 1)"),
                    Line2D([], [], color=ORANGE, lw=6, label="spline steering (Part 2)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), borderaxespad=0)
fig.text(0.5, -0.03, "on target = decoded within ±15° (direction), ±0.375 m/s (speed), ±0.975 m/s² (acceleration); 10% of each range",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "f5_manifold_steering.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
