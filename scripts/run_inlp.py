"""Experiment 2: iterative nullspace probing.

Main layer: chosen from Experiment 1 using *val* metrics only. The paper's criterion (first layer
where direction R² jumps) is degenerate here: all variables reach R² > 0.85 at layer 1. Direction
error, though, keeps improving with depth (12° -> 3°), so we take the earliest layer where direction
val circular MAE is within 1.25x of its best. At that layer we run
INLP plus two controls (random directions, top-PC directions) for each variable. We also sweep
all layers with INLP only, recording how many probes it takes before R² drops below 0.3 / 0.1. The sweep uses val R²
at a val-chosen alpha, so those counts are slightly optimistic; main-layer curves are reported on test.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.inlp import run_inlp
from src.plotting import COLORS, INK2, NEUTRAL, plt
from src.probes import targets

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--n-iter", type=int, default=150)
ap.add_argument("--sweep-iter", type=int, default=80)
args = ap.parse_args()

probe = json.loads((OUT / "results" / "probe_layers.json").read_text())
if args.layer is None:
    v = np.array([m["val"]["circ_mae_deg"] for m in probe["direction"]["raw"]["vjepa2"]])
    args.layer = int(np.argmax(v <= 1.25 * v.min()))
print("INLP layer:", args.layer)

res, curves = {"layer": args.layer, "main": {}, "sweep": {}}, {}
art = OUT / "inlp"
art.mkdir(parents=True, exist_ok=True)
for name in DATASETS:
    df = load_manifest(name)
    y, idx = targets(name, df), split_idx(name)
    F = load_features(name).astype(np.float32)
    X = F[:, args.layer]
    curves[name] = {}
    for src in ("probe", "random", "pca"):
        r = run_inlp(name, X, y, idx, args.n_iter, direction_source=src)
        curves[name][src] = r
        res["main"].setdefault(name, {})[src] = {"val": r.val, "test": r.test}
        print(f"{name}/{src}: test R² iter0 {r.test[0]['r2']:.3f}, iter10 {r.test[10]['r2']:.3f}, "
              f"iter50 {r.test[50]['r2']:.3f}, last {r.test[-1]['r2']:.3f}")
    with open(art / f"inlp_{name}_L{args.layer}.pkl", "wb") as f:
        pickle.dump(curves[name]["probe"], f)
    # layer sweep: number of probes until val R² < thr
    sw = res["sweep"][name] = {"n_to_0.3": [], "n_to_0.1": []}
    for layer in range(F.shape[1]):
        r = run_inlp(name, F[:, layer], y, idx, args.sweep_iter, eval_splits=("val",))
        v = np.array([m["r2"] for m in r.val])
        for thr in (0.3, 0.1):
            below = np.nonzero(v < thr)[0]
            sw[f"n_to_{thr}"].append(int(below[0]) if len(below) else None)  # None = never within budget
    print(name, "probes to R²<0.3 by layer:", sw["n_to_0.3"])

(OUT / "results" / "inlp.json").write_text(json.dumps(res, indent=1))

# ---- figure 1: R² vs number of removed probes at the main layer
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, name in zip(axes, DATASETS):
    k = 2 if name == "direction" else 1
    for src, sty, lab in (("probe", dict(color=COLORS[name]), "INLP (probe directions)"),
                          ("random", dict(color=NEUTRAL), "random directions"),
                          ("pca", dict(color=INK2, ls="--", lw=1.5), "top principal components")):
        t = [m["r2"] for m in res["main"][name][src]["test"]]
        ax.plot(np.arange(len(t)) * k, t, **sty, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(f"{name} (layer {args.layer})")
    ax.set_xlabel("dimensions removed")
    ax.set_ylabel("test R² of next probe")
    ax.set_ylim(-0.1, 1.02)
axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figures" / "inlp_curves.png", dpi=130)

# ---- figure 2: dimensions needed vs layer
fig, ax = plt.subplots(figsize=(7, 4.3))
budget = {}
for name in DATASETS:
    k = 2 if name == "direction" else 1
    n = res["sweep"][name]["n_to_0.3"]
    ax.plot(range(len(n)), [k * (x if x is not None else args.sweep_iter) for x in n], "-o",
            color=COLORS[name], label=name)
ax.set_xlabel("layer")
ax.set_ylabel("dims removed until val R² < 0.3")
ax.set_title("Redundancy by layer")
ax.text(0.99, 0.02, f"budget {args.sweep_iter} probes; points at the cap never dropped below 0.3",
        transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figures" / "inlp_by_layer.png", dpi=130)
