"""Experiment 1: layer-wise linear probing of direction, speed, acceleration.

Protocol: ridge on z-scored mean-pooled residual stream; fit on train, alpha chosen on val,
reported on test (bootstrap 95% CI over test clips). Baselines: random-init V-JEPA 2 (all layers),
raw pixels, and degree-2 polynomial of the disk centroid trajectory.
Extra: direction probes trained on the direction set, evaluated on θ in the speed/acceleration sets.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_baseline, load_features, split_idx
from src.plotting import COLORS, INK2, NEUTRAL, plt
from src.probes import angle_deg, circ_err, fit_ridge, metrics, r2, targets

rng = np.random.default_rng(0)
BOOT = rng.integers(0, 320, size=(500, 320))  # all test sets have 320 clips


def evaluate(name, X, y, idx, log=False, **kw):
    p = fit_ridge(X[idx["train"]], y[idx["train"]], X[idx["val"]], y[idx["val"]], **kw)
    yt, yh = y[idx["test"]], p.predict(X[idx["test"]])
    m = metrics(name, yt, yh, log)
    boots = [r2(yt[b], yh[b]) for b in BOOT[:, : len(yt)] % len(yt)]
    m["r2_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    m["alpha"] = p.alpha
    m["val"] = metrics(name, y[idx["val"]], p.predict(X[idx["val"]]), log)  # for layer selection
    return m, p


res = {}
for name in DATASETS:
    df = load_manifest(name)
    idx = split_idx(name)
    res[name] = {}
    for log in ([False] if name == "direction" else [False, True]):
        y = targets(name, df, log)
        tag = "log" if log else "raw"
        r = res[name][tag] = {}
        for model in ["vjepa2", "vjepa2_random"]:
            F = load_features(name, model)
            r[model] = []
            for layer in range(F.shape[1]):
                m, p = evaluate(name, F[:, layer].astype(np.float32), y, idx, log)
                r[model].append(m)
                if name == "direction" and model == "vjepa2":
                    # transfer: θ in the other datasets' test splits
                    for other in ("speed", "acceleration"):
                        Fo = load_features(other)[:, layer].astype(np.float32)
                        io = split_idx(other)["test"]
                        th = load_manifest(other).theta_degrees.values[io]
                        m[f"transfer_{other}_circ_mae_deg"] = float(
                            circ_err(angle_deg(p.predict(Fo[io])), th).mean())
            print(f"{name}/{tag}/{model}: best test R² {max(x['r2'] for x in r[model]):.3f} "
                  f"@ layer {int(np.argmax([x['r2'] for x in r[model]]))}")
        # temporal-concat ablation: 8 time-step means concatenated (8192 dims)
        T = load_features(name, "vjepa2", "tmean")
        r["vjepa2_tconcat"] = [evaluate(name, T[:, l].reshape(len(df), -1).astype(np.float32), y, idx, log)[0]
                               for l in range(T.shape[1])]
        for b in ["pixels", "centroid_poly2"]:
            # pixels: many near-constant dims, so z-scoring would amplify noise -> center only
            kw = dict(standardize=False, alphas=np.logspace(-2, 7, 19)) if b == "pixels" else {}
            r[b] = evaluate(name, load_baseline(name, b), y, idx, log, **kw)[0]
            print(f"{name}/{tag}/{b}: test R² {r[b]['r2']:.3f}")

out = OUT / "results"
out.mkdir(parents=True, exist_ok=True)
(out / "probe_layers.json").write_text(json.dumps(res, indent=1))

# ---- figure: R² vs layer, one panel per variable
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
layers = np.arange(25)
for j, name in enumerate(DATASETS):
    r = res[name]["raw"]
    c = COLORS[name]
    for row, (key, lab) in enumerate([("r2", "test R²"),
                                      ("circ_mae_deg" if name == "direction" else "mae",
                                       "circular MAE (°)" if name == "direction" else
                                       ("MAE (m/s)" if name == "speed" else "MAE (m/s²)"))]):
        ax = axes[row, j]
        v = [m[key] for m in r["vjepa2"]]
        ax.plot(layers, v, "-o", color=c, label="V-JEPA 2 (mean pool)", zorder=3)
        if key == "r2":
            lo, hi = zip(*[m["r2_ci"] for m in r["vjepa2"]])
            ax.fill_between(layers, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(layers, [m[key] for m in r["vjepa2_tconcat"]], "--", color=c, lw=1.5,
                label="V-JEPA 2 (per-timestep concat)")
        ax.plot(layers, [m[key] for m in r["vjepa2_random"]], "-", color=NEUTRAL, lw=1.5,
                label="random-init V-JEPA 2")
        ax.axhline(r["pixels"][key], color=INK2, ls=":", lw=1.5, label="raw pixels")
        ax.axhline(r["centroid_poly2"][key], color=INK2, ls="-.", lw=1.5, label="centroid traj. (poly-2)")
        if name == "direction" and key != "r2":
            for other, ls in (("speed", (0, (1, 1))), ("acceleration", (0, (3, 1, 1, 1)))):
                ax.plot(layers, [m[f"transfer_{other}_circ_mae_deg"] for m in r["vjepa2"]],
                        color=c, ls=ls, lw=1.2, label=f"transfer → {other} set θ")
        if key == "r2":
            ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.02)
            ax.set_title(name)
        else:
            ax.set_xlabel("layer (0 = patch embedding)")
        ax.set_ylabel(lab)
        if row == 1 and j == 0 or (row == 0 and j == 1):
            ax.legend(fontsize=8, loc="best")
fig.suptitle("Layer-wise linear probes on frozen V-JEPA 2 ViT-L (test split, 95% bootstrap CI)", x=0.01, ha="left")
fig.tight_layout()
(OUT / "figures").mkdir(exist_ok=True)
fig.savefig(OUT / "figures" / "probe_layers.png", dpi=130)
