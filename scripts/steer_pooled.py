"""Experiment 3a: multi-probe subspace steering on pooled activations (paper protocol).

Steering subspace + probes: INLP fit on the *train* split at the INLP layer.
Steered clips: *test* split only. Read-out at the same layer by evaluation probes that never saw
the steering subspace's training data:
  - "paper":  ridge fit on clean *test* activations (as in Joseph et al. App. C.12)
  - "strict": ridge fit on clean *val* activations (disjoint from both steering fit and steered clips)
Controls: random perturbation of matched norm inside a random subspace of the same dimension.
Off-target check (speed / acceleration sets, where θ varies): θ read-out before vs after steering.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.plotting import COLORS, INK2, NEUTRAL, plt
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.steer import clamp_coords, steer, target_vec

L = json.loads((OUT / "results" / "inlp.json").read_text())["layer"]
N_LIST = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
TARGETS = {"direction": np.arange(0, 360, 45.0),
           "speed": np.linspace(0.25, 4.0, 8), "acceleration": np.linspace(0.25, 10.0, 8)}
rng = np.random.default_rng(0)


def decode(name, probe, X):
    y = probe.predict(X)
    return angle_deg(y) if name == "direction" else y[:, 0]


def err(name, a, b):
    return circ_err(a, b) if name == "direction" else np.abs(a - b)


res = {"layer": L, "n_list": N_LIST}
for name in DATASETS:
    df = load_manifest(name)
    y, idx = targets(name, df), split_idx(name)
    X = load_features(name)[:, L].astype(np.float64)
    inlp = pickle.load(open(OUT / "inlp" / f"inlp_{name}_L{L}.pkl", "rb"))
    Xte = X[idx["test"]]
    zte = inlp.z(Xte)
    yt = y[idx["test"]]
    truth = angle_deg(yt) if name == "direction" else yt[:, 0]
    alpha = json.loads((OUT / "results" / "probe_layers.json").read_text())[name]["raw"]["vjepa2"][L]["alpha"]
    evals = {"paper": fit_ridge(Xte, y[idx["test"]], alpha=alpha),
             "strict": fit_ridge(X[idx["val"]], y[idx["val"]], alpha=alpha)}
    theta_probe = None
    if name != "direction":
        yth = targets("direction", df)
        theta_probe = fit_ridge(X[idx["val"]], yth[idx["val"]], alpha=alpha)
        th_true = df.theta_degrees.values[idx["test"]]
    r = res[name] = {"targets": TARGETS[name].tolist(), "clean": {}, "steer": [], "random": []}
    for k, p in evals.items():
        r["clean"][k] = float(err(name, decode(name, p, Xte), truth).mean())
    if theta_probe is not None:
        r["clean"]["theta_mae"] = float(circ_err(decode("direction", theta_probe, Xte), th_true).mean())
    to_raw = lambda z: z * inlp.sd + inlp.mu
    for N in N_LIST:
        row, rrow = {"to_target": {}, "to_truth": {}, "decoded": {}}, {"to_target": {}}
        per_t = {k: [] for k in evals}
        per_truth = {k: [] for k in evals}
        rnd = {k: [] for k in evals}
        dec_mean = {k: [] for k in evals}
        th_err = []
        for tv in TARGETS[name]:
            V, c = clamp_coords(inlp, N, target_vec(name, tv))
            zs = steer(zte, V, c)
            # matched-norm random perturbation in a random subspace of equal dimension
            Rb, _ = np.linalg.qr(rng.standard_normal((zte.shape[1], V.shape[1])))
            g = rng.standard_normal((len(zte), V.shape[1])) @ Rb.T
            g *= (np.linalg.norm(zs - zte, axis=1) / np.linalg.norm(g, axis=1))[:, None]
            Xs, Xr = to_raw(zs), to_raw(zte + g)
            for k, p in evals.items():
                d = decode(name, p, Xs)
                per_t[k].append(err(name, d, tv).mean())
                per_truth[k].append(err(name, d, truth).mean())
                rnd[k].append(err(name, decode(name, p, Xr), tv).mean())
                # direction: mean signed error wrapped to (-180, 180], added back to the target
                dec_mean[k].append(float(np.mean(d)) if name != "direction" else
                                   float(tv + ((d - tv + 180) % 360 - 180).mean()))
            if theta_probe is not None:
                th_err.append(circ_err(decode("direction", theta_probe, Xs), th_true).mean())
        for k in evals:
            row["to_target"][k] = float(np.mean(per_t[k]))
            row["to_truth"][k] = float(np.mean(per_truth[k]))
            row["decoded"][k] = dec_mean[k]
            rrow["to_target"][k] = float(np.mean(rnd[k]))
        if th_err:
            row["theta_mae"] = float(np.mean(th_err))
        r["steer"].append(row)
        r["random"].append(rrow)
        print(f"{name} N={N:3d}: to-target paper {row['to_target']['paper']:.3f} strict {row['to_target']['strict']:.3f} "
              f"| random {rrow['to_target']['strict']:.3f} | to-truth {row['to_truth']['strict']:.3f}"
              + (f" | θ MAE {row['theta_mae']:.1f}° (clean {r['clean']['theta_mae']:.1f}°)" if th_err else ""))

(OUT / "results" / "steer_pooled.json").write_text(json.dumps(res, indent=1))

# ---- figure: error to target vs number of probes
units = {"direction": "circular MAE (°)", "speed": "MAE (m/s)", "acceleration": "MAE (m/s²)"}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for j, name in enumerate(DATASETS):
    r, c = res[name], COLORS[name]
    ax = axes[0, j]
    ax.plot(N_LIST, [s["to_target"]["strict"] for s in r["steer"]], "-o", color=c, label="steered → target (strict eval probe)")
    ax.plot(N_LIST, [s["to_target"]["paper"] for s in r["steer"]], "--", color=c, lw=1.5, label="steered → target (paper eval probe)")
    ax.plot(N_LIST, [s["to_truth"]["strict"] for s in r["steer"]], ":", color=c, lw=1.5, label="steered → original label")
    ax.plot(N_LIST, [s["to_target"]["strict"] for s in r["random"]], "-", color=NEUTRAL, lw=1.5, label="matched-norm random → target")
    ax.set_xscale("log"); ax.set_xlabel("number of INLP probes in steering subspace")
    ax.set_ylabel(units[name]); ax.set_title(f"{name} (layer {L})")
    if j == 0:
        ax.legend(fontsize=8)
    ax = axes[1, j]
    tg = r["targets"]
    for N, shade in ((1, 0.35), (5, 0.6), (20, 1.0)):
        i = N_LIST.index(N)
        ax.plot(tg, r["steer"][i]["decoded"]["strict"], "-o", color=c, alpha=shade, label=f"N={N}")
    ax.plot(tg, tg, color=INK2, lw=1, ls="--", label="ideal")
    ax.set_xlabel("target"); ax.set_ylabel("mean decoded value (strict probe)")
    ax.legend(fontsize=8)
fig.suptitle("Multi-probe subspace steering on pooled activations, held-out test clips", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "figures" / "steer_pooled.png", dpi=130)
