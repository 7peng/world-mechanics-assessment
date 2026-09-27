"""Steering to held-out target values (Part 1 gap).

Split `value_heldout`: every 4th label value (16 of 64) is withheld entirely. INLP subspace fit on
train (48 values); read-out probe fit on val (same 48 values); steered clips are the held-out-value
clips, which no probe has seen. Targets: the 16 unseen values, and the 16 nearest seen values.
Same-layer read-out at the INLP layer.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.inlp import run_inlp
from src.probes import angle_deg, circ_err, fit_ridge, targets
from src.splits import LABEL, load_splits
from src.steer import clamp_coords, steer, target_vec

L = json.loads((OUT / "results" / "inlp.json").read_text())["layer"]
N_LIST = [1, 3, 5, 10, 20, 50]
res = {"layer": L, "n_list": N_LIST}
for name in DATASETS:
    df = load_manifest(name)
    y = targets(name, df)
    idx = split_idx(name, "value_heldout")
    held = np.array(load_splits()[name]["heldout_values"])
    seen = np.setdiff1d(np.unique(np.round(df[LABEL[name]].values, 6)), held)
    nearest_seen = np.array([seen[np.argmin(np.abs(seen - h))] for h in held])
    X = load_features(name)[:, L].astype(np.float64)
    inlp = run_inlp(name, X.astype(np.float32), y, {"train": idx["train"], "val": idx["val"]}, max(N_LIST), eval_splits=("val",))
    probe = fit_ridge(X[idx["val"]], y[idx["val"]], alpha=json.loads((OUT / "results" / "probe_layers.json").read_text())[name]["raw"]["vjepa2"][L]["alpha"])
    dec = (lambda Z: angle_deg(probe.predict(Z))) if name == "direction" else (lambda Z: probe.predict(Z)[:, 0])
    err = circ_err if name == "direction" else (lambda a, b: np.abs(a - b))
    ho = idx["heldout_values"]
    truth = angle_deg(y[ho]) if name == "direction" else y[ho][:, 0]
    r = res[name] = {"held_out_values": held.tolist(), "nearest_seen_values": nearest_seen.tolist(),
                     "probe_error_on_heldout_clips": float(err(dec(X[ho]), truth).mean()),
                     "probe_error_on_val_clips": float(err(dec(X[idx["val"]]), angle_deg(y[idx["val"]]) if name == "direction" else y[idx["val"]][:, 0]).mean()),
                     "unseen_targets": [], "seen_targets": []}
    z = inlp.z(X[ho])
    to_raw = lambda zz: zz * inlp.sd + inlp.mu
    for N in N_LIST:
        for key, tv in (("unseen_targets", held), ("seen_targets", nearest_seen)):
            e = []
            for t in tv:
                V, c = clamp_coords(inlp, N, target_vec(name, float(t)))
                e.append(err(dec(to_raw(steer(z, V, c))), float(t)).mean())
            r[key].append(float(np.mean(e)))
    print(name, f"probe on held-out-value clips: {r['probe_error_on_heldout_clips']:.3g} (val clips {r['probe_error_on_val_clips']:.3g})")
    print(name, "to unseen targets:", " ".join(f"N{N}:{v:.3g}" for N, v in zip(N_LIST, r["unseen_targets"])))
    print(name, "to seen targets:  ", " ".join(f"N{N}:{v:.3g}" for N, v in zip(N_LIST, r["seen_targets"])))
(OUT / "results" / "steer_heldout.json").write_text(json.dumps(res, indent=1))
