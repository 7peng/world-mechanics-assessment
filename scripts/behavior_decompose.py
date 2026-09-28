"""Where does the information the predictor uses live: in the average of the tokens, or in their pattern?

Context-only protocol. For each test clip, pick a donor train clip with a different label (one of the 4
steering targets, nearest value). At layer L, three transplants:
  full     all tokens from the donor
  mean     original tokens, with their average replaced by the donor's average  (h - mean(h) + mean(hd))
  pattern  donor tokens, with their average replaced by the original's average  (hd - mean(hd) + mean(h))
Metric: % of forecasts decoded within tolerance of the DONOR label, and of the ORIGINAL label.
Pooled probes and pooled steering only see the average; if the forecast follows the pattern, the average
is a readout the model does not use.
Writes outputs/results/behavior_decompose_<dataset>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.behavior import encode_to, fit_readout, forecast_from
from src.data import OUT, load_manifest
from src.features import split_idx
from src.model import load_encoder
from src.probes import angle_deg, circ_err, targets
from src.splits import LABEL

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--layers", default="0,4,8,12,16,20,24")
ap.add_argument("--n-test", type=int, default=128)
args = ap.parse_args()
name = args.dataset
LAYERS = [int(v) for v in args.layers.split(",")]
TOL = {"direction": 15.0, "speed": 0.375, "acceleration": 0.975}[name]
TARGETS = {"direction": [0.0, 90.0, 180.0, 270.0], "speed": [0.5, 1.5, 2.5, 3.5], "acceleration": [1.0, 4.0, 7.0, 10.0]}[name]
BS = 8
err = (lambda a, b: circ_err(a, b)) if name == "direction" else (lambda a, b: np.abs(a - b))
dec = (lambda p, X: angle_deg(p.predict(X))) if name == "direction" else (lambda p, X: p.predict(X)[:, 0])

model = load_encoder()
df = load_manifest(name)
lab = df[LABEL[name]].values
y = targets(name, df)
idx = split_idx(name)
tr, va, te = idx["train"], idx["val"], idx["test"][: args.n_test]
readout = fit_readout(model, name, df, y, tr, va)
rng = np.random.default_rng(0)
# donor per test clip: the target value farthest from the clip's own label, nearest train clip to it
dist = (lambda a, b: circ_err(a, b)) if name == "direction" else (lambda a, b: np.abs(a - b))
trv = np.round(lab[tr], 6)
donors = []
for i in te:
    t = max(TARGETS, key=lambda v: dist(lab[i], v))
    cand = tr[np.argsort(dist(trv, t))[:10]]
    donors.append(rng.choice(cand))
donors = np.array(donors)
res = {"dataset": name, "tolerance": TOL, "layers": {}}
for L in LAYERS:
    got = {k: [] for k in ("full", "mean", "pattern")}
    for s in range(0, len(te), BS):
        h = encode_to(model, df, te[s:s + BS], L)
        hd = encode_to(model, df, donors[s:s + BS], L)
        m, md = h.mean(1, keepdim=True), hd.mean(1, keepdim=True)
        for k, hh in (("full", hd), ("mean", h - m + md), ("pattern", hd - md + m)):
            got[k].append(dec(readout, forecast_from(model, hh, L)))
    r = res["layers"][str(L)] = {}
    for k, v in got.items():
        v = np.concatenate(v)
        r[k] = {"follows_donor_pct": float((err(v, lab[donors]) <= TOL).mean() * 100),
                "keeps_original_pct": float((err(v, lab[te]) <= TOL).mean() * 100)}
    print(f"L{L}", {k: (round(v['follows_donor_pct'], 1), round(v['keeps_original_pct'], 1)) for k, v in r.items()}, flush=True)
    (OUT / "results" / f"behavior_decompose_{name}.json").write_text(json.dumps(res, indent=1))
