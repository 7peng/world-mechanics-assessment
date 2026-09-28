"""Frozen train/val/test splits, stratified by primary label value.

- Primary split: 60/20/20 of clips, stratified so every label value appears in every split.
- Value-held-out split: every 4th label value (16 of 64) is withheld entirely -> "heldout_values";
  remaining clips split 75/25 into train/val. Used to test interpolation to unseen values.
"""
import json

import numpy as np

from src.data import DATASETS, OUT, load_manifest

LABEL = {"direction": "theta_degrees", "speed": "speed_mps", "acceleration": "acceleration_mps2"}
SPLIT_FILE = OUT / "splits.json"


def make_splits(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    splits = {}
    for name in DATASETS:
        df = load_manifest(name)
        vals = np.round(df[LABEL[name]].values, 6)
        uniq = np.unique(vals)
        primary = {"train": [], "val": [], "test": []}
        for u in uniq:
            ids = rng.permutation(df.id.values[vals == u])
            n = len(ids)
            n_te, n_va = round(0.2 * n), round(0.2 * n)
            primary["test"] += ids[:n_te].tolist()
            primary["val"] += ids[n_te : n_te + n_va].tolist()
            primary["train"] += ids[n_te + n_va :].tolist()
        held = set(uniq[1::4].tolist())
        vh = {"train": [], "val": [], "heldout_values": []}
        for u in uniq:
            ids = rng.permutation(df.id.values[vals == u])
            if u in held:
                vh["heldout_values"] += ids.tolist()
            else:
                k = round(0.25 * len(ids))
                vh["val"] += ids[:k].tolist()
                vh["train"] += ids[k:].tolist()
        # ends: lowest 8 and highest 8 values withheld (extrapolation test); direction: 8 values around 0° (wrap)
        if name == "direction":
            ends = set(uniq[:4].tolist() + uniq[-4:].tolist())
        else:
            ends = set(uniq[:8].tolist() + uniq[-8:].tolist())
        ve = {"train": [], "val": [], "heldout_values": []}
        rng_e = np.random.default_rng(seed + 1000 + DATASETS.index(name))   # separate stream: keeps the other splits unchanged
        for u in uniq:
            ids = rng_e.permutation(df.id.values[vals == u])
            if u in ends:
                ve["heldout_values"] += ids.tolist()
            else:
                k = round(0.25 * len(ids))
                ve["val"] += ids[:k].tolist()
                ve["train"] += ids[k:].tolist()
        splits[name] = {"primary": {k: sorted(v) for k, v in primary.items()},
                        "value_heldout": {k: sorted(v) for k, v in vh.items()},
                        "heldout_values": sorted(held),
                        "value_heldout_ends": {k: sorted(v) for k, v in ve.items()},
                        "heldout_values_ends": sorted(ends)}
    return splits


def load_splits() -> dict:
    if not SPLIT_FILE.exists():
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(make_splits(), indent=1))
    return json.loads(SPLIT_FILE.read_text())
