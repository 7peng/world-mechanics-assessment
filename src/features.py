"""Load cached features aligned with manifest order and split indices."""
import numpy as np

from src.data import OUT, load_manifest
from src.splits import load_splits


def load_features(name: str, model: str = "vjepa2", key: str = "mean") -> np.ndarray:
    z = np.load(OUT / "features" / model / f"{name}.npz")
    assert (z["ids"] == load_manifest(name).id.values).all()
    return z[key]


def load_baseline(name: str, key: str) -> np.ndarray:
    z = np.load(OUT / "baselines" / f"features_{name}.npz")
    return z[key].astype(np.float32)


def split_idx(name: str, split: str = "primary") -> dict[str, np.ndarray]:
    """Row indices (into manifest order) for each part of a split."""
    ids = load_manifest(name).id.values
    pos = {i: k for k, i in enumerate(ids)}
    return {k: np.array([pos[i] for i in v]) for k, v in load_splits()[name][split].items()}
