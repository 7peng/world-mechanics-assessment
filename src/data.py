"""Dataset loading. Reads the supplied manifests; never writes under data/."""
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATASETS = ("direction", "speed", "acceleration")


def load_manifest(name: str) -> pd.DataFrame:
    """One row per clip: id, absolute video path, and all metadata fields."""
    base = DATA / name
    rows = []
    with open(base / "manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta = json.loads((base / r["metadata"]).read_text())
            assert meta["id"] == r["id"]
            meta["video"] = str(base / r["video"])
            meta["dataset"] = name
            rows.append(meta)
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    df["x0"], df["y0"] = zip(*df.pop("start_position_xy_m"))
    return df


def read_video(path: str) -> np.ndarray:
    """Decode to uint8 array [T, H, W, 3]."""
    with av.open(path) as f:
        return np.stack([fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)])
