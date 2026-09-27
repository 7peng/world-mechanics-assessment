"""Input-level baseline features (no network):
  pixels:   red channel (disk channel), 8x8 average-pooled to 32x32 per frame -> [N, 16384]
  centroid: per-frame disk centroid (px) + visible-area fraction, degree-2 polynomial expansion
            (partly visible disks at the frame edge bias the centroid; the area fraction lets the
            probe account for that; absent frames have centroid 0 and area 0)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from src.data import DATASETS, OUT, load_manifest, read_video

for name in DATASETS:
    df = load_manifest(name)
    pix = []
    for p in tqdm(df.video, desc=name):
        v = read_video(p)[..., 0].astype(np.float32) / 255.0  # [16, 256, 256]
        pix.append(v.reshape(16, 32, 8, 32, 8).mean((2, 4)).reshape(-1))
    z = np.load(OUT / "baselines" / f"centroids_{name}.npz")
    vis = (z["area"] > 0).astype(np.float32)
    frac = np.clip(z["area"] / np.median(z["area"][z["area"] > 0]), 0, 1).astype(np.float32)
    cen = np.nan_to_num(z["centroid"] / 256.0) * vis[..., None]
    base = np.c_[cen.reshape(len(df), -1), frac]
    poly = PolynomialFeatures(2, include_bias=False).fit_transform(base)
    np.savez(OUT / "baselines" / f"features_{name}.npz", ids=df.id.values,
             pixels=np.stack(pix).astype(np.float16), centroid_poly2=poly.astype(np.float32))
    print(name, np.stack(pix).shape, poly.shape)
