"""Sanity checks: frame counts, disk visibility, centroid trajectories, montages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data import DATASETS, OUT, load_manifest, read_video

fig_dir = OUT / "figures" / "sanity"
fig_dir.mkdir(parents=True, exist_ok=True)
(OUT / "baselines").mkdir(parents=True, exist_ok=True)

for name in DATASETS:
    df = load_manifest(name)
    cents, areas, shapes = [], [], set()
    for p in tqdm(df.video, desc=name):
        v = read_video(p)
        shapes.add(v.shape)
        # decodes as orange (~224,112,32) although DATA.md says blue: likely an RGB/BGR swap at render
        mask = (v[..., 0].astype(int) - v[..., 2].astype(int)) > 60
        area = mask.sum((1, 2))
        ys, xs = np.mgrid[: v.shape[1], : v.shape[2]]
        cx = (mask * xs).sum((1, 2)) / np.maximum(area, 1)
        cy = (mask * ys).sum((1, 2)) / np.maximum(area, 1)
        cents.append(np.stack([cx, cy], 1))
        areas.append(area)
    cents, areas = np.stack(cents), np.stack(areas)
    np.savez(OUT / "baselines" / f"centroids_{name}.npz", ids=df.id.values, centroid=cents, area=areas)
    full = np.median(areas)
    print(f"[{name}] shapes={shapes} median disk area={full:.0f}px "
          f"frames w/ partial disk={(areas < 0.9 * full).mean():.3%} "
          f"clips w/ any partial frame={(areas < 0.9 * full).any(1).mean():.2%} "
          f"clips w/ empty frame={(areas == 0).any(1).mean():.2%}")
    ok = (areas >= 0.9 * full).all(1)  # disk fully visible in every frame
    disp = np.linalg.norm(cents[ok, -1] - cents[ok, 0], axis=1)
    print(f"   pixel displacement (fully visible clips, n={ok.sum()}): "
          f"min={disp.min():.1f} median={np.median(disp):.1f} max={disp.max():.1f}")

    # montage: 6 clips spanning the label range, frames 0,5,10,15
    key = {"direction": "theta_degrees", "speed": "speed_mps", "acceleration": "acceleration_mps2"}[name]
    idx = df.sort_values(key).index[np.linspace(0, len(df) - 1, 6).astype(int)]
    f, ax = plt.subplots(6, 5, figsize=(10, 12))
    for r, i in enumerate(idx):
        v = read_video(df.video[i])
        for c, t in enumerate([0, 5, 10, 15]):
            ax[r, c].imshow(v[t]); ax[r, c].axis("off")
        ax[r, 4].imshow(v.max(0)); ax[r, 4].axis("off")
        ax[r, 0].set_title(f"{key}={df[key][i]:.2f}, {df.motion[i]}", fontsize=8, loc="left")
    ax[0, 4].set_title("max-proj", fontsize=8)
    f.tight_layout(); f.savefig(fig_dir / f"montage_{name}.png", dpi=80); plt.close(f)
