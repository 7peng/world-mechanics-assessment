"""Per-patch features for per-patch probing (Joseph et al. App. C.5 / Fig. 18).

For each spatial patch (16x16 grid), the residual-stream token averaged over the 8 temporal
positions, at a subset of layers. Saves outputs/features/vjepa2_patch/<dataset>_L<l>.npy as
float16 [N, 256, 1024], in manifest order.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tqdm import tqdm

from src.data import OUT, load_manifest, read_video
from src.model import H_TOK, T_TOK, W_TOK, hidden_states, load_encoder, preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="direction")
ap.add_argument("--layers", default="1,2,4,6,8,9,10,12,16,24")
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
layers = [int(x) for x in args.layers.split(",")]
out_dir = OUT / "features" / "vjepa2_patch"
out_dir.mkdir(parents=True, exist_ok=True)
model = load_encoder()
df = load_manifest(args.dataset)
buf = {l: np.zeros((len(df), H_TOK * W_TOK, 1024), np.float16) for l in layers}
for i in tqdm(range(0, len(df), args.batch), desc=args.dataset):
    vids = np.stack([read_video(p) for p in df.video[i:i + args.batch]])
    hs = hidden_states(model, preprocess(vids).cuda())
    for l in layers:
        h = hs[l].view(len(vids), T_TOK, H_TOK * W_TOK, 1024).mean(1)   # time-averaged per patch
        buf[l][i:i + len(vids)] = h.half().cpu().numpy()
for l in layers:
    np.save(out_dir / f"{args.dataset}_L{l}.npy", buf[l])
np.save(out_dir / f"{args.dataset}_ids.npy", df.id.values)
