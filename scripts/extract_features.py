"""Extract pooled residual-stream features at every layer for all clips.

Saves outputs/features/<model>/<dataset>.npz with
  mean  [N, 25, 1024]    mean over all space-time tokens (2048 at 256px, 1568 at 224px)
  tmean [N, 25, 8, 1024] spatial mean per temporal token (tubelet)
both float16, plus clip ids.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tqdm import tqdm

from src.data import DATASETS, OUT, load_manifest, read_video
from src.model import T_TOK, hidden_states, load_encoder, preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--random-init", action="store_true")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--size", type=int, default=256, help="224 = input size stated in Joseph et al. App. C.6")
args = ap.parse_args()

tag = ("vjepa2_random" if args.random_init else "vjepa2") + ("" if args.size == 256 else f"_{args.size}")
out_dir = OUT / "features" / tag
out_dir.mkdir(parents=True, exist_ok=True)
model = load_encoder(random_init=args.random_init)

for name in DATASETS:
    df = load_manifest(name)
    means, tmeans = [], []
    for i in tqdm(range(0, len(df), args.batch), desc=f"{tag}/{name}"):
        vids = np.stack([read_video(p) for p in df.video[i : i + args.batch]])
        x = preprocess(vids)
        if args.size != 256:  # bicubic antialiased resize of the full frame (no crop)
            B, T, C, H, W = x.shape
            x = torch.nn.functional.interpolate(x.view(B * T, C, H, W), size=(args.size, args.size), mode="bicubic",
                                                antialias=True, align_corners=False).view(B, T, C, args.size, args.size)
        hs = torch.stack(hidden_states(model, x.cuda()), 1)  # [B, 25, N_tokens, D]
        B, L, N, D = hs.shape
        means.append(hs.mean(2).half().cpu())
        tmeans.append(hs.view(B, L, T_TOK, N // T_TOK, D).mean(3).half().cpu())
    np.savez(out_dir / f"{name}.npz", ids=df.id.values,
             mean=torch.cat(means).numpy(), tmean=torch.cat(tmeans).numpy())
