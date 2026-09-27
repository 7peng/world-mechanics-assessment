"""Frozen V-JEPA 2 ViT-L/16 (256px) encoder and preprocessing.

Preprocessing: clips are already 256x256x16, so we skip the HF processor's default
resize-to-292 + center-crop (which would crop ~6% off each border and can cut the disk)
and only rescale to [0,1] and apply ImageNet mean/std normalization.

Layer indexing: hidden_states[0] is the patch-embedding output ("layer 0"),
hidden_states[i] (i=1..24) is the residual stream after transformer block i.
Tokens are ordered (t, h, w) with t=8 tubelets (2 frames each), h=w=16.
"""
import numpy as np
import torch
from transformers import VJEPA2Config, VJEPA2Model

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
N_LAYERS = 24
T_TOK, H_TOK, W_TOK = 8, 16, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load_encoder(device="cuda", random_init=False, seed=0, size=256) -> VJEPA2Model:
    """size != 256: HF VJEPA2 derives the RoPE token grid from config.crop_size (16x16 at 256px),
    not from the input, so a 224px input (14x14 patches) would get scrambled positions. We set the
    grid on every encoder attention module to match the actual input."""
    if random_init:
        torch.manual_seed(seed)
        model = VJEPA2Model(VJEPA2Config.from_pretrained(MODEL_ID))
    else:
        model = VJEPA2Model.from_pretrained(MODEL_ID, dtype=torch.float32)
    model = model.to(device).eval()
    if size != 256:
        for blk in model.encoder.layer:
            blk.attention.grid_size = size // model.config.patch_size
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def preprocess(videos: np.ndarray) -> torch.Tensor:
    """uint8 [B, T, H, W, 3] -> normalized float [B, T, 3, H, W]."""
    x = torch.from_numpy(videos).permute(0, 1, 4, 2, 3).float() / 255.0
    return (x - MEAN) / STD


@torch.no_grad()
def hidden_states(model: VJEPA2Model, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Residual stream at all 25 depths, each [B, 2048, 1024]."""
    out = model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
    return out.hidden_states


@torch.no_grad()
def run_to(model: VJEPA2Model, x: torch.Tensor, layer: int) -> torch.Tensor:
    """Residual stream after block `layer` (0 = embeddings), [B, 2048, 1024]."""
    h = model.encoder.embeddings(x)
    for blk in model.encoder.layer[:layer]:
        h = blk(h, None)[0]
    return h


@torch.no_grad()
def run_from(model: VJEPA2Model, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Continue from the residual stream after block `layer`; return token-mean-pooled
    residual stream at layers layer..24 as [B, 25 - layer, 1024]."""
    pooled = [h.mean(1)]
    for blk in model.encoder.layer[layer:]:
        h = blk(h, None)[0]
        pooled.append(h.mean(1))
    return torch.stack(pooled, 1)
