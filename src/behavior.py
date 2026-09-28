"""Behavioral readout via the V-JEPA 2 predictor, context-only protocol.

The encoder sees frames 1-8 only (1,024 tokens, temporal positions 0-3), as the V-JEPA context encoder
does in training; running the encoder on the full clip lets layers mix the true future into the context
tokens (checked: a donor-token transplant at layer 12 then barely moves the forecast). The predictor
forecasts tokens 1024-2047 (frames 9-16) from the 1,024 context tokens.
An edit is applied to the context-encoder residual stream after block L; the encoder finishes, the final
LayerNorm is applied, and the predictor forecasts.
Readout: ridge from the forecast features (per-time-step mean of predicted tokens, 4 x 1024, or the plain
mean) to the variable, fit on clean forecasts of all train clips; never on steered data. Cached in
outputs/behavior/.
"""
import numpy as np
import torch

from src.data import OUT, read_video
from src.model import preprocess, run_to
from src.probes import fit_ridge

_M = {}


def _masks(B):
    if "ctx" not in _M:
        _M["ctx"] = torch.arange(0, 1024, device="cuda")
        _M["tgt"] = torch.arange(1024, 2048, device="cuda")
    return [_M["ctx"][None].repeat(B, 1)], [_M["tgt"][None].repeat(B, 1)]


@torch.no_grad()
def forecast_from(model, h, layer, feat="tmean"):
    """h: context residual stream after block `layer` [B, 1024, D] -> forecast features.
    feat='tmean': per-time-step mean of predicted tokens, flattened [B, 4*D]; 'mean': [B, D]."""
    for blk in model.encoder.layer[layer:]:
        h = blk(h, None)[0]
    enc = model.encoder.layernorm(h)
    B = enc.shape[0]
    cm, tm = _masks(B)
    pred = model.predictor(enc, context_mask=cm, target_mask=tm).last_hidden_state      # [B, 1024, D]
    if feat == "mean":
        return pred.mean(1).double().cpu().numpy()
    return pred.view(B, 4, 256, -1).mean(2).reshape(B, -1).double().cpu().numpy()


@torch.no_grad()
def encode_to(model, df, rows, layer):
    x = preprocess(np.stack([read_video(df.video[i])[:8] for i in rows])).cuda()
    return run_to(model, x, layer)


def clean_forecasts(model, name, df, rows, bs=8, feat="tmean"):
    path = OUT / "behavior" / f"ctx_forecast_{name}_{feat}_{len(rows)}_{int(rows[0])}_{int(rows[-1])}.npz"
    path.parent.mkdir(exist_ok=True)
    rows = np.asarray(rows)
    if path.exists():
        z = np.load(path)
        if (z["rows"] == rows).all():
            return z["F"]
    F = np.concatenate([forecast_from(model, encode_to(model, df, rows[s:s + bs], 0), 0, feat) for s in range(0, len(rows), bs)])
    np.savez(path, rows=rows, F=F)
    return F


def fit_readout(model, name, df, y, train_rows, val_rows, feat="tmean"):
    """Ridge on clean forecasts of train clips; alpha chosen on clean forecasts of val clips."""
    Ftr = clean_forecasts(model, name, df, train_rows, feat=feat)
    Fva = clean_forecasts(model, name, df, val_rows, feat=feat)
    return fit_ridge(Ftr, y[np.asarray(train_rows)], Fva, y[np.asarray(val_rows)])
