# world-mechanics

How does a frozen V-JEPA 2 video encoder represent direction, speed and acceleration?
This repo holds the probing, nullspace and steering experiments; see `docs/TASK.md` for the brief
and `docs/PLAN.md` for the plan.

## Setup

```bash
uv sync                      # python 3.12, torch (CUDA), transformers>=4.53
```

The supplied dataset lives in `data/`, which is read-only and gitignored. Everything derived goes to `outputs/`, which is also gitignored.

## Pipeline (Part 1)

```bash
uv run python scripts/check_videos.py                 # label checks, centroid trajectories, montages
uv run python scripts/extract_features.py             # pooled residual stream, all 25 depths
uv run python scripts/extract_features.py --random-init
uv run python scripts/extract_features.py --size 224  # RoPE grid patched to 14x14 (see src/model.py)
uv run python scripts/baseline_features.py            # pixel + centroid-trajectory baselines
uv run python scripts/probe_layers.py                 # Exp 1: layer-wise probes (our protocol)
uv run python scripts/run_inlp.py                     # Exp 2: iterative nullspace probing
uv run python scripts/steer_pooled.py                 # Exp 3a: subspace steering, same-layer read-out
uv run python scripts/steer_propagate.py --dataset direction   # Exp 3b (also speed, acceleration)
uv run python scripts/plot_steer_propagate.py
uv run python scripts/diag_persistence.py             # how much of the steering edit survives, with baselines
uv run python scripts/probe_protocol_check.py         # probe type x data regime
uv run python scripts/paper_protocol.py --features vjepa2       # Joseph et al. protocol, 256 px
uv run python scripts/paper_protocol.py --features vjepa2_224   # same, 224 px
uv run python scripts/sanity_checks.py                # label/trajectory checks, displacement confound, CV
uv run python scripts/build_report.py                 # docs/report/kinematics-v-jepa2.html
```

## Key conventions

- **Model:** `facebook/vjepa2-vitl-fpc64-256`, frozen. Frames are rescaled to [0,1] and ImageNet-normalized. There is no resize or crop, because the clips are already 256².
- **224 px:** the HF implementation takes the RoPE token grid from `config.crop_size`, not from the input. `load_encoder(size=224)` sets it to 14×14 on every attention layer; without that fix, token positions are scrambled.
- **Layer ℓ:** the residual stream after block ℓ, taken before the final LayerNorm. Layer 0 is the patch-embedding output.
- **Pooling:** the mean over all 8×16×16 space-time tokens. We also cache per-timestep means.
- **Splits:** `outputs/splits.json`, stratified 60/20/20 by label value (seed 0).
  - **Train** fits probes, INLP directions and steering subspaces.
  - **Val** selects ridge α and layers.
  - **Test** is used only for reported numbers.
- **Data quirk:** the disk decodes as orange (≈RGB 224,112,32), although `DATA.md` says blue. This is likely a channel swap at render time. Frames are fed to the model as decoded.
