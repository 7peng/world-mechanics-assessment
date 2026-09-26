# Part 1 plan — probing, nullspace, subspace steering

## Scope

Part 1 has three experiments, all run on the frozen V-JEPA 2 ViT-L/16 encoder
(24 blocks, width 1024; a 16×256×256 clip becomes 8×16×16 = 2048 tokens):

1. **Layer-wise probing**: probe direction, speed and acceleration at all 24 layers.
2. **Iterative nullspace probing (INLP)**: at one chosen layer, see how many
   dimensions each variable occupies and how redundantly.
3. **Multi-probe subspace steering**: intervene in the INLP subspace and
   evaluate the effect on held-out clips.

It ends in results and figures that feed the ~15-minute presentation. Part 2
(splines) reuses the same features, splits and steering harness.

## Data facts (checked 2026-09-26)

| set | clips | primary label | nuisance variables |
|---|---:|---|---|
| direction | 1500 | θ: 64 values, 5.625° apart, ~23 clips each | half constant velocity (≈7 speeds), half accelerating from rest (≈5 accelerations); random start position |
| speed | 1536 | 64 speeds from 0.25 to 4.0 m/s, 24 clips each | θ random (64 values); acceleration = 0 |
| acceleration | 1536 | 64 accelerations from 0.25 to 9.85 m/s², 24 clips each | θ random; starts at rest |

Implications:
- Speed and acceleration are **confounded with total displacement** over 16
  frames, so a probe can succeed just by reading where the disk ends up. A pixel
  or position baseline is needed to interpret the curves.
- θ also varies in the speed and acceleration sets, which allows a
  cross-dataset direction check.

## Step 0 — environment and setup

- Create a fresh env for this project, because the old `cnc` env's python binary
  is broken. Needs torch, transformers (with `VJEPA2Model`), scikit-learn,
  PyAV, matplotlib.
- Read both papers closely and pin down Joseph et al.'s exact choices: probe
  type, pooling, the layer they select for INLP, and how steering is applied and
  read out. Where the paper is ambiguous, document the choice made.
- Sanity-check the videos: montages of example clips, whether the disk stays in
  frame at high speed/acceleration, and whether decoded frame counts are correct.

## Step 1 — feature extraction (`src/extract.py`)

- Use the HF `facebook/vjepa2-vitl-fpc64-256` checkpoint and its official
  video processor (resize and ImageNet normalization). Cross-check a few clips
  against Meta's `vjepa2_vit_large` implementation.
- Hook the output of every block (plus the patch embedding as "layer 0").
- Pooling:
  - **primary**: mean over all tokens → `[N, 25, 1024]`;
  - **also cached**: spatial mean per time step → `[N, 25, 8, 1024]`, so that
    time-resolved probes are possible.
  - Tokens are kept only at the chosen steering layer, generated on the fly.
- Store as fp16 under `outputs/features/` (≈2 GB total). Forward passes take
  minutes on one L40S.
- Baselines:
  - raw-pixel probe (downsampled frames);
  - hand-crafted disk-trajectory features, such as centroid per frame;
  - a randomly initialized V-JEPA.

## Step 2 — splits (`src/splits.py`, frozen to JSON)

Each dataset is split independently, stratified by label value:
**60% train / 20% val / 20% test**, fixed seed.

- Train: fit probes, INLP directions and steering subspaces.
- Val: choose ridge α, the INLP layer and the steering strength.
- Test: touched only for final numbers.

A second **value-held-out split** (every 4th label value withheld) tests
interpolation to unseen magnitudes and angles. Part 2 needs it anyway.

## Step 3 — layer-wise probing

- **Probe**: standardize features, then ridge regression, with α chosen on val.
  Direction targets are (sin θ, cos θ); speed and acceleration probes are fit on
  both raw and log scale.
- **Metrics**: R² and MAE; for direction, circular MAE and R² on sin/cos.
  Bootstrap CIs over test clips.
- **Plot**: metric vs layer, all three variables, with the baselines as
  horizontal lines.
- **Extras**:
  - nonlinear (MLP) probe at a few layers, to separate "not present" from "not
    linear";
  - direction probed across sets (train on the direction set, test on the θ in
    the speed and acceleration sets);
  - speed ↔ acceleration cross-prediction, to expose the displacement confound.

## Step 4 — iterative nullspace probing

At the chosen layer (the earliest one where each variable plateaus, chosen on
val):

1. Fit a ridge probe on train.
2. Record val R².
3. Project out the probe's weight directions: 1 per step for scalars, 2 for
   sin/cos.
4. Refit, for about 100–200 iterations.

- **Plot**: R² vs number of dimensions removed.
- **Controls**: removing the same number of random directions, and removing
  the top principal components.
- **Read-out**:
  - a sharp drop means the variable sits in a compact subspace;
  - slow decay means it is redundant or distributed.
  - The subspace of the first k directions is saved for Step 5.

## Step 5 — multi-probe subspace steering

- **Subspace**: the orthonormal basis U (k dims) from INLP on train.
- **Intervention** at layer L, applied to every token:
  h ← h + U(Uᵀ(μ_target − μ_current)), where μ_target is the train-set class
  mean projected into U. This follows the paper's recipe where it differs.
- **Readout**:
  - run the remaining blocks and decode with independent probes fit on
    unsteered train data at later layers, up to the final layer;
  - steered clips come only from **test**, and targets include values from the
    value-held-out split.
- **Metrics**:
  - dose-response (target value vs decoded value, slope and R²) and success
    rate;
  - specificity: steering speed should leave the direction readout unchanged,
    and vice versa;
  - ablations over k and strength α (chosen on val), plus a random-subspace
    control of the same k.
- Direction steering is done in angle space via the sin/cos plane, ahead of
  Part 2's circular handling.

## Step 6 — write-up

- Figures in `outputs/figures/`, key numbers in `outputs/results/*.json`.
- Short findings notes in `docs/`, which become the slide outline.

## Repository layout

```
src/        extract.py, splits.py, probes.py, inlp.py, steer.py, baselines.py
scripts/    one runner per experiment
docs/       TASK.md, DATA.md, PLAN.md, notes
data/       supplied data, read-only, gitignored
outputs/    features, probes, figures, results (gitignored)
```

## Risks and open questions

- The exact steering protocol in Joseph et al. needs confirming from the paper
  (Step 0).
- The displacement confound could make "speed" and "acceleration" look trivially
  decodable; the baselines are what separate the two.
- With only ~23 clips per direction value, class-mean targets are noisy, so
  pooled regression-based targets may be better.
