# Part 1: findings so far (2026-09-26)

All numbers are on the held-out **test** split (320 clips per dataset) unless stated otherwise.
Layers and α were chosen on **val**. Figures are in `outputs/figures/` and raw numbers in `outputs/results/`.

## Data checks
- Labels match measured centroid trajectories: about 32 px per m, and median direction error about 1°.
- The disk decodes as **orange**, not blue as `DATA.md` states. This is probably an RGB/BGR swap at render time.
- Direction set: 7.5% of clips have the disk fully leave the frame, and 12% have at least one partially visible frame. This comes from the fast velocity clips, up to 7 m/s. The speed and acceleration sets always stay in frame.
- The direction set mixes constant-velocity clips (speed 1–7 m/s) with accelerating ones (2–10 m/s², starting at rest). In the speed and acceleration sets, θ varies uniformly, which lets us check off-target effects.

## Experiment 1: layer-wise probing (`probe_layers.png`)

Probe: ridge regression on mean-pooled residual-stream features.

- **Layer 0** (patch embedding, mean-pooled) carries almost nothing: R² ≈ 0.05. Mean-pooling a linear map of pixels gives roughly the mean pixel.
- **Layer 1 already decodes everything linearly.** R² is 0.85 for direction and about 0.98 for speed and acceleration.
- **A random-init V-JEPA also reaches R² 0.84–0.93.** On this simple stimulus, one attention+MLP block plus mean-pooling exposes the kinematics.
- **Pretraining shows up as gradual refinement, visible in error rather than R²:**

  | variable | pretrained, L1 | pretrained, best | best layers | random-init plateau |
  |---|---:|---:|---|---:|
  | direction circular MAE | 12° | ≈3.1–3.5° | L13–24 | ≈10° |
  | speed MAE (m/s) | 0.11 | 0.07 | L19–21 | 0.24 |
  | acceleration MAE (m/s²) | 0.36 | 0.20 | L19–20 | 0.7 |

- **Baselines** (test R², direction / speed / acceleration):
  - raw pixels, linear: 0.52 / 0.72 / 0.63;
  - degree-2 polynomial of the centroid trajectory: 0.73 / 0.95 / 0.94.
- **Transfer:** direction probes trained on the direction set carry over to θ in the other sets.

  | layer | direction set | → speed set | → acceleration set |
  |---:|---:|---:|---:|
  | 1 | 12.4° | 11.9° | 20.1° |
  | 12 | 3.9° | 4.0° | 5.3° |
- Concatenating per-timestep means does about the same as the plain mean pool.
- **Versus the paper:** Joseph et al. find direction emerging abruptly at layer 8 (their "PEZ"). We see no abrupt emergence in R², only a gradual error decline that levels off around L10–13. Plausible reasons:
  - their stimulus is a shaded 3D sphere on a textured floor, with 8 discrete directions and 7 speeds;
  - their probe protocol is different.

## Experiment 2: iterative nullspace probing at layer 12 (`inlp_curves.png`, `inlp_by_layer.png`)

Layer choice: the earliest layer where direction val error is within 1.25× of its best. The paper's R²-jump rule is degenerate here.

- **R² falls below 0.3 after removing:**
  - direction: about 60 dimensions (30 sin/cos probes);
  - speed: about 55;
  - acceleration: about 50.
  
  All three reach R² ≈ 0 by 100–200 dimensions. The paper reports about 44 direction probes and 28 speed probes at its layer 8, so ours is the same order of magnitude.
- **Removing the same number of random directions has no effect**: R² stays at 0.98.
- **Removing the top principal components kills decoding almost immediately.** Direction falls below R² 0.1 after 4 PCs, and speed and acceleration after 13. The kinematic variables *are* the dominant variance of the pooled representation, which makes sense because nothing else varies in these clips.
- **Across layers:**
  - Layers 1–7 are extremely redundant: none of the variables drops below 0.3 within an 80-probe budget.
  - A transition at **layer 8** makes the representation markedly more compact, at about 45–60 dimensions for all three variables.
  - The dimension count then climbs slowly toward the output.
  
  This layer-8 transition coincides with the paper's PEZ, and is worth highlighting.
- There is no sawtooth. We remove sin and cos together (2 dims per step), and the paper's sawtooth was in accuracy-within-15°.

## Experiment 3a: multi-probe subspace steering, paper protocol (`steer_pooled.png`)

- **Setup:** the in-subspace coordinates of the pooled layer-12 vector are clamped (Eq. 8), and the result is read out at the same layer.
  - Subspace: INLP on the train split.
  - Steered clips: the test split.
  - Read-out probes: fit on clean val clips ("strict"), or on clean test clips ("paper").

- **Error to target** (strict probe, averaged over 8 targets):

  | N probes | direction | speed | acceleration |
  |---:|---:|---:|---:|
  | 1 | 75° | 0.87 m/s | 1.96 m/s² |
  | 3 | 14° | 0.25 | 0.58 |
  | 10 | 4.0° | 0.10 | 0.28 |
  | 20 | 4.2° | 0.09 | 0.27 |

  From 10 probes on, this is at the unsteered probe noise floor, which reproduces the paper's qualitative curve (their 20-probe result is 11.9°). Using the strict or the paper probe makes little difference.

- **Controls and side effects:**
  - The matched-norm random control does nothing (≈90° / 1.35 / 3.5).
  - Specificity: steering speed or acceleration leaves θ read-out unchanged up to about 20–50 probes. At N = 100 it degrades θ, from 5.2° to 17.6° for speed and from 7.2° to 33.8° for acceleration, because the subspace starts eating into shared dimensions.

## Experiment 3b: the same steering, propagated through blocks 13–24 (`steer_propagate.png`)

- **Intervention:** add Δ = x* − x to every layer-12 token. The pooled vector then equals x* exactly, and token structure is preserved.
- **Read-out:** probes fit on clean val clips at each downstream layer.

- **The steering does not survive.** Direction error to target, N = 20:

  | layer | 12 | 13 | 14 | 15 | 24 |
  |---|---:|---:|---:|---:|---:|
  | error to target | 4° | 25° | 45° | 73° | 85° |

  The random control gives 90°. Speed and acceleration behave the same way.

- **The network actively cancels Δ** (`diag_persistence.json`). The fraction of Δ still present along its own direction:

  | blocks after injection | direction | speed / acceleration |
  |---:|---:|---:|
  | 1 | 0.48 | 0.60 |
  | 4 | 0.10 | 0.21 |

  The residual connection would carry Δ forward unchanged, so the blocks must be writing close to −Δ back. The layer-12 probe applied downstream also fails (40° after one block).

- **Interpretation (tentative):** the pooled probe subspace is a faithful *read-out* of the layer-12 state, but not a *control* variable for later computation.
  - Later blocks appear to re-derive the kinematics from token-level evidence, meaning where the disk tokens are, and overwrite a uniform shift that contradicts it.
  - The paper never tested this, because its steering is read out at the same layer.
  - This is a key limitation to present, and a natural baseline for Part 2.

## Possible follow-ups
- Steering variants:
  - add Δ only to object tokens;
  - clamp at every layer, not just layer 12;
  - steer at later layers, such as 18–20, where fewer blocks remain to undo it;
  - fit a token-level (not pooled) INLP subspace.
- Run INLP and steering at layer 8, the paper's PEZ and our compactness transition, as a direct comparison.
- Add bootstrap CIs to the steering curves.
- Set up a value-held-out steering evaluation, where targets never appear in the train split, for comparability with Part 2.
