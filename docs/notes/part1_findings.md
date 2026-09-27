# Part 1 findings (revised 2026-09-27, after code review)

The full write-up is `docs/report/kinematics-v-jepa2.html`, built by `scripts/build_report.py`. Every number below comes from `outputs/results/*.json`.

Conventions:
- Our layer ℓ is the output of block ℓ, and layer 0 is the patch embedding. We read the paper's layer as ours − 1, but the paper doesn't state this (see "Layer indexing" below).
- "Test" means our fixed 60/20/20 split. "CV" means 5-fold stratified cross-validation.

## Data
- The labels match the measured centroid trajectories: 32.0 px/m, with the fitted direction within 1–2° of the label (median).
- The disk decodes as orange, although DATA.md says blue.
- Direction set:
  - In 113 of 1500 clips (7.5%) the disk leaves the frame entirely; in 179 (11.9%) it is at least partly cut off.
  - It mixes constant-velocity clips (1–7 m/s) with clips accelerating from rest (2–10 m/s²).
- Speed labels run 0.25–4.0 m/s; acceleration labels 0.25–10.0 m/s². θ varies uniformly in both sets.
- Distance travelled over the 0.625 s clip is fixed by the label, from 5 px (0.25 m/s) and 2 px (0.25 m/s²) up to about 77 px. Positions are quantised to whole pixels.

## Experiment 1: layer-wise probing
- **Our protocol** (ridge on z-scored mean-pooled features), test R² at layer 1: direction 0.85, speed 0.98, acceleration 0.98. CV gives 0.86 for direction.
- **Random-init V-JEPA** (HF default init, fp32 features), layer 1: 0.83 / 0.93 / 0.91.
- **Baselines:** pixels 0.52 / 0.72 / 0.63; centroid trajectory plus area fraction 0.70 / 0.94 / 0.95.
- **Direction test error by layer:** 12.4° at L1, 4.3° at L9, 3.1–3.6° from L13. The random-init model stays around 11°.
- **Speed and acceleration error:** best 0.069 m/s and 0.19 m/s² at L19.
- **Paper protocol on our data** (App. B: gradient-trained probe, lr×wd sweep, grouped 5-fold CV), R² at their layers 0 / 4 / 8:
  - direction 0.66 / 0.94 / 0.97;
  - speed 0.88 / 0.94 / 0.98;
  - acceleration 0.82 / 0.90 / 0.98.
  - The corrected 224-px run is within 0.02 of these.
  - Selecting the configuration on the reported fold (the literal reading) inflates R² by at most 0.028 compared with nested selection.
- **Versus the paper:**
  - Speed and acceleration agree: both are available early.
  - Direction does not. The paper's pooled direction curve steps from about 0.5 to 0.93 at its layer 8 (Fig. 2c; Fig. 18a shows 0.47 → 0.85). We see a smooth rise with no step.
  - App. C.5 says pooled performance "improves more gradually" only relative to per-patch probes.
- **Probe type × data** (`probe_protocol_check.json`), direction CV R² at our layer 1:

  | | ridge | paper-style gradient probe | same probe, 1000 epochs |
  |---|---:|---:|---:|
  | full set | 0.87 | 0.66 | 0.72 |
  | 96-clip paper-like subset | 0.63 | 0.31 | 0.57 |

  - Low early scores come from little data plus an undertrained, heavily regularised probe, not from the features lacking the information.
  - In every setting the rise is complete by our L6, not L9.
  - Raw-feature and standardised ridge agree once the α grid is scaled correctly.

## Experiment 2: nullspace probing
- **Our protocol at L12** (chosen on val): test R² falls below 0.3 after removing 58 dims (direction), 51 (speed) and 48 (acceleration).
  - Removing random directions leaves R² at 0.98.
  - Removing the top 4 PCs (direction) or 13 (speed, acceleration) brings it below 0.1.
- **Layer sweep** (val R² < 0.3, budget 80 probes):
  - Direction never drops at L1–3, needs 106–136 dims at L4–7, and 48–86 from L8.
  - Speed and acceleration never drop at L1–6.
  - Compactness increases sharply at L8, one block before the paper's emergence layer.
- **Paper protocol** (C.11; median of 3 seeds, each layer with its own minibatch order):
  - At their layer 8, the stop comes after 60 direction probes (120 dims), 44 speed and 26 acceleration.
  - The paper reports ≈44 and ≈28 in Fig. 22. Its Table 3 says 136 direction dims (68 probes) at the same layer, so the paper is itself inconsistent.
  - Before layer 8 our direction count is 15–34; the paper's is below 10.
  - Counts vary by ±5–10 probes across seeds.

## Experiment 3: subspace steering
- **Paper protocol** (C.12, direction, target 90°, mean of 3 random 70/30 splits), at their layer 8 read as the output of block 8:

  | N probes | 1 | 5 | 10 | 20 |
  |---|---:|---:|---:|---:|
  | error to target, 256 px | 76° | 21° | 11° | 8.4° |

  - 224 px gives 6.1° at N = 20.
  - The paper reports 11.9°. Across splits, N = 20 ranges from 3.5° to 14°.
  - The steering-probe sequence never reached R² < 0.1 within 25 probes (minimum 0.80–0.83), unlike the paper.
  - With "layer 8" read as hidden state 8 instead, steering works with fewer probes: 32° at N = 1 and 4.3° at N = 20.
- **Our protocol** (L12, 8 targets, read-out probe fit on val), error to target:

  | N probes | direction | speed | acceleration |
  |---:|---:|---:|---:|
  | 1 | 75° | 0.87 m/s | 1.96 m/s² |
  | 10 | 4.0° | 0.10 | 0.28 |

  - A random edit of the same norm (matched in raw units) does nothing: about 90° / 1.35 / 3.5.
  - The θ read-out is unchanged up to about 20 probes and degrades at 100.
- **Propagated** (Δ added to every token at L12, then blocks 13–24; read-out probes fit on val at each layer):
  - Direction error to target is 4° → 25° → 45° → 73° by L15, and 85° at L24 (random edit 90°).
  - Speed and acceleration keep part of the effect. At L24: 1.13 m/s vs 1.35 for a random edit, and 2.96 vs 3.50 m/s².
- **How much of the edit survives** (component along Δ, `diag_persistence.json`), fraction left after 1 block / at L24:

  | Δ injected | after 1 block | at L24 |
  |---|---:|---:|
  | steering edit | 0.48–0.60 | 0.07–0.10 |
  | real clip-to-clip difference | 0.55–0.56 | 0.08–0.11 |
  | random edit, same norm | 0.47–0.48 | 0.02 |

  - Speed and acceleration edits decay like a real clip difference. The direction edit decays like a random edit for three blocks, then levels off at 7%, between the random edit (2%) and a clip difference (11%).
  - So the network does not single out the steering edit for cancellation. It attenuates and rotates offsets in the pooled state in general, and probes fit on clean downstream data do not read the target.
  - The effect is the same at 5% of the edit size, checked in review.
  - The earlier diagnostic that applied the layer-12 probe to downstream layers is not meaningful: that probe fails on unsteered clips as well.

## Displacement confound
- The layer-12 acceleration probe, applied to constant-velocity clips (true acceleration 0), predicts 6.76 m/s² on average.
  - Its correlation with distance travelled is r = 0.993.
  - Its slope against the acceleration that would cover the same distance from rest is 0.97.
- So the probe reads displacement. The paper's datasets have the same structure.

## Bugs fixed in review (2026-09-27)
- **224 px (critical):** HF VJEPA2 builds its RoPE grid from `config.crop_size`, so token positions were scrambled. Now patched in `load_encoder(size=...)`, and all 224-px results were rerun.
- **fp16 feature cache:** rounding noise was 2–4% of variance for the random-init model. Features are now stored in fp32.
- **Direction "decoded mean" in the steering plots:** it came out near the target even when steering failed. Now a circular mean with its resultant length R, plus the fraction of clips within ±22.5°.
- **Headline wording:** "actively cancelled" overstated the persistence result. Random and natural baselines were added.
- **Paper-protocol stop counts:** noisy steps were shared across layers because every layer used the same seed. Fixed with a per-layer minibatch order and 3 seeds.
- **K ("probes until R² < 0.1") was reported as 25** when the threshold was never reached. It is now reported as None.
- **Layer indexing:** it had been presented as stated by the paper; it is now marked as our choice.
- **Minor:**
  - raw-ridge α grid;
  - inner splits that differed across layers;
  - random-control norm matched in z-space instead of raw units;
  - bootstrap tied to n = 320;
  - centroid bias from partly visible disks;
  - displacement printout for clips where the disk exits;
  - wrong numbers in the report: acceleration range, 8 px, 0.67 s, 7%.

## Next
- Per-patch probes (where the paper's appendix places the transition).
- A training-set size sweep.
- Motion-type classification with distance travelled matched.
- Stronger steering: disk tokens only, re-applied at every layer, or at later layers.
- More splits for our own protocol.
