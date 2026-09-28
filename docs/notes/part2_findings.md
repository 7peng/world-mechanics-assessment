# Part 2 findings: manifold (spline) steering (2026-09-27)

Layer 12, pooled activations, same splits and val-fit read-out probes as Part 1. Results in
`outputs/results/manifolds.json`, `steer_manifold.json`, `steer_manifold_paths.json`; figures `f4_manifolds.png`,
`f5_manifold_steering.png`, `supp_manifolds.png`.

## Manifold construction (`src/manifold.py`, `scripts/fit_manifolds.py`)
- PCA-64 on train clips (94% of variance); one centroid per label value (64 values, ~14 clips each).
- Curve per PCA coordinate as a function of the intrinsic coordinate: θ (radians) for direction, log speed, acceleration.
  - `interp`: cubic through the centroids, periodic for direction (the paper's construction).
  - `lsq`: least-squares fit to all train clips; Fourier harmonics for direction (periodic by construction), cubic B-spline with K interior knots otherwise. K chosen on val: 4 harmonics; K = 2 (speed), 1 (acceleration). Larger K fits centroid noise.
- Direction is a ring: atan2 of the top-2 centroid PCs recovers θ with r = 0.997 without labels; 69% of centroid variance in that plane. Speed and acceleration are open curves with one dominant axis (74%) and a hook at the low end.
- Clips scatter far off the curve: mean residual 8.7–9.7 vs curve extent 13–17 (PCA units). The manifold is a thin signal in a wide cloud.
- Val distance to curve: lsq beats raw centroids by 7–8%; withheld-value centroids lie 0.5–0.6 neighbour-spacings from the lsq curve vs 0.9–1.0 from the interpolant.
- Nearest-point decoding (manifold as decoder), val: 8.5°, 0.39 m/s, 1.15 m/s² (ridge probe: 3.5°, 0.09, 0.27).

## Steering, same-layer read-out (`steer_manifold.json`)
Methods: Part 1 clamp (N = 10 INLP probes); spline replace (paper Eq. 2: swap the PCA-64 component for s(target), keep the residual); spline replace restricted to the curve's own span; interpolating spline; chord (centroid of the nearest seen value).
Conditions: seen values (primary split, 8 targets); held-out values (fit on 48 values, steer the 16 withheld-value clips to the withheld targets); range ends (fit on the middle values, targets are the withheld extremes).

| | condition | floor | clamp | spline | span-only | interp | chord |
|---|---|---:|---:|---:|---:|---:|---:|
| direction (°) | seen | 4.7 | 4.0 | 4.5 | 9.3 | 4.9 | 4.9 |
| | held-out | 4.8 | 4.3 | 4.4 | 8.6 | 4.7 | 7.1 |
| | ends | 5.7 | 4.1 | 4.6 | 11 | 7.6 | 15.9 |
| speed (m/s) | seen | 0.117 | 0.096 | 0.124 | 0.43 | 0.129 | 0.129 |
| | held-out | 0.115 | 0.088 | 0.109 | 0.47 | 0.116 | 0.125 |
| | ends | 0.287 | 0.105 | 0.37 | 0.52 | 4.8 | 0.36 |
| accel (m/s²) | seen | 0.37 | 0.28 | 0.33 | 1.5 | 0.36 | 0.36 |
| | held-out | 0.35 | 0.23 | 0.31 | 1.6 | 0.32 | 0.34 |
| | ends | 0.81 | 0.24 | 1.03 | 1.4 | 11.3 | 1.03 |

- Interior targets (seen, held-out): spline ≈ clamp ≈ floor. Both interpolate to unseen values. Chord degrades for held-out direction (snaps to the nearest seen centroid).
- Range ends: clamp unaffected; interpolating spline extrapolates wildly (4.8 m/s, 11 m/s²); lsq spline degrades moderately. Spline failure case.
- Off-target drift (θ read-out while steering speed / acceleration; floor 5–7°): clamp 6–7°; spline replace 83–85°. The paper's replace operation overwrites the whole PCA-64 component and θ lives there too (even a PCA-2 replacement costs 39° of θ). Restricting the replacement to the curve's own span keeps θ (8°) but then barely moves speed (0.43 vs floor 0.117): the curve's span is not where the probe reads speed.
- Displacement variant (add s(target) − s(nearest point)) is poor everywhere (8°, 0.39, 1.2) and is dropped.

## Paths (`steer_manifold_paths.json`, K = 21 waypoints, 4 source→target pairs each)
- All three paths (along the curve, chord in PCA space, clamp coordinates) are monotone and near the ideal; no "teleportation". Deviation from the u-linear ideal: curve 1.4°, chord 12.5°, clamp 14.8° for direction (the chord and clamp are linear in activation space, the ideal is linear in log-speed / θ; for speed this is a log-vs-linear parameterisation difference, not a discontinuity).
- Off-manifold distance of waypoints (nearest curve point in PCA space): curve 0.003–0.006, chord 1.7–2.3, clamp 8.9–9.7. Unsteered clips are already 8.5–9.4 off the curve; the clamp leaves them there and moves only the probe coordinates.

## Downstream (edit at layer 12 added to every token, read out at 24; 64 clips × 4 targets)
- Clamp and spline fade identically: direction 4° → 85°, speed 0.12 → 1.3 m/s, acceleration 0.3 → 4.1 m/s². Neither survives propagation. The manifold does not help here.

## Comparison
- Strengths of the spline: correct geometry (the paths stay on the data manifold, the chord and clamp leave it); a ring for direction with no branch cut; interpolation to unseen values as good as the clamp; unsupervised recovery of θ from the centroid PCs.
- Limitations: replacing the PCA subspace destroys every other variable encoded there (θ when steering speed); restricting to the curve's span fixes that but loses steering power; extrapolation past the fitted range fails; no advantage downstream; needs a 1-D labelled variable to parameterise.
- Strengths of the clamp: equal or better target error everywhere, specific (θ preserved), robust at range ends, no manifold fit needed. Limitation: paths leave the data manifold by a wide margin (off-manifold 9 vs 0.005), and it reads out what probes read, not what the model uses.
- Behaviour manifold: V-JEPA has no output distribution; the paper's video-model "behavior" (softmax over distances to activation centroids) is derived from the activations themselves. Not built here; the honest analog would go through the predictor.
