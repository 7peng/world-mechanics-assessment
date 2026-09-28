# Behavioral steering test (overnight 2026-09-28)

## Methodology
- **Behavior** = the V-JEPA 2 predictor's forecast of frames 9–16 (tokens 1024–2047) from frames 1–8. This is the model's trained task; nothing is read at the edited layer.
- **Context-only encoding.** The encoder sees frames 1–8 only, as the V-JEPA context encoder does in training. Running the encoder on the full clip (the Hugging Face default) mixes the true future into the context tokens: with that protocol, transplanting a target clip's context tokens at layer 12 moved only 16–31% of forecasts. With context-only encoding the same transplant moves 91–100% (the positive control / ceiling). All results below use context-only encoding; the earlier full-clip results (`steer_behavior.json`, `behavior_sweep_{name}.json`, `behavior_token_{name}.json`) are superseded.
- **Readout.** Ridge from the per-time-step mean of predicted tokens (4 × 1,024) to the variable, fit on clean forecasts of train clips, α on val clips. Clean test clips decode within tolerance: 94% direction (±15°), 97% speed (±0.375 m/s), 87% acceleration (±0.975 m/s²).
- **Edits** after encoder block L, then the encoder finishes and the predictor forecasts:
  - pooled subspace (Part 1), pooled spline (Part 2), both as one delta added to every token;
  - per-token subspace (shared token-level INLP), per-token spline (one curve per token position; displacement h_p += s_p(target) − s_p(û)).
- **Metric:** % of forecasts decoded within tolerance of the target; for speed/acceleration also % whose forecast direction stays within 15° of the true direction.

## Findings
1. **Steering at the paper's layers does not reach behavior at unit strength.** Edits at layers 4–16 move ≤ 5 points above no-steering, for all four methods, including transplanting the full target centroid (all 1,024 dims of the pooled mean). Effects appear only at layers 20–24 (`f6_behavior_layers.png`, `behavior_sweep_ctx_*`, `behavior_token_ctx_*`, `behavior_token_spline_*`).
2. **The token average never carries what the predictor uses.** Swapping only the token average from a donor clip moves 0% of forecasts at layers 0–20; swapping only the token pattern (tokens minus their average) moves 100% at layers 0–8, falling after layer 8. Pooled probes and pooled steering operate on the average, which is why they fail downstream (`f7_mean_vs_pattern.png`, `behavior_decompose_*`).
3. **With strength tuned on val, edits at layer 24 work** (`f8_behavior_final.png`, `behavior_final_*`). % on target (test):

   | layer 24 | direction | speed | acceleration |
   |---|---:|---:|---:|
   | no steering | 9 | 20 | 16 |
   | subspace, pooled | **94** | **62** | **93** |
   | spline, pooled | 49 | 47 | 51 |
   | subspace, per-token | 21 | 48 | 59 |
   | spline, per-token | 66 | 56 | 46 |

   Direction intact while steering speed (clean 90%): pooled subspace 80, pooled spline 52, per-token spline 83. The paper's replace operation (pooled spline) damages direction in behavior too. At layer 12 (strengths ×2–8): per-token spline is best for direction (68% vs 42%).
4. **The paper's central claim reproduces behaviorally for direction** (`f9_behavior_paths.png`, `behavior_paths_*`, 128 test clips). Turning the forecast direction by 180° in 10 steps, % of intermediate waypoints whose forecast is at the intended intermediate direction (±15°):

   | | layer 24 | layer 12 |
   |---|---:|---:|
   | straight path, pooled subspace steering (Part 1) | 4 | 7 |
   | straight path, pooled chord (paper's baseline) | 10 | 20 |
   | straight path, per-token | 7 | 16 |
   | spline path, pooled (paper) | 44 | 35 |
   | spline path, per-token | **77** | **64** |

   The straight-path forecasts keep the source direction, then jump to the target ("teleportation"). The per-token straight path and per-token spline path share the edit type, the endpoints and the subspace; only the path geometry differs (7% vs 77%). For 90° turns the straight line stays near the ring and is nearly as good (per-token 71% vs 78% at layer 24).
5. For speed and acceleration (open curves, no wrap) there is no consistent path advantage; the pooled subspace method is best or tied on target accuracy.

## Conclusions
- Probe accuracy at a layer is not evidence of control: the predictor ignores pooled edits at layers 4–16. Behavioral evaluation needs context-only encoding.
- The information the model uses early on is in the spatial/temporal pattern across tokens, not in the pooled average that probes read.
- Where the manifold matters is the circular variable: spline paths give smooth, coherent changes of forecast direction; linear paths teleport. Per-token splines are the best way to do this.
- For hitting a single target, the Part 1 subspace method (at the top of the encoder, ×2 strength) is the strongest and most specific.
