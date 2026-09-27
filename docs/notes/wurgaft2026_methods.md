# Wurgaft et al. 2026, "Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior": methods digest

- arXiv 2605.05115 (Goodfire, with Stanford, UCL, Northeastern, Harvard and Technion). The source was read from the arXiv e-print (`arxiv.tex`, 876 lines).
- Code: https://github.com/goodfire-ai/causalab/tree/manifold_steering (commit `1b6f43a`, 2026-05-06). Paths below are relative to `causalab/`.
- Blog: https://www.goodfire.ai/research/manifold-steering ("Steering Along Manifolds to Control Neural Networks", 2026-05-07). It covers the weekdays case only and adds no new methods. An earlier post in the same series ("The World Inside Neural Networks") has the mountain-car demo.
- Local copies: `/tmp/claude-471013/.../scratchpad/papers/{src,causalab}` (the scratchpad may be cleaned up).

**Labels.** **[P]** means the paper says it. **[C]** means it is in the released code but not in the paper. **[I]** is my inference or recommendation.

---

## 1. Core idea and claims

- **[P]** Fit two manifolds on unintervened data:
  - an **activation manifold** M_h through class-conditional activation centroids;
  - a **behavior manifold** M_y through class-conditional output distributions, placed in Hellinger (√p) coordinates.
- **[P]** Claim 1, isometry: geodesic (arc-length) distances on M_h correlate with those on M_y. Euclidean chord distances in activation space correlate worse (§2.3, Figs 2–3).
- **[P]** Claim 2, M_h → M_y: *manifold steering* interpolates in the spline's intrinsic coordinate and decodes back to activation space. Its output trajectories stay close to M_y, meaning they are "natural". Linear interpolation "cuts through off-manifold regions" and shows **"teleportation"**: "mass jumps between non-adjacent concepts as the straight line cuts through the manifold's interior" (§3.2, Fig 4).
- **[P]** Claim 3, M_y → M_h (the "pullback"): optimize activation paths so that their outputs follow M_y, and the recovered paths trace M_h (§3.3, Fig 5).
- **[P]** Framing (§3.4, Def. 1): steering is choosing a Riemannian metric on activation space. There are three:
  - G_I = I (linear);
  - G_E = (α e^{−E(h)} + β)^{-1} I, a density metric whose geodesics follow M_h;
  - G_F = J_Fᵀ g_y J_F + εI, the pullback of the behavior metric.
  - Key quote: "This recasts the core problem of steering from finding the right *direction* to finding the right *geometry*."
- **[P]** Note on G_E: it is a *motivation* only. **No density model or energy is ever fit in activation space.** Manifold steering is always implemented as spline interpolation (Eq. 2).
- **[P]** Domains:
  - Llama 3.1 8B, layer 28, on four tasks: weekdays and months (cyclic), letters and ages (sequential).
  - In-context-learned (ICLR) 5×5 grid and 9×9 cylinder graphs (2-D).
  - A Mountain-Car recurrent world model with a conv encoder and 64-d latent (§5). This is the case closest to the planned V-JEPA setup.

## 2. Manifold construction

### 2.1 Language-model tasks (§2.2, App. A.3)

- **Which activations [P].**
  - Residual stream at the **last token** (the answer position), layer 28, bf16 (App. A.2).
  - One vector per prompt. There is no per-token manifold.
- **Dimensionality reduction [P].** "we first … transform them into a 64-dimensional subspace obtained via PCA over the activations h(x) across all prompts in the task. The manifold lives entirely in the 64-dimensional PCA subspace; the orthogonal complement is preserved during all subsequent interventions" (App. A.3).
  - The PCA is unsupervised, fit on all prompts, and uses k = 64 for every task.
  - **[C]** `configs/analysis/subspace.yaml` has `method: pca, k_features: 64`. It also offers `das`, `dbm` and `boundless` as alternative subspace finders.
- **Centroids [P].** "concept centroids c_i as the mean of the projected activations across all prompts whose ground-truth result equals the i-th concept value."
  - Centroids are grouped by the **answer** value and marginalize over the entity and the increment k.
- **Spline type [P].** "a one-dimensional cubic spline (Reinsch 1967): a natural cubic spline (with vanishing second derivatives at the endpoints) for the sequential tasks (letters, ages), and a periodic cubic spline for the cyclic tasks (weekdays, months) so that the curve closes smoothly."
  - It is an **exact interpolant**: "In every case the spline interpolates the centroids exactly, so M_h passes through every c_i."
  - One spline per ambient (PCA) coordinate, all sharing the knot vector.
  - **[C]** `methods/spline/cubic.py` implements the Reinsch system with an optional smoothing penalty λ∫f''². The paper uses λ = 0 (`smoothness: 0.0` in `activation_manifold.yaml`).
  - **[C]** With natural boundary conditions the spline **extrapolates linearly** outside the knot range.
- **Parameterization (intrinsic coordinate) [P].**
  - Sequential tasks: "we use the ground-truth ordinal index of each concept as its intrinsic coordinate." This is **not** arc length, and ages use the integer value itself.
  - Cyclic tasks: "the centroids form a near-circular loop in the top two principal components …, so we instead derive the intrinsic coordinate θ = atan2(PC2, PC1) in an unsupervised manner."
  - **[C]** Knots are normalized to [0,1] (`(u − min)/range`), or to `u/period` for periodic dimensions (`manifold.py`).
  - **[C]** The `intrinsic_mode: pca` mode auto-detects periodic pairs of principal components. A pair qualifies when the two eigenvalues are within 45% of each other and each carries ≥ 10% of the variance (`builders.detect_periodic_dims`); the pair is then remapped to an angle. For sequential tasks the configs set `intrinsic_mode: parameter` (the ground-truth value).
  - **[C]** An optional `chord_length` reparameterization exists (1-D only) but is off by default.
  - **[C]** In `parameter` mode the 64 PCA coordinates are **z-scored per dimension** before fitting and un-scored after decoding. In `pca` mode they are not. The paper does not mention this.
- **Closed / circular variables [P].**
  - A periodic cubic spline (cyclic tridiagonal Reinsch system) on θ.
  - For the 2-D cylinder, a thin-plate spline (TPS) with "ghost points" duplicated at ±1 period, with the linear-in-θ polynomial term dropped.
  - **[C]** The path between two angles takes the **shortest arc**: `delta = ((delta + P/2) % P) − P/2` in `path_mode._build_geodesic_path`.
- **2-D (ICLR) tasks [P].** TPS with kernel r² log r. Ground-truth graph coordinates serve as intrinsic coordinates, with one centroid per node from last-token activations.

### 2.2 Behavior manifold M_y (App. A.4)

- **[P]** Build the output distribution p(x):
  - softmax over the full vocabulary;
  - sum each concept's token variants (e.g. " Monday", "Monday", "monday");
  - put all remaining mass in an "other" bin, giving |Z|+1 classes.
- **[P]** Behavior centroids are the mean output distribution per answer value, mapped through b ↦ √b onto the unit sphere.
- **[P]** The spline is fit in the **tangent plane** at the normalized mean base point b\*:
  - log-map the centroids, fit the spline on the tangent vectors, then exp-map back.
  - Result: M_y stays on the sphere, and squaring a decoded point gives a valid distribution.
  - It uses the same spline family and intrinsic coordinates as M_h, again with no smoothing.

### 2.3 Mountain Car, the vision world model (§5, App. B.1)

- **[P]** Setup:
  - 100 rollouts under a mixed stochastic policy. Activations are the **encoder output** v_t = LayerNorm(f_enc(x_t)) ∈ R^64.
  - The paper mentions no PCA; with n = 64 the spline lives in the full latent.
  - Position range split into **B = 100 bins**; mean encoder output per occupied bin.
- **[P]** Spline fit: "fit a smoothing spline γ_M : [0,1] → M through these means (one univariate spline per coordinate, weighted by the square root of bin counts to regularize sparse regions)."
  - It is parameterized by **position**, normalized to [0,1].
  - Unlike the LM case, this spline **smooths**; the smoothing strength is not reported.
- **[P]** A ridge probe on z_t gets R² ≈ 0.95 for position and ≈ 0.90 for velocity.
- **[P]** The "behavior" is **not the model's decoder**. It is a nearest-centroid softmax readout built from M_h itself (Eq. 7/8):
  - F(z) = softmax(−‖z − μ_b‖₂ / τ), with τ = 0.5 and μ_b = γ_M(p_b) for B = 128 evenly spaced positions.
  - M_y is a 1-D smoothing spline through √E[F(z) | bin].
  - **[I]** This readout is partly circular: M_y is defined through distances to points *on* M_h, so M_h → M_y agreement is built in to a degree. The decoded frames (Fig 7c) are the only readout independent of the manifold, and they are qualitative.
- **[P]** Topology: the encoder manifold for position (a sequential variable) is a **closed** curve in PCA view, because "visually distinctive states at the wall, p ≈ −1.2, and goal, p ≈ 0.4, are mapped to neighboring activations."
  - Linear chord distance correlates with behavior arc length at only r = 0.06, against r = 0.996 for arc length on M_h.

## 3. Steering procedure

- **[P] The intervention is replacement, not addition** (§3.1): "replacing the model's activation at a chosen layer with a target activation, and continuing the forward pass." The steered output is written p_{h←h\*}(x).
- **[P] Paths** (Eqs. 1–2):
  - linear: π_lin(t) = (1−t) h\*₀ + t h\*₁;
  - manifold: π_m(t) = s((1−t) u₀ + t u₁), with u_i = s⁻¹(h\*_i).
  - s is the spline, mapping intrinsic coordinate → activation. The steering "strength" is just t ∈ [0,1]; **there is no α scale**.
- **[P] Endpoints are always concept centroids** ("For each pair of concept values (z_a, z_b), we steer the model from the centroid c_a to the centroid c_b"). So s⁻¹ of an endpoint is simply its known knot coordinate.
  - The paper never steers "from this input's own activation". The base prompt only supplies the off-subspace residual and the rest of the context.
  - **[C]** For arbitrary h, `SplineManifold.encode` defaults to the **nearest centroid's knot** (`encode_mode: nearest_centroid`). The alternative `encode_to_nearest_point` is a Gauss–Newton projection onto the curve that also returns the residual.
- **[P] What gets replaced** (App. A.5). This is critical, and the two methods differ:
  - **Manifold:** "decode π(t) onto M_h in the 64-dimensional PCA subspace, lift it back to the residual-stream basis via the PCA inverse, and combine it with the prompt's unchanged off-subspace residual — so the steered activation differs from the base only in its top-64 PCA components."
  - Equivalently: h' = h − P Pᵀ(h − μ) + P s(u(t)) (+ μ as appropriate), with P the 64 principal components.
  - **Linear:** "c_a, c_b are the raw activation centroids in the full residual stream … the entire residual-stream activation is replaced by π(t)."
  - **[I] Confound:** linear steering discards the prompt's off-subspace content and manifold steering keeps it. Part of the linear baseline's "other"-token mass may come from this, not from the geometry.
  - **[C]** The code also has a `linear_subspace` mode: a straight line in the 64-d PCA subspace with the residual preserved, which is the fair control. Results for it are **not reported** in the paper.
  - **[C]** There is also an `oversteer` option (extrapolate past t = 1), off by default.
- **[P] Where:**
  - LLM: residual stream, last token, layer 28 (layer 70 for 70B).
  - Mountain Car: the encoder latent, followed by the GRU and decoder.
- **[P] Sampling:**
  - K = 50 waypoints (K = 20 for Mountain Car).
  - Up to 50 random ordered value pairs per task (all pairs when W(W−1) < 50).
  - 16 fixed base prompts (5 for ICLR) drawn from the task distribution with **mixed ground-truth answers**, reused for every pair.
  - The trajectory is the pointwise mean over base prompts.
- **[P] Baselines:** linear / diff-in-means interpolation only. There are no additive steering vectors with a tuned α, no probe-based (multi-direction) steering, and no DAS.
- **[P] Pullback** (App. A.6):
  - Target: K = 20 points along the M_y geodesic between the two behavior centroids.
  - Path: a natural cubic spline through 10 control vectors, restricted to the **first 32 of the 64 PCs**; the remaining PCs and the residual stay at base values.
  - Loss: Σ_t d_H²(p_{h←π(t)}(x_n), p̂_t), averaged over 16 fresh prompts conditioned on z_a.
  - Optimizer: L-BFGS with strong-Wolfe line search, 50 outer × 5 inner steps, initialized at the chord.
  - A norm regularizer (weight 5e-4 to 1e-3) prevents "a high-norm shortcut basin".
  - Mountain Car version: K = 30, 30 pairs. The target is a "conformal" geodesic on the simplex with cost exp(α·d_H(p, M_y)), swept over α.

## 4. Evaluation

- **Naturalness [P]** (Eq. 3, App. A.7):
  - E_BC = Σ over the K waypoints of D_BC(γ(t), M_y), where D_BC = −log Σ√(γ_i q_i) and q is the **closest point on M_y**.
  - Averaged over base prompts to one scalar per pair.
  - Reported as mean ± SE over pairs, with paired t-tests.
- **Isometry [P]** (App. A.4):
  - Arc length on M_h is the sum of Euclidean steps over 150 intrinsic sub-intervals in the 64-d PCA space. M_y uses Hellinger distance.
  - Report the Pearson r over upper-triangular pairs. Centroids are augmented with K interior points per centroid pair, and same-geodesic pairs are excluded.
  - MDS embeddings are shown for visualization.
- **Pullback R² [P]** (App. A.8):
  - Project both paths onto the SVD basis capturing 99% of the manifold path.
  - Residual = closest-point distance to the manifold path.
  - Baseline = the chord in the 64-d PCA space.
- **Mountain Car [P]:**
  - decoded frames (qualitative);
  - spread of the position distribution (qualitative);
  - mean Euclidean distance of the path to M_h: chord 2.22, manifold 0.20, pullback 0.29.
- **Qualitative [P]:** "smooth and ordered" transitions versus "teleportation", shown as stacked probability plots per pair (Fig 4).
- **[C] Extra metric:** `methods/scores/coherence.py` measures the mean and worst-case probability on concept tokens along the path. It is not in the paper.
- **Held-out protocol [P, by omission].** There is **none**:
  - manifolds are fit on all prompts of the task;
  - steering prompts come from the same template distribution;
  - no values are held out from the spline fit, and there are no train/test splits of inputs.
  - Steering endpoints are the fitted centroids themselves, so reaching the endpoint is guaranteed by construction.
  - **No target-accuracy or dose-response metric is reported.** Success means naturalness (E_BC) plus qualitative smoothness.
  - The ages config (`n_train: 1000, n_test: 1000`) implies a split exists in the code, but the paper does not describe one.

## 5. Key results and failure cases

- **Isometry r, arc on M_h vs arc on M_y [P]:**

| setting | r, manifold arc | r, linear chord |
|---|---:|---:|
| weekdays | .99 | .89 |
| months | .89 | .53 |
| letters | .999 | .71 |
| ages | .999 | .36 |
| grid, cylinder | .99 | .90 / .81 |
| Mountain Car | .996 | .06 |

- **E_BC, manifold vs linear [P]:**

| task | manifold | linear |
|---|---:|---:|
| weekdays | 0.34 ± .03 | 0.93 ± .11 |
| months | 0.36 ± .01 | 1.09 ± .06 |
| letters | 2.42 ± .07 | 6.95 ± .27 |
| ages | 5.21 ± .09 | 13.49 ± .29 |

  - The average improvement is 2.8×, with p < .001 in every case.
- **Pullback R², pullback vs linear [P]:**

| task | pullback | linear |
|---|---:|---:|
| weekdays | .77 | .42 |
| months | .75 | .32 |
| ages | .47 | .24 |
| letters | .78 | .23 |

- **2-D ICLR [P]:** "factored control". Steering one grid coordinate leaves the other fixed (Fig 6c, App. Fig 8). Linear steering teleports.
- **Uncertainty [P]** (App. C.2, Llama 70B, layer 70): weekday centroids split by increment size form concentric circles, a cylinder. Steering along one circle keeps the order, but with higher entropy.
- **Failure cases and weaknesses (reported or visible) [P]:**
  - Sequential tasks keep **much higher absolute E_BC** under manifold steering (ages 5.21, letters 2.42 vs ~0.35 for cyclic), so they are far from perfectly natural.
  - Pullback R² for ages is only .47.
  - Mountain-Car pullback degrades "on pairs with one endpoint at the extreme wall position (p ≈ −1.2), where the encoder geometry has tighter curvature."
  - The pullback needs a norm regularizer, or the optimizer escapes into high-norm shortcuts.
  - With α = 0 (the unconstrained simplex geodesic target), the pullback leaves M_h.
  - Limitations (§8): intrinsic coordinates need ground-truth labels ("An unsupervised protocol would … broaden the applicability"); only simple, template-isolated concepts are tested; readout is only the next-token distribution; intermediate variables are not manipulated.
- **[I] Unreported weaknesses:**
  - No held-out values or inputs.
  - Linear baseline confounded by full-residual replacement.
  - No additive-vector or probe-subspace baselines.
  - Mountain-Car "behavior" built from M_h itself.
  - Exact interpolation through centroids, which is fine for 7–24 centroids but may overfit noisy ones.

## 6. Adapting to frozen V-JEPA 2 ViT-L (direction / speed / acceleration, 64 values each)

The situation differs from the paper in two ways:
- There is no generative output. "Behavior" must be the output of probes on downstream layers.
- The representation is 2048 tokens × 1024 per layer, not a single vector.

The choices below are **all [I]**, meaning recommendations. Each lists the paper's choice for contrast.

1. **Intervention site and readout layer.**
   - Paper: a late layer, with a readout that is the model's own output.
   - Default: intervene at a middle layer L (the Part 1 INLP layer) and **run the remaining blocks**. Read out with probes fit on unsteered train data at L+Δ and at the final layer.
   - Don't intervene at the probe layer itself. Then "behavior" is just the probe applied to our edit, which is circular, like Mountain Car's F.
2. **Token handling.**
   - Paper: a single vector, the last token.
   - Default: fit the manifold on the **mean-pooled** tokens at layer L.
   - Intervene by adding the *same* in-subspace delta to every token: Δ = P[s(u(t)) − Pᵀ(h̄ − μ)], where h̄ is the clip's pooled activation.
   - Effect: the pooled in-subspace coordinates become exactly s(u(t)), and per-token variation and the off-subspace residual are kept. This is the closest analogue of "replace the top-k PCA components and keep the residual".
   - Alternative to try: time-resolved manifolds per temporal token group (8 steps).
3. **Subspace.**
   - Paper: 64-d unsupervised PCA over all task inputs.
   - Default: PCA k = 64 of 1024 on pooled train activations *per dataset*.
   - Also try PCA fit on the **64 class centroids** (a supervised, less nuisance-dominated subspace) and the INLP subspace from Part 1. Using the same subspace for linear and manifold steering isolates geometry from subspace choice.
   - Report the fraction of centroid variance captured.
4. **Centroids.**
   - Paper: the mean per answer value.
   - Default: the mean per label value over **train** clips (~14 per value at a 60% split; ~23–24 clips per value in total), marginalizing nuisances.
   - Direction clips mix constant-velocity and accelerating motion. Check that the centroids are not dominated by that split (for example, fit per-subset centroids and compare).
   - Centroids are noisy with ~14 clips each, which motivates smoothing (next item).
5. **Spline fit and smoothing.**
   - Paper: an exact interpolant (LM) or a smoothing spline weighted by √count (Mountain Car).
   - Default: a Reinsch smoothing spline (reuse `CubicSpline1D` with `smoothness` = λ), with λ chosen by **held-out-value cross-validation**: leave out every 4th value (the PLAN.md value-held-out split) and minimize the error of the held-out centroids against s(u).
   - Report λ = 0 as the paper-faithful variant.
6. **Intrinsic coordinate.**
   - Paper: the ordinal index (sequential), or atan2 of PC1/PC2 of the centroids (cyclic).
   - Direction: a **periodic cubic spline on the ground-truth θ** (period 2π, 64 knots at 5.625°), with shortest-arc interpolation. Compare against the paper's unsupervised atan2 coordinate as a check that PC1/PC2 form a loop.
   - Speed and acceleration: natural cubic spline on **log value**, or equivalently the ordinal index if the 64 values are geometrically spaced. Check the spacing in DATA.md or the labels.
   - Also report raw-value and chord-length parameterizations. The parameterization changes *where* along the curve t = 0.5 lands, but not the curve itself.
   - Don't extrapolate beyond the end knots (the natural boundary is linear there).
7. **Topology check.**
   - The Mountain-Car encoder folded a sequential variable into a closed loop.
   - Check whether the speed and acceleration curves nearly close or self-approach, for example the extreme speeds colliding because of motion blur or the disk leaving the frame. If so, linear steering between distant values is especially misleading, and it should be reported.
8. **Start point.**
   - Paper: centroid → centroid, with base inputs of arbitrary value.
   - Default, which is more useful for a per-clip claim: project each test clip onto M_h with `encode_to_nearest_point` to get u₀ and residual r. Then steer u₀ → u\* and set the in-subspace coordinates to s(u(t)) + r, keeping the within-subspace residual.
   - Also run the paper-faithful centroid → centroid version for comparability.
9. **Baselines, all with the same site, tokens, subspace and residual preservation.**
   - (a) Linear interpolation in the same k-d subspace. This is the code's `linear_subspace` mode and the **fair** comparison.
   - (b) The paper's full replacement with the raw centroid.
   - (c) An additive diff-in-means vector h + α(μ_target − μ_source), with α tuned on val.
   - (d) The Part 1 multi-probe INLP-subspace steering, h ← h + U Uᵀ(μ_target − μ_current).
   - (e) Random subspaces and random curves of matched length.
   - For direction, a linear chord between antipodal θ passes through the circle's centre, where the angle is undefined. It is the sharpest predicted failure of linear steering.
10. **Behavior manifold.**
    - Paper: the model's output distribution in Hellinger space.
    - Default: train a **64-way softmax classification probe** (logistic, on unsteered train activations at the readout layer) per variable. Its predicted distribution is the "behavior".
    - Fit M_y as in App. A.4: √ of the class-mean probe distributions, a tangent-plane spline, and E_BC.
    - Use a probe trained on *different* clips than those used for M_h, so the readout does not depend on the manifold. This avoids the Mountain-Car circularity.
    - An optional "other" bin is not needed.
11. **Metrics beyond the paper**, which has no held-out protocol:
    - Target reach: circular MAE for θ; absolute log-error for speed and acceleration, from ridge probes at the readout layer at t = 1.
    - Monotonicity and smoothness along t: Spearman ρ of the decoded value against the intended u(t), and the maximum per-step jump, which quantifies "teleportation".
    - Naturalness: E_BC of the probe distributions against M_y; the entropy of the probe distribution along the path; and the distance of steered activations at the readout layer to their k nearest unsteered train activations.
    - Specificity: drift in the other variables' probes. For example, steering speed should not move the direction probe. This is the analogue of the paper's "factored control".
    - Held-out values: steer to values excluded from the spline fit.
    - Statistics: per-pair scalars averaged over base clips, 50 random ordered pairs per variable, paired tests between methods, and test clips only.
12. **Waypoints.** Paper: K = 50 (LM), 20 (Mountain Car). Default: K = 20–25 to keep forward passes cheap; that is up to 50 pairs × K × n_base_clips passes through the remaining blocks.

**Quantitative targets to compare against the paper [I].** Expect linear chord-distance isometry to be clearly worse for direction (a circle) than for speed and acceleration, unless those curve strongly. The paper's 2.8× E_BC gap is the headline number to compare with.
