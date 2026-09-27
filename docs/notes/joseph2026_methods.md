# Methods digest: Joseph et al. 2026, "Interpreting Physics in Video World Models"

- arXiv 2602.07050v1 (4 Feb 2026). Authors: S. Joseph, Q. Garrido, R. Balestriero, M. Kowal, T. Fel, S. Bakhtiari, B. Richards, M. Rabbat (FAIR/Meta, Mila/McGill, ...).
- Source read: the PDF (33 pages; main text pp. 1-9, appendix pp. 12-33). Local copy: `/tmp/claude-471013/.../scratchpad/papers/joseph2026.pdf` (text: `j.txt`).
  - The e-print tarball served at `arxiv.org/e-print/2602.07050` did not contain this paper. It held the LaTeX for a different Goodfire paper ("Manifold Steering Reveals the Shared Geometry..."), so I ignored it. The HTML version was consistent with the PDF.
- Code: the paper and its HTML link to no official repository. There is an unofficial third-party reproduction, https://github.com/Leesangoh/PEZ-Reproduction. It covers Fig. 1, Fig. 2b/c, Fig. 6 and Fig. 8 (layer-wise probing and IntPhys) but **not** INLP or steering. Its notes on unspecified details are quoted in Section 9.

Conventions in this note: **[P]** means the paper says this (with its location). **[I]** means my inference. **[R]** means it comes from the third-party reproduction repo.

---

## 1. Models, inputs and preprocessing

- **[P] §1, §3.1:** Models are "V-JEPA 2 (Large, Huge, and Giant) ... and VideoMAE-v2 G". The smaller VideoMAE-v2 variants are in App. C.1 (Fig. 7, 9). The paper analyses "the frozen pretrained encoder fθ" and does not use the predictor. The one exception is App. C.1.4, where a V-JEPA 2-L predictor is trained from scratch on each layer.
- **[P]** Every experiment relevant to us (Fig. 2, 4, 18, 22-24; Tab. 3) uses **V-JEPA 2-L**, "d = 1024" (App. C.4), with 24 layers indexed 0-23 (Tab. 3). "Layer 8" is the reference Physics Emergence Zone (PEZ) layer. Fig. 2 marks it at a layer fraction of about 0.35 (8/23).
- **[P] App. A.1.2:** Clips have 16 frames at 24 fps and are "rendered at 256 × 256".
- **[P] App. C.6, conflicting:** "16 input frames are encoded into 8 temporal tokens ... T × N = 8 × 196 = 1568 tokens ... (14 × 14 grid from 224 × 224 images with 16 × 16 patches)". So the paper may have **resized to 224**, even though it rendered at 256 and the public V-JEPA 2-L checkpoint is 256-res. The per-patch heatmaps in Fig. 18c look like 14×14 grids, which supports 224.
- **[P]** The paper does not state which checkpoint it used, the normalization, the frame sampling, or any cropping.
- **[R]** The reproduction used direct 256×256 resize with ImageNet mean/std normalization and the 256-res `vitl.pt` checkpoint.
- **For us [I]:** Use `facebook/vjepa2-vitl-fpc64-256` at native 256 with ImageNet mean/std normalization. That gives 8 × 16 × 16 = 2048 tokens and d = 1024. The 224-vs-256 difference is a known deviation.

## 2. Activation extraction

- **[P] §3.2:** "we probe the **residual stream at every layer**. We primarily use **linear probes on mean-pooled space–time patches** ... we complement these with patch-preserving attentive-mlp probes".
- **[P] App. B:** "linear probes of the form f(hℓ) = W hℓ + b on **spatiotemporally pooled activations from each layer ℓ ∈ {0,...,n−1}**".
- The paper does not say:
  - whether "layer ℓ" means the input to block ℓ (resid-pre) or its output (resid-post);
  - whether the final LayerNorm is applied to the last layer.
- **[R]** The reproduction found resid_post better for the motion probes. It found "temporal_last" pooling (mean over space at the last temporal token only) "critical" for matching Fig. 2c. That contradicts the paper's stated full space-time mean pooling, so treat it as a fitting choice, not the paper's method.
- **Per-patch analyses [P] App. C.5, Fig. 18:** per-patch linear probes, and a "Cross-Patch Spatial Generalization" test that trains on one half of the frame and tests on the other. These are not among our three experiments.
- **MLP-neuron tuning [P] §7.1, App. C.7:** GLM fits on "MLP layers (fc1/fc2)" per neuron and per spatiotemporal position: y = β0 + βcos cos θ + βsin sin θ (Eq. 4), ridge α = 1e-3, 5-fold cross-validated ΔR².
- **Normalization:** not specified in the paper. **[R]** used z-score (polar) or centering (Cartesian).

## 3. Probes, targets, splits and metrics

**Layer-wise probes [P] App. B:**
- Linear probe f(h) = W h + b, trained by gradient descent. This is not closed-form ridge.
- Hyperparameter sweep of 20 configurations: lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 5e-3} × weight decay ∈ {0.01, 0.1, 0.4, 0.8}, "selecting the best model based on validation performance".
- "5-fold **grouped** cross-validation and report results as mean ± standard deviation across folds."
- Not specified: the optimizer (AdamW implied by "weight decay"), the number of epochs, the group key, and whether a nested split separates hyperparameter selection from reporting.

**INLP and steering probes [P] App. C.11:**
- "Direction (θ): Circular regression with outputs (sin θ, cos θ), trained with MSE loss."
- "Speed: Linear regression with scalar output, MSE."
- "IntPhys: Binary logistic regression."
- "Adam optimizer with learning rate η = 10−3 and weight decay λ = 10−4 for 100 epochs (direction) or 50 epochs (speed, IntPhys). We use an 80/20 train/test split with a fixed random seed."

**Direction target:** sin/cos regression (App. C.4, C.11). No classification bins are used for the probes. Fig. 4c reports "Accuracy within 15° (%)".

**Metrics:**
- Fig. 2 y-axis: "Validation R²" for every variable, including direction. The paper does not say how R² is computed for a 2-output sin/cos target. **[I]** Most likely the sklearn-style uniform average over the two outputs.
- Steering uses **mean angular error (MAE, degrees)** (App. C.12).
- The INLP stopping rule uses R² or circular MAE (App. C.11).
- **[R]** For Fig. 2c, the reproduction got its best match with a raw "angle" target rather than sin/cos. With only 8 angles this is plausible, but it is not what the paper describes.

**Units [P], inconsistent:**
- §5.1 says "Ground-truth motion variables are measured in **pixels per frame**".
- App. A.1.2 gives speeds in m/s (1-7) and accelerations in m/s² (2-10).

## 4. Layer-wise results (§5.2-5.3, Fig. 2, V-JEPA 2-L)

Values below are read off the Fig. 2 plot, so they are approximate.

**Polar coordinates (Fig. 2c):**
- Speed: R² ≈ 0.85 at layer 0, rising to about 0.96 at layer 8 and staying flat.
- Acceleration magnitude: about 0.78 at layer 0, then about 0.9.
- **Direction: about 0.22 at layer 0**, rising noisily through about 0.5-0.67 over layers 1-7, then **jumping to about 0.93 at layer 8** and staying about 0.9-0.95 to the output.
- Quote: "speed and acceleration magnitude are available from early layers, while directional information becomes reliably decodable only at the Physics Emergence Zone".

**Cartesian coordinates (Fig. 2b):**
- (vx, ax): velocity about 0.25 and acceleration about 0.1 at layer 0. Both jump to about 0.82-0.88 at layer 8, dip to about 0.65 around layer fraction 0.6-0.7, and recover to about 0.8.
- Quote: "acceleration is also decodable with high R² from early layers ... without relying on explicit intermediate velocity representations".
- **Caveat [I]:** the caption says acceleration is "available at the same time as velocity". That does not really show "high R² from early layers" for Cartesian acceleration, which starts near 0.1. The claim holds for the magnitude only.

**Headline claims:**
- PEZ at about one third of depth. IntPhys probe accuracy goes "from near chance (∼50%) to high performance (∼85–95%)" (§4.2).
- Physics signal peaks in the middle layers and degrades toward the output.
- Direction R² at layer 8 is 0.97 in the Tab. 2/Tab. 4 baseline, a slightly different setup.

**Per-patch results (App. C.5, Fig. 18):**
- Direction is "fragmented across patches" early on. There is "a sharp transition between Layers 7 and 8, where direction becomes decodable from individual patches". Cross-half spatial generalization rises at the PEZ.

## 5. Iterative nullspace probing (§7.2, App. C.11, Fig. 4c, 22, 23; Tab. 3)

The paper calls it the "Orthogonal Probe Sequence Method" and says it is "closely related to ... iterative nullspace projection and amnesic probing (Ravfogel et al., 2020)".

**Exact algorithm [P] App. C.11.** Start with X ∈ R^{N×d}, d = 1024. N is the number of samples, so the activations are **pooled [I]**. Then:
1. "Train a linear probe P_k on the current activations X^(k)".
2. "Extract the probe weights W_k and compute an orthonormal basis via QR decomposition." For direction, W_k ∈ R^{2×d}, so each iteration removes **2 dimensions**. For speed and IntPhys, each iteration removes 1.
3. "Project out the learned direction: **X^(k+1) = X^(k) − X^(k) Q_k Q_kᵀ**".
4. "Repeat until probe performance falls below threshold."

Other details:
- Dimensionality is reported as 2K for direction and K for scalars.
- Probes are retrained from scratch each iteration with Adam (lr 1e-3, wd 1e-4, 100 or 50 epochs) on an 80/20 split with a fixed seed.
- The paper does not say whether earlier Q's are explicitly re-orthogonalized.
- Stopping thresholds are **inconsistent**:
  - C.11: direction "R² < 0.1 or circular MAE > 80° (chance ≈ 90°)"; speed "R² < 0.05 or MAE > 90% of random baseline"; IntPhys "Accuracy < 55% or AUC < 0.55".
  - Fig. 22 caption: "Direction: R² < 0.3; Speed: R² < 0.1; IntPhys: accuracy < 55%".

**Which layer and why:** the method runs at every layer 0-23 (Fig. 22, Tab. 3). The detailed curves (Fig. 4c, 23) are at **layer 8, "the Physics Emergence Zone"**, i.e. the first layer where direction is linearly decodable.

**Plotted quantities:**
- Fig. 22: "Number of Orthogonal Probes" versus layer fraction, for direction, speed and IntPhys.
- Fig. 4c and Fig. 23 left: "Accuracy within 15° (%)" versus orthogonal probe number, 0-100, at layer 8.
- Fig. 23 right: speed "Validation R² (%)" versus probe number, 1-28.

**Findings:**
- §7.2: "Possible-impossible discrimination requires approximately 20 independent features at the Physics Emergence Zone, while direction decoding requires roughly 40–50 features, increasing to up to 80 near the output layers."
- Fig. 22 at layer 8 shows direction ≈ 44 probes, speed ≈ 28, IntPhys ≈ 15. Direction is < 10 probes before layer 8 and climbs to about 82 at the output.
- **Sawtooth:** direction accuracy alternates between about 95-100% and about 20-35% on successive probes for roughly the first 30 probes, then decays noisily to about 20% by probe 100. The paper reads this as "structured redundancy consistent with paired (e.g., sine–cosine) feature encodings". Speed decays smoothly from about 95% to about 5% R² over about 28 probes, with no sawtooth.
- **Internal inconsistency:** Tab. 3 and App. C.11 report direction "subspace dimension" 14-136 (136 at layer 8; 400 at layers 20-23, which looks like a cap). This does not cleanly match Fig. 22's probe counts (44 × 2 = 88 ≠ 136 at layer 8). The counts are sensitive to threshold and split.
- **Subspace overlap (App. C.4):** principal angles between direction and IntPhys are 69-75°, and projection overlap equals the random baseline k_A/d (Eq. 3). The paper concludes the subspaces are "nearly orthogonal".

## 6. Multi-probe subspace steering (§7.2, App. C.12, Fig. 24)

**Subspace [P] Eq. 8.** "From the orthogonal probe sequence ... we obtain K probes with weights W_k ∈ R^{2×d} predicting [sin θ, cos θ]. We construct an orthonormal basis **V ∈ R^{d×2K}** ... V, _ = QR([W1ᵀ, ..., W_Kᵀ])". The held-out run trains "25 probes until R² < 0.1" on the train split and steers with the first N ∈ {1, 2, 3, 5, 10, 15, 20} of them (Fig. 24 x-axis).

**Intervention [P]:** a clamp/replace of the in-subspace coordinates. It is **not** additive with a scale coefficient. Given x ∈ R^d and a target θ*:
1. "c = Vᵀx, x⊥ = x − Vc"
2. "Solve for target coordinates c* via least squares such that all probes predict θ*"
3. "x* = Vc* + x⊥"

The paper does not give the least-squares system. **[I]** A natural reading:
- Stack the rows of all N probes into A.
- For probe k, use W_k·(projection used at step k) applied to V; this equals W_k V because Q_k ⊂ span V.
- Solve min_c Σ_k ‖A_k c + W_k x⊥ + b_k − [sin θ*, cos θ*]‖².
- Note W_k x⊥ = 0 because the rows of W_k lie in span V, so this reduces to A c = t − b.
- A is 2N × 2N, so the system is typically exactly determined.

**Where it is applied:**
- **[P]** to "activations x ∈ R^d" at **layer 8** of V-JEPA 2-L. **[I]** These are the same pooled layer-8 vectors the probes were trained on.
- The paper does **not** push steered activations through later layers, the predictor, or any generation. The effect is read out at the same layer with a probe. The paper never mentions tokens or timesteps. **[I]** The steering is applied to the pooled vector, so it is not per-token.

**Held-out protocol [P] App. C.12:**
1. "Split the dataset into disjoint train (70%, 240 videos) and test (30%, 103 videos) sets". **[I]** 343 videos does not match the 392 in the velocity dataset. The split unit and grouping are unspecified.
2. "Train steering probes on train set activations using the orthogonal probe sequence (25 probes until R² < 0.1)".
3. "Train a **held-out evaluation probe on test set activations only** (R² = 0.99)". This probe is trained on clean test activations with true labels.
4. "Apply steering (constructed from train-set probes) to test set activations".
5. "Evaluate whether the held-out probe reads the target direction from steered activations".

**Target:** all test videos from the 8 directions are steered to θ* = 90°. Required shifts range from 0 to 180°, averaging about 90°.

**Metrics and results [P]:**
- MAE of the held-out probe's decoded angle versus the target, and versus the true label.
- Baseline with no steering: 4.9° to the truth and 82.9° to the target.
- With 20 probes: **11.9° to the target**.
- Quote: "steering with only 1–5 probes yields modest improvement (MAE > 50), while steering with ∼20 probes achieves MAE ≈ 12". Reading Fig. 24: N = 1 → about 77°, 2 → 66°, 3 → 61°, 5 → 51°, 10 → 24°, 15 → 14°, 20 → 11.9°.
- MAE to the true labels rises correspondingly, from about 5° to about 71°. The paper says this confirms "genuinely shifting the representation ... rather than simply injecting the target into a separate subspace".
- Main text (§7.2): "When steering across all probe directions at layer 8, we achieve < 0.5° error to target angles, compared to > 80° for single-probe interventions". **[I]** This is presumably read out by the steering probes themselves, not held-out, so it is near-tautological. The held-out number (11.9°) is the one to reproduce.
- **Unit-circle / MLP steering:** §7.1 says "manipulating only the unit-circle subspace does not effectively steer direction" (no numbers given).

**Controls:** none are reported for steering. There is no random-subspace steering, no matched-norm random perturbation, and no speed-subspace steering. The only random baseline in the paper is the analytic subspace-overlap baseline (App. C.4, Eq. 3).

## 7. Baselines and controls in the paper

- **Control tasks for PEZ specificity (§6.1, Fig. 12):** CLEVRER counting, ImageNet, SSv2, and shuffled-vs-unshuffled detection. Only shuffled detection shows the one-third emergence.
- **Cross-architecture comparison:** VideoMAE-v2 family. G shows a PEZ; smaller variants do not.
- **Attention-locality ablations (App. C.6, Tab. 2/4):** masking attention to nearby tokens at the PEZ. Combined spatial and temporal ablation drops direction R² from 0.97 to 0.14, while ImageNet stays at about 33%.
- **Multi-object check:** per-object direction probes on CLEVRER (App. C.3).
- **Missing baselines:** no pixel/raw-input baseline, no random-init encoder, no shuffled-label control. Layer 0 plays the role of a low-level baseline.

## 8. Datasets and how ours differs

**[P] App. A.1.2, Kubric (PyBullet + Blender), 3D render:**
- A single sphere of radius 0.3 m on an 8×8 m floor, overhead perspective camera at (0, 0, 10), one directional light, zero friction.
- **Velocity set:** 392 videos = **8 directions (0°, 45°, ..., 315°)** × **7 speeds (1-7 m/s)** × 7 start positions ~U[−2, 2]². Constant velocity.
- **Acceleration set:** 280 videos = 8 directions × **5 accelerations (2, 4, ..., 10 m/s²)** × 7 start positions. Starts at rest with a constant force along θ.
- Both: 16 frames, 24 fps, 256×256.
- **IntPhys:** possible/impossible pairs.
- **CLEVRER:** multi-object direction.

**Differences from our dataset (blue disk on a dark background, 16 × 256², with θ, speed and acceleration):**
- Ours is 2D and flat-shaded. Theirs is a 3D-rendered sphere with shading and a floor texture.
- **Their direction takes only 8 discrete values and speed only 7.** This makes grouped cross-validation and "accuracy within 15°" nearly a classification problem. Ours is presumably continuous, so expect the numbers and the sawtooth to differ.
- They use **two separate datasets**: constant velocity for speed and direction, and uniform acceleration from rest for acceleration magnitude. In theirs, acceleration is always parallel to the motion. If our θ, speed and acceleration vary jointly within one set, the targets can be correlated (e.g. mean speed with acceleration). Decorrelate them or report partial R².
- Their N is small (280-392), which caps how many INLP iterations can be fit before overfitting. Probes have d = 1024 features for about 300 training samples. With N < d, a linear probe can fit the training set almost perfectly at every iteration, so test-set R² is what matters.

## 9. Open choices (unspecified or inconsistent) with suggested defaults

| # | Choice | What the paper says / conflict | Suggested default for us |
|---|---|---|---|
| 1 | Input resolution | Rendered at 256; App. C.6 says 224 (14×14 patches) | Native 256 (16×16×8 = 2048 tokens); note the deviation |
| 2 | Preprocessing | Unspecified | Resize to 256, [0, 1], ImageNet mean/std |
| 3 | Readout point | "residual stream", layers 0..n−1 | Output of block ℓ (resid-post), ℓ = 0..23. Also save the patch-embed output as "input". Take hidden states before the final LayerNorm. Check how the HF `hidden_states` are indexed |
| 4 | Pooling | Space-time mean pool (§3.2, App. B); [R] found temporal-last pooling matched better | Main: mean over all 2048 tokens. Ablation: mean over the last temporal token |
| 5 | Feature normalization | Unspecified | Per-dimension z-score fit on the train fold only |
| 6 | Layer-wise probe solver | Gradient descent with a 20-config lr × wd sweep, 5-fold grouped CV | Closed-form ridge with α swept by inner CV (equivalent and faster), 5-fold GroupKFold; report mean ± sd |
| 7 | CV group key | Unspecified | Group by trajectory or seed so no video appears in both folds; for Kubric-like grids, group by start position |
| 8 | Direction target / metric | sin/cos regression; Fig. 2 plots "R²" | sin/cos target. Report R² (mean over the two outputs) **and** circular MAE of atan2(ŝ, ĉ) |
| 9 | Speed / acceleration units | px/frame (§5.1) vs m/s (A.1.2) | px/frame, which is what we have natively |
| 10 | INLP probe | Adam, lr 1e-3, wd 1e-4, 100 epochs, 80/20 split, QR of W_k, X ← X − X Q Qᵀ | Closed-form ridge (small α) per iteration. Project W_k onto the current data span and Gram-Schmidt it against all previous Q's to guarantee exact orthogonality. Fit on the 80% split and evaluate on the 20% split, also projected |
| 11 | INLP stopping rule | C.11 says R² < 0.1 / MAE > 80°; Fig. 22 says R² < 0.3 | Run a fixed number of iterations (e.g. 100 for direction, 50 for speed and acceleration). Plot the full curve and report K at both R² < 0.1 and < 0.3 |
| 12 | INLP layer | Layer 8 (PEZ) for curves; all layers for Fig. 22 | Our empirical PEZ layer (first layer where direction R² jumps), plus a sweep over all layers |
| 13 | Steering subspace | QR of stacked W from the train-split INLP, first N probes | Same; N ∈ {1, 2, 3, 5, 10, 15, 20, 25, all} |
| 14 | Steering LSQ | "least squares such that all probes predict θ*" | Solve [W_1 V; ...; W_N V] c = [t − b_1; ...; t − b_N] with t = (sin θ*, cos θ*) by lstsq; x* = x⊥ + V c*. Use the same normalization as the probes and undo it afterwards |
| 15 | Steering location | Pooled layer-8 vector, read out at the same layer | Reproduce as stated. Optional extension: add Δ = V(c* − c) to every token at layer L and propagate, then read out at later layers |
| 16 | Held-out protocol | 70/30 split (240/103); evaluation probe trained on clean test activations | Same 70/30 split, grouped by video. Evaluation probe: ridge on clean test-split activations, with steering probes fit only on the train split |
| 17 | Target angles | Single θ* = 90° | θ* = 90° to replicate. Also sweep several θ* (e.g. 0, 90, 180, 270) and report the mean |
| 18 | Controls | None for steering | Add three controls: (a) a random orthonormal subspace of equal dimension 2N, with coordinates set to match the norm change; (b) a speed-INLP subspace; (c) MAE to true labels and speed-probe R² after steering, to check specificity |
| 19 | Pixel / random-init baselines | None | Add a raw-pixel (downsampled) ridge baseline and a random-init V-JEPA 2-L at the same readout |
| 20 | Stated "< 0.5° with all probes" | Likely evaluated with the steering probes, not held-out | Report both in-probe and held-out MAE; treat held-out as the primary result |
