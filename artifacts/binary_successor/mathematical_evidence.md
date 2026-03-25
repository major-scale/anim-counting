# Mathematical Evidence Compilation
## "A World Model as Mental Abacus" — Every Equation, Metric, and Statistical Test

---

## 1. Representation Structure

### 1.1 Linear Probe (Ridge Regression)

**Equation:**
```
w* = argmin_w ||Xw - y||^2 + alpha * ||w||^2
w* = (X'X + alpha*I)^{-1} X'y
```

**What it measures:** Finds the best linear mapping from hidden states (X, shape N x 512) to target variable (y). Ridge regularization (alpha=1.0) prevents overfitting.

**Our values:**
| Metric | Trained RSSM | Random RSSM | Raw Obs |
|--------|-------------|-------------|---------|
| Count R^2 | 0.998 | 0.977 | — |
| Per-bit accuracy (all 4) | 100% | 99%+ | — |
| Exact count accuracy | 100% | 58.7% | — |

**Why R^2 is contaminated:** A GRU is a nonlinear temporal filter. Even with random weights, it preserves observation information through its recurrent dynamics. The 72-d binary observation directly encodes bit states across correlated dimensions, so a linear probe on the GRU's 512-d hidden state easily extracts count from temporally smoothed observations. R^2 = 0.977 for an untrained model means R^2 alone cannot distinguish learned structure from architectural pass-through.

**Source:** `artifacts/battery/binary_random_baseline.json`

---

### 1.2 Probe Signal-to-Noise Ratio (SNR)

**Equation:**
```
SNR = ||w||^2 / sigma^2_residual
```
where w is the probe weight vector and sigma^2 is the mean squared residual.

**What it measures:** How "loud" the signal is relative to noise. High SNR means the representation dedicates substantial variance to encoding the target variable in a low-noise subspace. Random models can achieve high R^2 with low SNR (weak signal, weak noise) while trained models achieve high R^2 with high SNR (strong signal, low noise).

**Our values:**
- Trained RSSM: SNR = 6,347
- Random RSSM: SNR = 45
- Ratio: 141x

**Why it survives:** SNR captures not just decodability but signal strength. The trained model dedicates 141x more representational energy to encoding count.

**Source:** `artifacts/battery/binary_random_baseline.json`

---

### 1.3 Per-Bit Decomposition (Step Vector Projection)

**Equation:**
```
delta_h = h_{t+1} - h_t                          (step vector)
proj_b = delta_h . w_b / ||w_b||                   (projection onto bit-b probe direction)
participation_b = |proj_b| > threshold              (binary: did bit b participate?)
```
where w_b is the Ridge probe weight vector for bit b.

**What it measures:** How much of each transition's representational change aligns with each bit's encoding axis. If the representation is compositional, only the bits that actually flip should show large projections.

**Our values:**
- Mean participating projection: +-1.478 units (along probe direction)
- Mean non-participating projection: 0.010 units
- **Participation ratio: 152:1** (1.478 / 0.010)
- This means bits that DON'T flip contribute 0.7% of the signal

**Why it's evidence:** A 152:1 ratio means transition step vectors decompose as nearly exact linear combinations of individual bit axes. This is compositional representation — the model has learned independent axes for each bit, not an entangled count representation.

**Source:** `artifacts/binary_successor/successor_analysis.json`

---

### 1.4 Step Magnitude vs Carry Depth

**Equation:**
```
rho = Spearman(||delta_h||, depth(count))
```
where depth(c) = number of bits that change in the c -> c+1 transition.

**What it measures:** Whether larger representational changes correspond to deeper carry cascades. Since each bit contributes ~1.5 units, a depth-3 cascade (4 bits flip) should produce ||delta_h|| roughly 4x larger than a depth-0 transition (1 bit flips).

**Our value:** rho = 0.9806 (Spearman), p << 0.001

**Why it's evidence:** Near-perfect correlation confirms that step magnitude is determined by how many bits participate, consistent with additive composition of independent bit axes.

**Source:** `artifacts/binary_successor/successor_analysis.json` (field: `depth_magnitude_correlation`)

---

### 1.5 PCA Variance Explained

**Equation:**
```
Sigma = (1/N) * Delta_H' * Delta_H                 (covariance of step vectors)
eigendecompose: Sigma = V * Lambda * V'
cumulative_var(k) = sum(lambda_1..k) / sum(lambda_1..d)
```

**What it measures:** How many orthogonal dimensions are needed to explain the variance in transition step vectors.

**Our value:** 5 PCA components capture 90% of successor function variance.

**Why it's evidence:** A 4-bit binary code has 4 degrees of freedom (4 bit-flip axes). The 5th component likely captures interaction effects (e.g., cascade propagation dynamics). This is near the theoretical minimum — the successor function lives in a ~5-dimensional subspace of the 512-d hidden state.

**Source:** `artifacts/binary_successor/successor_analysis.json`

---

### 1.6 Mutual Information Analysis

**Equation:**
```
I(X; Y) = H(Y) - H(Y|X)
```
Estimated via sklearn's `mutual_info_classif` using k-nearest-neighbor density estimation (Kraskov et al., 2004).

**What it measures:** How much information each hidden dimension carries about each bit, without assuming a linear relationship.

**Our values:**

*Block-diagonal score:* 0.529 (vs 0.25 for uniform/no structure)
- Cluster hidden dims by MI profile -> each cluster has a dominant bit
- All 4 bits represented as cluster dominants

*Subspace overlap (Jaccard on top-20 MI dimensions):*
| Bit pair | Jaccard | Intersection |
|----------|---------|-------------|
| bit0-bit3 (LSB/MSB) | **0.000** | 0 dims |
| bit0-bit1 | 0.176 | 6 dims |
| bit2-bit3 | 0.212 | 7 dims |

*MI decomposition:*
```
Spearman(MI(h_dim, count), sum_j MI(h_dim, bit_j)) = 0.950
```
Count-level MI is almost entirely explained as the sum of bit-level MI.

**Why it's evidence:** Jaccard=0.000 for LSB/MSB means they encode through completely disjoint hidden dimensions. The MI decomposition (r=0.950) confirms count information IS compositional bit information, not a separate entangled code. This is independent of probing — it uses information-theoretic measures, not linear readout.

**Source:** `artifacts/binary_successor/mi_analysis.json`

---

### 1.7 Representational Similarity Analysis (RSA)

**Equation:**
```
centroid_c = (1/N_c) * sum_{i: count_i=c} h_i      (mean hidden state per count)
D_rep = pairwise_euclidean(centroids)                (representational distance matrix)
D_hamming[i,j] = popcount(i XOR j)                  (Hamming distance between binary codes)
D_ordinal[i,j] = |i - j|                            (ordinal distance)

RSA = Spearman(upper_tri(D_rep), upper_tri(D_target))
```

**What it measures:** Whether the geometry of the hidden state space matches the expected geometry. Hamming RSA tests hypercube structure; ordinal RSA tests number-line structure.

**Our values:**
| Metric | Trained RSSM | Random RSSM | ESN |
|--------|-------------|-------------|-----|
| RSA ordinal | 0.500* | 0.470 | 0.230 |
| RSA Hamming | 0.558 | 0.337 | 0.917 |

*Note: The ESN figure directly computed RSSM RSA from battery data as 0.713 ordinal, 0.811 Hamming — the 0.500/0.558 values are from the binary random baseline analysis which used a different computation context.

**Why Hamming RSA survives random baseline:** 0.558 vs 0.337 is a genuine gap (+0.22). The trained model organizes count centroids more faithfully according to Hamming distance than a random temporal filter does.

**Why ESN Hamming RSA is HIGHER:** The ESN is a passive filter that preserves input structure more faithfully. The trained RSSM reorganizes its representation — it doesn't just mirror input Hamming distances, it develops ordinal structure (0.713 vs ESN's 0.230) that's absent from observations.

**Source:** `artifacts/battery/binary_random_baseline.json`, `artifacts/binary_successor/esn_control.json`

---

## 2. Autonomous Simulation

### 2.1 Imagination Span Measurement

**Definition:**
```
span = |crossing_time(last_flipping_bit) - crossing_time(first_flipping_bit)|
```
where `crossing_time(bit_b)` is the interpolated timestep at which probe activation for bit b crosses 0.5.

**Threshold crossing method:**
For a bit transitioning from 0 to 1: find consecutive timesteps t, t+1 where probe(t) < 0.5 and probe(t+1) >= 0.5. Interpolate:
```
t_cross = t + (0.5 - probe(t)) / (probe(t+1) - probe(t))
```

**Our values (30 episodes, column-state probes):**
| Carry depth | N | Posterior span | Imagination span | Ratio |
|---|---|---|---|---|
| 0 | 209 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | — |
| 1 | 90 | 2.00 +/- 0.02 | 2.12 +/- 0.10 | 1.06 |
| 2 | 60 | 6.02 +/- 0.04 | 6.41 +/- 0.28 | 1.07 |
| 3 | 30 | 10.02 +/- 0.05 | 10.31 +/- 0.22 | 1.03 |

**Per-bit crossing times (depth 3, 7->8):**
| Bit | Posterior mean | Imagination mean |
|-----|---------------|-----------------|
| 0 (LSB) | -10.49 | -11.19 |
| 1 | -6.47 | -6.94 |
| 2 | -2.54 | -3.02 |
| 3 (MSB) | -0.47 | -0.88 |

Negative values = steps BEFORE cascade completion. Sequential ordering (LSB first, MSB last) preserved in both modes.

**Why it's evidence:** The imagination mode (no observations, dynamics only) reproduces the correct sequential carry cascade at 1.03-1.07x physical speed. This is not replay — it's autonomous computation.

**Source:** `artifacts/binary_successor/imagination_rollout_colstate.json`

---

### 2.2 Carry Bleed (Non-Participating Bit Deviation)

**Equation:**
```
bleed_b = max_t |probe_b(t) - probe_b(t_start)| / |threshold|
```
for bits b that should NOT change during a cascade.

**What it measures:** Whether bits that shouldn't flip show any deviation during an imagined cascade. Zero bleed means the model knows exactly which bits participate.

**Our value:** < 3.4% maximum deviation on non-participating bits during imagination.

**Why it's evidence:** The model computes carry scope correctly — it knows that 7->8 flips all 4 bits, while 8->9 flips only bit 0.

**Source:** `artifacts/binary_successor/imagination_rollout_colstate.json` (field: `imag_bleed`)

---

### 2.3 Imagination from Count 7

**Test:** Fork RSSM to prior-only mode at count 7. Can it imagine all 7 remaining transitions (7->8->9->...->14)?

**Result:** 7/7 correct across all 10 rollout seeds (single training seed). Zero variance.

**Source:** `artifacts/binary_successor/imagination_stress_test.json` (field: `exp1_multi_start.7`)

---

## 3. Dynamical Analysis

### 3.1 Jacobian and Spectral Radius

**Equation:**
```
J_c = d h_{t+1} / d h_t |_{h = centroid_c}
```
Computed via torch autograd at each of the 15 count centroids.

Spectral radius:
```
rho(J_c) = max_i |lambda_i(J_c)|
```
where lambda_i are eigenvalues of J_c.

**What it measures:**
- rho > 1.0: mildly expansive (perturbations grow)
- rho < 1.0: contractive (perturbations shrink, genuine attractor)
- rho = 1.0: neutral stability boundary

**Our values:**
| Count | Spectral radius | Interpretation |
|-------|----------------|----------------|
| 0 | 1.050 | Mildly expansive |
| 1 | 1.228 | Expansive |
| 2 | 1.116 | Expansive |
| 3 | 1.199 | Expansive |
| 4 | 1.257 | Expansive |
| 5 | 1.307 | Most expansive |
| 6 | 1.218 | Expansive |
| 7 | 1.221 | Expansive |
| 8 | 1.172 | Expansive |
| 9 | 1.135 | Expansive |
| 10 | 1.191 | Expansive |
| 11 | 1.173 | Expansive |
| 12 | 1.191 | Expansive |
| 13 | 1.085 | Expansive |
| **14** | **0.998** | **Sole attractor** |

**Why it's evidence:** The system is everywhere mildly expansive (1.05-1.31) except count 14 (0.998). This means the hidden state is inherently unstable without continuous observation correction — consistent with "controlled explosion" dynamics where sensory input holds the system on-manifold.

**Source:** `artifacts/binary_successor/critical_slowing_down.json` (field: `eigenvalue_summary`)

---

### 3.2 Variance vs Cascade Depth (Anticipatory Destabilization)

**Equation:**
```
var_c = mean(Var(h_t) for t in idle_period_before_transition(c))
depth_c = number of bits flipping in c -> c+1

rho = Spearman(var, depth)
```

**What it measures:** Whether the hidden state becomes more variable before deeper transitions. If the model "anticipates" complex cascades, variance should increase before they happen.

**Our value:** rho = 0.923, p = 2.46e-6

**Per-depth mean variance:**
| Depth | Mean variance | Example counts |
|-------|--------------|----------------|
| 0 | 0.073 | 0, 2, 4, 6, 8, 10, 12 |
| 1 | 0.082 | 1, 5, 9, 13 |
| 2 | 0.088 | 3, 11 |
| 3 | 0.089 | 7 |
| Terminal | 0.014 | 14 |

**Source:** `artifacts/binary_successor/critical_slowing_down.json` (field: `correlations.variance`)

---

### 3.3 Partial Correlation (Ruling Out Timing Artifact)

**Equation:**
```
r_{XY.Z} = (r_{XY} - r_{XZ} * r_{YZ}) / sqrt((1 - r_{XZ}^2)(1 - r_{YZ}^2))
```
where X = variance, Y = carry depth, Z = idle duration.

**What it measures:** Whether the variance-depth correlation survives after controlling for the confound that later counts have longer idle periods (and thus more time to accumulate variance).

**Our value:** r_partial = 0.798, p = 0.0006

**Why it's evidence:** The correlation drops from 0.923 to 0.798 but remains highly significant. Idle duration explains some variance, but the depth-specific signal is genuine and independent of timing.

**Source:** Previously verified (partial correlation computed in the CSD analysis pipeline)

---

### 3.4 What IS NOT Critical Slowing Down

**AR1 Autocorrelation vs Depth:**
```
AR1_c = mean autocorrelation at lag 1 for count c's hidden state
rho = Spearman(AR1, depth) = 0.129, p = 0.66
```
AR1 tracks count ORDER (r=0.99 with count index), not depth. It's a memory management effect — later states have longer histories — not an instability signal.

**Recovery Half-Life vs Depth:**
```
half_life_c = time for perturbation at centroid c to decay by 50%
rho = Spearman(half_life, depth) = 0.145, p = 0.62
```
Not significant. Recovery speed doesn't depend on depth.

**Henrici Departure from Normality vs Depth:**
```
Delta(J) = ||J||_F^2 - sum|lambda_i|^2  (Schur decomposition measure)
rho = Spearman(Delta, depth) = 0.459, p = 0.099
```
Not significant. Non-normal transient amplification doesn't explain the variance pattern.

**Source:** `artifacts/binary_successor/critical_slowing_down.json`

---

## 4. Observation Cliff

### 4.1 Accuracy Measurements

| Condition | Step Accuracy |
|-----------|--------------|
| Continuous observations | 96.2% |
| Peeks every 10 steps | 16.9% |
| Peeks every 25 steps | 10.3% |
| Peeks every 50 steps | 9.5% |
| Peeks every 100 steps | 9.3% |
| Peeks at transitions only | 9.6% |

**Why it's a cliff, not gradual:** The drop from 96.2% to 16.9% occurs at the first interruption. Further spacing makes little additional difference (16.9% -> 9.3%). This is a binary phase transition, not compound error accumulation.

**Source:** `artifacts/binary_successor/imagination_stress_test.json` (field: `exp3_periodic_peeks`)

---

### 4.2 GRU Gate Analysis

**GRU update gate equation:**
```
z_t = sigma(W_z * x_t + U_z * h_{t-1} + b_z)
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
```
z_t near 0 = keep old state, z_t near 1 = accept new input.

**Our values:**
| Condition | Mean gate value |
|-----------|----------------|
| Normal posterior | 0.277 |
| Blind 5 steps | 0.273 |
| Blind 10 steps | 0.273 |
| Blind 20 steps | 0.271 |
| Blind 50 steps | 0.259 |

**Why it's evidence:** Gates are nearly identical across all conditions (0.259-0.277). The model does NOT close its gates when observations are missing — it continues processing at the same rate, but with wrong inputs (zero observations -> wrong posterior corrections).

**Source:** `artifacts/binary_successor/observation_cliff.json` (field: `gates`)

---

### 4.3 Multi-Peek Recovery Plateau

| Peeks | Recovery % | Distance to correct centroid |
|-------|-----------|----------------------------|
| 1 | 99.5% | 7.024 |
| 2 | 99.5% | 6.722 |
| 5 | 100% | 6.503 |
| 10 | 100% | 6.419 |
| 20 | 100% | 6.424 |

**Key observation:** Distance plateaus at 6.4-6.5 even after 20 consecutive peeks. Mean inter-centroid distance is 6.77. The model converges to a stable off-manifold equilibrium that observation-driven correction cannot escape.

**Surgery test:** Directly replacing hidden state with correct centroid achieves 100% (166/166) vs 97% (161/166) without surgery.

**Source:** `artifacts/binary_successor/observation_cliff.json` (fields: `multi_peek`, `surgery`)

---

### 4.4 Drift Trajectory

| Blind steps | Distance to correct centroid | Residual |
|------------|----------------------------|----------|
| 5 | 6.681 | 1.523 |
| 10 | 6.932 | 1.704 |
| 20 | 7.350 | 1.959 |
| 50 | 8.168 | 2.408 |

**Source:** `artifacts/binary_successor/observation_cliff.json` (field: `drift`)

---

## 5. ESN Reservoir Computing Control

### 5.1 ESN Architecture

**Equations:**
```
h_t = (1 - alpha) * h_{t-1} + alpha * tanh(W_in * x_t + W * h_{t-1} + b)
```
- W_in: random input weights (512 x 72), scale 0.1
- W: random recurrent weights (512 x 512), 10% sparse, spectral radius 0.95
- alpha = 0.3 (leak rate)
- All weights FIXED (never trained). Only readout probe is trained.

**Key difference from random RSSM:** The ESN uses a standard reservoir architecture (tanh activation, fixed random weights). The random RSSM uses the full DreamerV3 architecture (GRU + categorical stochastic state) with random weights. Both are "random recurrent networks" but from different architectural families.

### 5.2 Results (3 seeds)

| Metric | Trained RSSM | ESN (mean +/- std) | Random RSSM |
|--------|-------------|-------------------|-------------|
| Probe accuracy | 1.000 | 0.988 +/- 0.002 | 0.587 (exact) |
| RSA ordinal | 0.713 | 0.230 +/- 0.040 | 0.470 |
| RSA Hamming | 0.811 | 0.917 +/- 0.010 | 0.337 |
| Imagination (from 7) | 7/7 (100%) | 0/7 (0%) | — |
| Cliff severity | 96% -> 17% | 92% -> 2% | — |

**Why ESN Hamming RSA > RSSM:** The ESN's inputs directly encode bits (18 identical dimensions per bit), so it passively preserves Hamming structure. The trained RSSM reorganizes AWAY from pure Hamming toward a mixed ordinal/Hamming geometry.

**Why ESN imagination fails:** After 2 steps without input, ESN predictions drift to ~2.5 (fixed-point of tanh dynamics). The ESN has no learned successor function — it can only echo observations.

**Source:** `artifacts/binary_successor/esn_control.json`

---

## 6. TDA (Persistent Homology)

### 6.1 Centroid Topology

**Method:** Compute Vietoris-Rips persistent homology on 15 count centroids in 512-d, up to dimension 3.

**Key results:**
- Distance stats: min=5.57, median=8.94, max=11.54
- At median scale: beta_0=1 (single connected component), beta_1=0 (no loops)
- At mean scale: beta_0=1, beta_1=3 (transient loops), beta_2=0
- No long-lived H1+ features (lifetime > 25% of distance range): 0
- Centroid distance vs Hamming distance: **Spearman r = 0.811, p = 1.0e-25**

**Why it's evidence:** Strong Hamming correlation (0.811) independently confirms hypercube geometry. No persistent high-dimensional holes means the centroid topology is tree-like (hypercube skeleton), not a filled manifold.

**Source:** `artifacts/binary_successor/tda_analysis.json`

---

## 7. Laplace Uncertainty

### 7.1 Last-Layer Laplace Approximation

**Equations:**
```
Posterior: N(w*, Sigma) where Sigma = sigma^2 * (X'X + alpha*I)^{-1}
Predictive: y_new ~ N(x'w*, sigma^2 * (1 + x' * (X'X + alpha*I)^{-1} * x))
```

**Key results:**
- Transition epistemic uncertainty: 1.72x steady-state
- Depth 0 epistemic: 0.000333 -> Depth 3: 0.000412
- Calibration: 83.7% within 1-sigma (overconfident), 97.6% within 2-sigma (well-calibrated)
- Bit 3 (MSB) lowest residual variance (0.0012) — easiest to predict

**Source:** `artifacts/binary_successor/laplace_uncertainty.json`

---

## 8. Conformal Prediction

### 8.1 Split Conformal Prediction

**Method:** Fit probe on training episodes, compute nonconformity scores (|y - y_hat|) on calibration episodes, use quantile for prediction sets on test episodes.

**Key results:**
- 90% target coverage achieved at 86.6% (slightly under-covering)
- Count 7 (deepest cascade): 100% coverage, narrowest width (0.466) — model MOST confident at hardest transition
- Cliff detection: at noise=0.5, suspicious fraction jumps 13.4% -> 64.5%
- Multi-alpha calibration: well-calibrated at all levels

**Source:** `artifacts/binary_successor/conformal_cliff.json`

---

## Artifact Index

| Artifact | Contents |
|----------|----------|
| `battery/binary_baseline_s0/battery.npz` | h_t (13280x512), counts, bits, probe results |
| `battery/binary_random_baseline.json` | Random RSSM baseline (3 seeds) |
| `binary_successor/successor_analysis.json` | Per-bit decomposition, PCA, depth correlation |
| `binary_successor/imagination_rollout_colstate.json` | Imagination vs posterior crossing times |
| `binary_successor/imagination_stress_test.json` | Multi-start, degradation, periodic peeks |
| `binary_successor/critical_slowing_down.json` | CSD metrics, eigenvalues, per-count dynamics |
| `binary_successor/observation_cliff.json` | Drift, gates, multi-peek, surgery |
| `binary_successor/nonnormal_directional.json` | Henrici departure, directional analysis |
| `binary_successor/full_pipeline_trace.json` | Episode traces, PCA centroids, phase durations |
| `binary_successor/mi_analysis.json` | Mutual information, block-diagonal, Jaccard |
| `binary_successor/esn_control.json` | ESN reservoir control (3 seeds) |
| `binary_successor/tda_analysis.json` | Persistent homology, Betti numbers, Hamming RSA |
| `binary_successor/laplace_uncertainty.json` | Laplace probe uncertainty by count/depth |
| `binary_successor/conformal_cliff.json` | Conformal prediction coverage and cliff detection |
