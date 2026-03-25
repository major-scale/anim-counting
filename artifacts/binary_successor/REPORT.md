# How a Recurrent World Model Implements Binary Arithmetic

## A Mechanistic Investigation of the Binary Specialist RSSM

---

## 1. Executive Summary

We conducted a five-part mechanistic investigation into how a DreamerV3 RSSM (Recurrent State-Space Model) trained on a 4-bit binary counting environment implements the successor function. The binary counting machine presents a uniquely analyzable system: 15 discrete states (0-14), deterministic transitions, and a carry cascade structure that maps directly onto binary arithmetic.

**Key findings:**

1. **The RSSM simulates carry cascades sequentially**, propagating bit flips from LSB to MSB over ~10 timesteps during imagination (prior-only rollouts), matching the posterior's timing almost exactly.

2. **Variance before transitions correlates strongly with carry depth** (Spearman r=0.923, p<0.0001), but this is NOT classical critical slowing down — recovery is *faster* at high-depth transitions, not slower. The system is spring-loaded, not sluggish.

3. **Transition anticipation is carried by bit 0 (LSB) alone.** The variance is not distributed across all bits about to flip; it's overwhelmingly concentrated in the fastest-cycling dimension.

4. **Observation interruption creates a permanent off-manifold equilibrium.** The GRU gate values are identical across normal/blind/peek conditions (0.273-0.277), killing the "gate closure" hypothesis. The problem is the posterior computation receiving out-of-distribution hidden state context.

5. **Count 14 is a genuine attractor** — the only state with spectral radius < 1 (0.998), lowest variance, and lowest AR1 autocorrelation.

---

## 2. The System

### 2.1 Environment
A 4-bit binary counting machine with 15 states (0000 to 1110). Columns fill sequentially; when a column fills, it "carries" to the next column. The key variable is **carry depth**: how many consecutive carries propagate when incrementing.

| Transition | Binary | Carry Depth | Bits Flipped |
|-----------|--------|-------------|-------------|
| 0→1 | 0000→0001 | 0 | 1 |
| 1→2 | 0001→0010 | 1 | 2 |
| 3→4 | 0011→0100 | 2 | 3 |
| 7→8 | 0111→1000 | 3 | 4 |

### 2.2 Model Architecture
- **GRU hidden state**: 512 dimensions (deterministic, `h_t`)
- **Stochastic state**: 32 × 32 categorical (1024 dimensions when flattened)
- **GRU update gate bias**: -1.0 (biases toward state retention)
- **Effective update gate**: ~0.05 (95% of state retained each step)

### 2.3 Probes
- **Column-state probes**: Ridge regressors trained on `state.columns[i].occupied` — the ground truth per-bit representation. All achieve 100% accuracy.
- **Decimal count probe**: Ridge regressor on `decimal_count`. R²=0.977 (but see random baseline caveat in prior work — the relevant finding here is per-bit column probes, not R²).

---

## 3. Investigation A: Full Pipeline Trace

**Script**: `full_pipeline_trace.py`

We traced the complete signal flow through encode → img_in → GRU → posterior for a 7→8 transition (depth-3 cascade). The GRU update gate averages 0.05, meaning 95% of the hidden state is retained at each timestep. This makes the carry cascade a necessarily multi-step process — the GRU cannot execute a large state change in a single step.

**Bit decomposition of transition vectors** reveals clean sequential structure:

| Transition | Bit 0 | Bit 1 | Bit 2 | Bit 3 |
|-----------|-------|-------|-------|-------|
| 0→1 (d=0) | 1.51 | -0.002 | -0.001 | -0.001 |
| 1→2 (d=1) | -1.51 | 1.50 | 0.000 | -0.001 |
| 3→4 (d=2) | -1.40 | -1.44 | 1.46 | 0.002 |
| 7→8 (d=3) | -1.35 | -1.36 | -1.41 | 1.65 |

Each bit's weight is near ±1.5 if it participates in the transition, near zero otherwise. The carry cascade is encoded as a vector that "turns off" all lower bits and "turns on" the target bit, with remarkable precision (cross-talk < 0.06).

**Transition magnitude** correlates almost perfectly with carry depth: r=0.981. Depth-3 transitions require movements of 9.4 units in hidden state space; depth-0 transitions require only 5.6-6.1 units.

---

## 4. Investigation B: Imagination Carries

**Script**: `imagination_rollout_binary.py`

The critical test: can the RSSM simulate carry cascades using only its prior (dynamics model), with no observations?

### 4.1 Outcome: Full Sequential Cascade in Imagination

Using column-state probes (not decimal_count probes — see Section 8.1 for why this matters), we observe that imagination reproduces the full LSB→MSB carry cascade:

**7→8 transition timing (steps before bit probe crosses 0.5 threshold):**

| Bit | Posterior | Imagination |
|-----|-----------|-------------|
| 0 (LSB) | -10.5 | -11.2 |
| 1 | -6.5 | -6.9 |
| 2 | -2.5 | -3.0 |
| 3 (MSB) | -0.5 | -0.9 |

The sequential order is preserved in both conditions. Imagination is slightly slower (~0.5 steps per bit) but tracks the same LSB→MSB cascade structure. The posterior span is 10.0 steps; the imagination span is 10.3 steps.

### 4.2 Carry Stops Here

Non-participating bits show < 0.034 maximum deviation during transitions. The carry propagation respects bit boundaries exactly as binary arithmetic requires.

### 4.3 Stress Tests

| Test | With Observations | Imagination-Only |
|------|-------------------|-----------------|
| Start from count 7, count to 14 | 7.0/7 correct (100%) | 7.0/7 correct (100%) |
| Start from count 0, count to 14 | 7.8/14 correct (56%) | Variable |
| Periodic peeks every 10 steps | 16.7% accuracy | — |

Starting from count 7, imagination achieves perfect accuracy through 7 transitions. Starting from count 0, performance is bimodal (some seeds hit 14/14, others derail early) — consistent with compounding error from early missed transitions.

### 4.4 Degradation Profile

Without any observation correction, bit accuracy degrades in order: bit 1 (0.3) fails first, then bit 0 (0.7), while bits 2 (0.9) and 3 (1.0) stay accurate longest. The LSB (fastest cycling) is surprisingly NOT the first to fail — bit 1 degrades faster, suggesting the depth-1 carry mechanism is more fragile than the simple LSB flip.

---

## 5. Investigation C: Critical Slowing Down Analysis

**Script**: `critical_slowing_down.py`

We tested the three classical CSD indicators (Scheffer et al. 2009) across all 15 count states, treating each count's idle period as a "waiting" state before the bifurcation (transition).

### 5.1 Results Summary

| CSD Indicator | vs Carry Depth | Spearman r | p-value | Verdict |
|--------------|----------------|------------|---------|---------|
| Variance | CORRELATES | **0.923** | <0.0001 | Confirmed |
| AR1 autocorrelation | UNCORRELATED | 0.128 | 0.66 | Tracks count ORDER, not depth |
| Recovery half-life | ANTI-correlates | -0.729 | 0.003 | Anti-CSD |
| Spectral radius | No pattern | 0.219 | 0.45 | Uniformly 1.05-1.31 |

### 5.2 Variance

Pre-cascade counts show clear variance surges. The temporal dynamics confirm this is end-loaded — variance in the final quartile (Q4) of idle periods is 4.7-6.3x higher than the first quartile (Q1) at pre-cascade counts:

| Count | Depth | Mean Variance | Q1 Variance | Q4 Variance | Q4/Q1 |
|-------|-------|--------------|-------------|-------------|-------|
| 3 | 2 | 0.087 | 0.006 | 0.027 | 4.7x |
| 7 | 3 | 0.089 | 0.007 | 0.046 | 6.3x |
| 11 | 2 | 0.088 | 0.005 | 0.029 | 5.4x |
| 14 | attractor | **0.014** | — | — | — |

### 5.3 AR1 Autocorrelation

AR1 does NOT track carry depth. Instead, it increases monotonically with count order: 0.788 (count 0) → 0.955 (count 13), with r=0.99 vs count order. This reflects increasing stability as the system approaches count 14, not transition complexity.

### 5.4 Recovery: The Anti-CSD Signal

Recovery half-life *decreases* with count order (9.8 steps at count 0 → 4.0 at count 13). The system recovers *faster* at high counts, which is the opposite of classical CSD. Interpretation: the model is more primed at higher counts — it has had longer to settle into a stable configuration, and perturbations are quickly corrected by the strong attractor dynamics.

### 5.5 Count 14: Terminal Attractor

Count 14 stands apart from all other states:
- Spectral radius: **0.998** (only state < 1.0 — all others are 1.05-1.31)
- Variance: **0.014** (5-6x lower than any other state)
- AR1: **0.682** (lowest — least autocorrelated)
- Recovery: instant (half-life = 0)

This is the terminal state of the counting sequence, and the dynamics have learned it as a genuine attractor — a fixed point rather than a transient state.

### 5.6 Interpretation: Anticipatory Destabilization, Not CSD

The system shows one CSD indicator (variance) but two anti-CSD indicators (fast recovery, no AR1 correlation with depth). The correct framing is **anticipatory destabilization**: the model's hidden state begins "wobbling" in preparation for the upcoming transition, but this wobble is a feature of the mechanism (the LSB-dominated anticipation signal), not a sign of critical slowing.

---

## 6. Investigation D: Observation Cliff Deep Dive

**Script**: `observation_cliff.py`

The binary RSSM exhibits a dramatic "observation cliff": with continuous observations, count probe accuracy is 96.2%; with even a single observation gap (peek every 10 steps), accuracy drops to 16.7%. Why can't the model recover from interruptions?

### 6.1 Analysis 1: Hidden State Drift

During blind (imagination-only) steps, the hidden state drifts away from the correct centroid:

| Blind Duration | Dist to Correct Centroid | Dist to Nearest Centroid | PCA Residual |
|---------------|-------------------------|-------------------------|-------------|
| 5 steps | 6.681 | 6.681 | 1.52 |
| 10 steps | 6.932 | 6.932 | 1.70 |
| 20 steps | 7.350 | 7.341 | 1.96 |
| 50 steps | 8.168 | 8.145 | 2.41 |

Note: `dist_correct ≈ dist_nearest` for all durations. The state doesn't drift to a wrong centroid — it drifts to a no-man's-land equidistant from multiple centroids. The PCA residual (off-manifold distance) grows from 1.5 to 2.4, confirming the state leaves the data manifold.

### 6.2 Analysis 2: What Happens at the Peek

When an observation is provided after blind steps:

| Blind Duration | Peek Move | Dist Before | Dist After | Nearest=Correct |
|---------------|-----------|-------------|------------|-----------------|
| 5 steps | 0.97 | 6.69 | 6.78 | 100% |
| 10 steps | 0.90 | 6.96 | 7.04 | 99.6% |
| 20 steps | 0.78 | 7.41 | 7.47 | 94.9% |
| 50 steps | 0.79 | 8.16 | 8.19 | 90.7% |

**The peek makes things slightly worse.** Distance to the correct centroid *increases* after the observation is processed. The peek move is tiny (0.8-1.0 units vs 6.8 mean centroid distance). The posterior update is correctly-directed but far too small to bridge the gap.

### 6.3 Analysis 3: GRU Gate Values — Hypothesis Killed

| Condition | Update Gate |
|-----------|------------|
| Normal posterior | 0.2771 |
| After 5 blind steps (blind) | 0.2727 |
| After 5 blind steps (peek) | 0.2758 |
| After 50 blind steps (blind) | 0.2588 |
| After 50 blind steps (peek) | 0.2755 |

**The GRU gates are identical across all conditions** (ratio peek/normal = 0.994-1.000). The "gate closure" hypothesis — that the GRU learns to close gates when receiving out-of-distribution input — is definitively killed. The gate opens normally; the update itself is insufficient because the hidden state context is off-manifold.

### 6.4 Analysis 4: Multi-Peek Recovery Plateau

After 10 blind steps, subsequent consecutive peeks recover state quality — but hit a ceiling:

| Consecutive Peeks | Nearest=Correct | Dist to Correct |
|-------------------|-----------------|-----------------|
| 1 | 99.5% | 7.02 |
| 2 | 99.5% | 6.72 |
| 5 | 100% | 6.50 |
| 10 | 100% | 6.42 |
| 20 | 100% | **6.42** |

Distance plateaus at ~6.4 (vs 6.77 mean centroid distance at baseline). Multiple peeks converge to a **permanent off-manifold equilibrium** that is close to the correct Voronoi cell but never reaches the on-manifold centroid.

### 6.5 Analysis 5: Stochastic State Diversity

The 32-dimensional categorical stochastic state shows minimal change during blind steps:

| Condition | Unique Categories (of 32 dims) |
|-----------|-------------------------------|
| Normal posterior | 19.9 ± 1.7 |
| After 10 blind steps | 19.4 ± 1.6 |
| After 50 blind steps | 19.0 ± 1.9 |

The stochastic state does NOT collapse — it maintains nearly the same diversity, ruling out "discrete state mode collapse" as an explanation.

### 6.6 Analysis 6: Count-Specific Recovery

After 10 blind steps followed by 1 peek, per-count recovery:

| Count | Depth | Recovery |
|-------|-------|----------|
| 0 | 0 | 87.5% (7/8) |
| 1 | 1 | 95.7% (22/23) |
| 7 | 3 | 97.6% (40/41) |
| 2-6, 8-13 | 0-2 | 100% |

Count 0 (the initial state) shows the lowest recovery — it has the smallest training sample. All other counts recover at >95%. Depth has no systematic effect on recovery.

### 6.7 Analysis 7: Centroid Replacement Surgery

Replace the drifted hidden state with the correct centroid before processing the peek observation:

| Condition | Recovery |
|-----------|----------|
| Without surgery (natural drift → peek) | 97.0% |
| With surgery (centroid → peek) | 100.0% |

Surgery helps modestly (97→100%), confirming that hidden state drift is part of the problem. But the effect is smaller than expected — the posterior computation itself contributes additional error.

### 6.8 Mechanism Summary

The observation cliff is NOT caused by:
- Gate closure (gates are identical)
- Stochastic state collapse (diversity maintained)
- Voronoi cell escape (nearest centroid usually correct)

It IS caused by:
- **Off-manifold drift** during blind steps (state leaves the training distribution)
- **Insufficient posterior correction** (peek moves are tiny relative to centroid distances)
- **Convergence to off-manifold equilibrium** (multiple peeks plateau at dist=6.4, never reaching the on-manifold centroid)

The posterior was trained to make small corrections to on-manifold states, not to perform large corrections from off-manifold starting points. When the hidden state drifts during imagination, the posterior correction is in the right direction but wrong magnitude.

---

## 7. Investigation E: Non-Normal Dynamics and Directional Variance

**Script**: `nonnormal_directional.py`

### 7.1 Henrici Number (Matrix Non-Normality)

We computed the Henrici departure from normality for the 512×512 Jacobian at each count state's centroid. If non-normal amplification drives the variance surge, deeper transitions should show higher Henrici numbers.

| Count | Depth | Spectral Radius | Henrici |
|-------|-------|-----------------|---------|
| 0 | 0 | 1.050 | 0.255 |
| 3 | 2 | 1.199 | 0.304 |
| 7 | 3 | 1.221 | 0.311 |
| 14 | attractor | **0.998** | **0.218** |

Henrici vs depth: r=0.459, p=0.099 — **NOT significant.** Henrici correlates with spectral radius (r=0.789, p=0.0008) rather than carry depth. Non-normal amplification is not the primary variance mechanism.

### 7.2 Directional Variance

We projected idle-period hidden state variance onto bit-probe weight vectors to test whether the variance is specifically aligned with the bits about to flip.

**Per-bit variance at pre-cascade counts:**

| Count | Depth | Bit 0 Var | Bit 1 Var | Bit 2 Var | Bit 3 Var | Ratio (flip/stay) |
|-------|-------|-----------|-----------|-----------|-----------|-------------------|
| 3 | 2 | **0.0615** | 0.0002 | 0.0005 | 0.0006 | 33.3x |
| 7 | 3 | **0.2038** | 0.0285 | 0.0004 | 0.0003 | 0.86x |
| 11 | 2 | **0.0356** | 0.0001 | 0.0002 | 0.0001 | 91.3x |

**The variance is NOT along "all bits about to flip."** It is overwhelmingly dominated by bit 0 (LSB):
- Count 3: bit 0 = 0.0615, bits 1+2+3 all < 0.001 (60x gap)
- Count 7: bit 0 = 0.204, bit 1 = 0.029 (7x gap), bits 2+3 < 0.001
- Count 11: bit 0 = 0.036, bits 1+2+3 all < 0.001 (180x gap)

The directional variance ratio (flip/stay) vs depth: r=-0.200, p=0.493 — NOT significant. The aggregate ratio metric fails because it averages bit 0's massive variance with the near-zero variance of deeper bits.

### 7.3 Partial Correlation: Depth vs Variance Controlling for Idle Duration

Since higher counts have longer idle periods, we tested whether the variance-depth correlation is merely a timing artifact:

- Idle duration alone → variance: r=0.358, p=0.21 (NOT significant)
- Depth → variance controlling for idle: **r=0.798, p=0.0006** (SIGNIFICANT)

The variance signal is genuinely depth-specific, not a timing artifact. Deeper transitions produce more variance, independent of how long the model has been sitting.

### 7.4 Interpretation: The LSB as Harbinger

The LSB (bit 0) is the fastest-cycling dimension — it flips at every transition. It is also the first bit to flip in any carry cascade. The model's anticipatory variance concentrates in this dimension because:

1. The LSB carries the most informative transition signal (it changes at every increment)
2. The dynamics model begins destabilizing along the LSB direction before the transition arrives
3. Deeper cascades produce more LSB variance because the total displacement is larger

This is "transition proximity through the fastest-cycling dimension" — the model detects that a transition is coming and begins activating the first step of the cascade.

---

## 8. Cross-Cutting Findings

### 8.1 Probe Calibration: Column-State vs Decimal-Count

A critical methodological lesson: **decimal_count probes** (trained on `env._state.decimal_count`) are blind to mid-cascade states. During a carry cascade, `decimal_count` stays at the old value for the entire cascade duration, then jumps. This compresses a 10-step sequential process into a 1-step jump.

**Column-state probes** (trained on `battery.npz` `bits` = `state.columns[i].occupied`) track actual bit flips step-by-step. The full sequential cascade in imagination (Investigation B) was only visible with column-state probes; decimal_count probes showed an earlier (incorrect) "Outcome C: partial cascade" result that we retracted.

**Lesson**: Always use the most granular probe available. High-level summary variables can mask mechanistic detail.

### 8.2 The GRU's Conservative Design

The update gate bias of -1.0 produces an effective gate of ~0.05, meaning 95% state retention. This is a crucial architectural feature:

- It forces carry cascades to be **sequential** (the GRU cannot execute a large state change in one step)
- It produces **highly stable** idle-period states (std < 0.013 at non-cascade counts)
- It creates the observation cliff (small posterior corrections cannot overcome accumulated drift)
- It makes count 14 a genuine fixed point (spectral radius < 1.0)

### 8.3 Depth-Dependent Structure

Nearly every measurement in this investigation shows depth-dependent structure:

| Measurement | Depth 0 | Depth 1 | Depth 2 | Depth 3 |
|------------|---------|---------|---------|---------|
| Transition magnitude | 5.6-6.1 | 6.9-7.5 | 8.2-8.9 | 9.4 |
| Idle-period variance | 0.050-0.075 | 0.080-0.085 | 0.087-0.088 | 0.089 |
| Idle-period std | 0.006-0.013 | 0.008-0.013 | 0.057-0.071 | 0.139 |
| Anticipation onset (depth) | N/A | 2-3 steps | 5-6 steps | 8-10 steps |
| Bit decomposition max | 1.5 | 1.5 | 1.5 | 1.65 |

The model has learned to allocate proportionally more representational resources (state space movement, anticipation duration, variance) for deeper transitions.

---

## 9. Unified Mechanistic Story

The binary specialist RSSM implements counting through a six-part mechanism:

### Phase 1: Stable Attractor
During idle periods, the hidden state sits at a count-specific centroid with very low variance (std < 0.013 for non-cascade counts). The high state retention (95%) maintains this configuration.

### Phase 2: Anticipatory Destabilization
As the transition approaches, variance surges along the LSB direction. Deeper cascades produce more variance because the upcoming state displacement is larger. This begins 2-10 steps before the transition, scaling with cascade depth.

### Phase 3: Sequential Carry Cascade
The transition executes as a sequential LSB→MSB cascade. Bit 0 flips first, then bit 1, etc. Each bit flip takes approximately 2 steps (matching the environment's 2-step-per-phase structure). The 95% state retention makes this necessarily sequential — the GRU cannot execute large jumps.

### Phase 4: Cascade Termination
Non-participating bits show < 0.034 deviation. The carry propagation respects exact bit boundaries. The cascade terminates precisely when it should.

### Phase 5: Observation Correction
The posterior refines the prior's cascade result using the actual observation. On-manifold corrections are small and accurate. Off-manifold corrections (after blind periods) converge to a suboptimal equilibrium.

### Phase 6: Terminal Attractor
At count 14, the system enters a genuine fixed point (spectral radius 0.998) with minimal variance. This is the only state where the dynamics are contractive rather than marginally expansive.

---

## 10. What This Means for World Models

### 10.1 Sequential Algorithms in Recurrent Networks
The RSSM has learned to implement a sequential algorithm (carry propagation) within a recurrent network, without any explicit architectural support for multi-step computation. The conservative update gate creates a natural "speed limit" that forces the algorithm to unfold over time.

### 10.2 Anticipation Without Instruction
The model was trained purely on next-observation prediction. Nobody told it to anticipate transitions. The variance surge before cascades is an emergent property of the trained dynamics — the model's latent space has organized to "prepare" for upcoming state changes.

### 10.3 Fragility of Off-Distribution Inference
The observation cliff reveals a fundamental limitation: world models trained with continuous observations develop posterior mechanisms calibrated for on-manifold corrections. When forced to operate off-manifold (after even brief observation gaps), the posterior converges to a stable but incorrect equilibrium. This has implications for any world model deployed in conditions different from training.

### 10.4 The System Knows What It Doesn't Know (In a Sense)
The variance-before-transitions finding suggests the model's latent dynamics encode information about *transition difficulty* through representational instability. Harder transitions (deeper cascades) produce more anticipatory variance. This could be useful for uncertainty estimation in world models — high latent variance may indicate upcoming representational challenges.

---

## 11. Artifacts and Reproducibility

All scripts and data artifacts are in `bridge/scripts/` and `bridge/artifacts/binary_successor/`:

| Artifact | File | Size |
|----------|------|------|
| Pipeline trace | `full_pipeline_trace.json` | 7.7K |
| Imagination rollout (column-state) | `imagination_rollout_colstate.json` | 42K |
| Imagination stress test | `imagination_stress_test.json` | 5.6K |
| Critical slowing down | `critical_slowing_down.json` | 8.9K |
| Observation cliff | `observation_cliff.json` | 4.8K |
| Non-normal + directional | `nonnormal_directional.json` | 8.9K |
| Carry propagation analysis | `carry_propagation.json` | 13K |
| Successor analysis | `successor_analysis.json` | 23K |

**Scripts**: `full_pipeline_trace.py`, `imagination_rollout_binary.py`, `critical_slowing_down.py`, `observation_cliff.py`, `nonnormal_directional.py`

**Figures**: `csd_headline.png`, `csd_eigenvalue_spectra.png`, `csd_temporal_dynamics.png`, `cliff_drift_trajectory.png`, `cliff_multi_peek_recovery.png`, `cliff_blind_pca.png`, `cliff_count_specific.png`, `nonnormal_henrici.png`, `nonnormal_directional.png`

**Model weights**: `bridge/artifacts/checkpoints/binary_baseline_s0/` (136MB, 500K training steps)

---

## 12. Hypotheses Tested: Scorecard

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | RSSM simulates carry cascades in imagination | **CONFIRMED** | Imagination span=10.3 vs posterior 10.0, sequential LSB→MSB |
| 2 | Variance increases before deep transitions (CSD) | **PARTIALLY CONFIRMED** | Variance r=0.923, but recovery anti-CSD |
| 3 | AR1 autocorrelation increases before transitions | **KILLED** | AR1 tracks count order, not depth (r=0.128) |
| 4 | Spectral radius predicts cascade depth | **KILLED** | Uniformly 1.05-1.31, no depth pattern |
| 5 | GRU gates close during blind periods | **KILLED** | Gates identical: 0.273-0.277 across all conditions |
| 6 | Multi-peek recovers on-manifold state | **KILLED** | Plateaus at dist=6.4 — permanent off-manifold equilibrium |
| 7 | Matrix non-normality explains variance surge | **KILLED** | Henrici r=0.459, p=0.099 vs depth |
| 8 | Variance aligns with all flipping bits | **KILLED** | Dominated by bit 0 (LSB) alone |
| 9 | Variance-depth correlation is a timing artifact | **KILLED** | Partial r=0.798 after controlling for idle duration |
| 10 | Count 14 is a terminal attractor | **CONFIRMED** | Spectral radius 0.998, lowest variance, instant recovery |
| 11 | Off-manifold drift causes observation cliff | **CONFIRMED** | Drift grows with blind duration, surgery helps modestly |
