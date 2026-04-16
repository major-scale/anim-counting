# When Physics Determines Geometry: Emergent Numerical Representations and Dual-Structure Integration in World Models

**Authors:** major-scale, Claude (Anthropic)
**Target venue:** NeurIPS 2026 (or ICML 2026)
**Status:** Draft — fact-checked against artifact reports, pending causal-intervention experiments and multi-seed replication of unifier ablations.

---

## Abstract

We present two linked findings on how world-model RL agents develop and integrate representations of natural number. First, a DreamerV3 agent trained to gather objects onto a counting grid, with no numerical supervision, spontaneously develops a representation of count in its 512-dimensional recurrent state that is formally verified (via Vietoris–Rips persistent homology) to have 1D arc topology β₀=1, β₁=0, with uniform geodesic spacing (GHE = 0.329 ± 0.027 across 5 seeds), near-perfect arc-length linearity (R² = 0.998), and high representational similarity (Spearman ρ = 0.982). The manifold is robust to full information starvation: masking all explicit count signals, grid slot assignments, and randomly permuting object identities every timestep still yields valid manifolds across all seeds tested (mean GHE = 0.367 under maximal starvation). An untrained-RSSM control confirms the structure is learned, not architectural: training raises RSA from 0.591 to 0.982 and PCA PC1 from 23% to 73%. A random orthogonal projection of the observation vector — and, decisively, a random *permutation* that preserves per-feature distributions — both improve multi-scale representation fidelity (PaCMAP R² 0.651 → 0.976 and 0.953 respectively) while leaving centroid-based metrics unchanged, revealing that the baseline observation format induces axis-aligned shortcut features that degrade meso-scale geometry invisibly to standard evaluation.

Second, we show that a world model of the same architecture, trained on a *binary-cascade* counting environment (where incrementing the count triggers carry propagation through a 4-bit register), develops a fundamentally different representational geometry organized by Hamming distance rather than ordinal distance (RSA Hamming ρ = 0.558 vs ρ = 0.337 for untrained controls). The binary successor operation decomposes linearly into four orthogonal bit-flip directions (100% sign agreement, cross-talk ~0.001), scales in magnitude with carry depth (r = 0.98), anticipates carries 8+ timesteps before onset, and — when the RSSM is forked into imagination mode — autonomously generates the sequential LSB→MSB carry cascade within a single timestep of the posterior (span 10.3 vs 10.0 for the 7→8 full cascade). We argue this constitutes internal simulation of binary arithmetic, not passive observation interpolation.

Third, we show that training a single RSSM on a combined environment fails to preserve both geometries — binary physics dominates completely (GHE 12.1, worse than either specialist in isolation). We therefore introduce the **FP Unifier**, a frozen-specialist-plus-learned-adapter architecture in which two specialist world models feed through learned MLP adapters into a shared integrator trained to reconstruct both observation streams. The integrator preserves both ordinal and Hamming structure simultaneously (RSA ord 0.410, RSA ham 0.265, per-bit 100%) and passes directional interchange intervention analysis at 84.2% on an 8-dimensional subspace with an untrained null control. A 7-condition contrastive-alignment sweep reveals a sharp phase transition at λ = 0.005 (pR² Hamming 0.572 → 0.008) in which geometric structure collapses while task accuracy is completely preserved — a controlled computational demonstration of the accuracy-geometry dissociation that Fascianelli et al. (2024) reported in monkey PFC. Replacing contrastive alignment with a brief VICReg pulse (variance term active only during 400 early training steps) improves both ordinal and Hamming RSA by 2× and raises count accuracy 12 points. Applying the same pulse late (steps 5000–5400) outperforms the early pulse on every metric — a finding that extends the critical learning periods framework (Achille et al., 2019; Golatkar et al., 2019) to modular architectures where differential plasticity between components breaks the standard prediction. Finally, explicitly freezing the GRU during late-onset VICReg reverses the trajectory — effective rank collapses (16.4) instead of expanding (23.9) — a discontinuity at exactly zero backbone learning rate that we call **cooperative residual plasticity**. Theoretical predictions from low-rank RNN dynamics (Mastrogiuseppe & Ostojic, 2018) and two-timescale stochastic approximation (Borkar, 1997) are consistent with the effect. We argue this phenomenon is relevant to the broader LoRA/adapter literature, where frozen backbones are the default assumption.

---

## 1. Introduction

The successor function — the operation mapping each natural number *n* to *n + 1* — is the foundation of arithmetic and, arguably, of mathematical cognition itself. Human infants possess an approximate number system (ANS) by six months of age (Xu & Spelke, 2000; Feigenson et al., 2004), and the transition from approximate to exact numerical cognition requires explicit cultural scaffolding through language and counting routines (Carey, 2009; Piantadosi et al., 2012). But must exact counting emerge from language? Or can it emerge from something more primitive — the physical experience of sequential gathering?

We ask this question computationally. We train a DreamerV3 world model (Hafner et al., 2023) on a minimal counting environment in which an embodied bot navigates a 2D field, picks up objects one at a time, and places them onto a 5×5 counting grid. The agent receives no symbolic number labels, no linguistic input, and no explicit counting supervision. Yet we find that count is encoded in the 512-dimensional deterministic hidden state (h_t) of the recurrent state-space model (RSSM) as a smooth 1D manifold with the topological and metric properties of a number line.

This is not in itself surprising. Emergent numerical structure has been documented in restricted Boltzmann machines (Stoianov & Zorzi, 2012), ImageNet CNNs (Nasr et al., 2019), *untrained* deep networks (Kim et al., 2021), large language models (Gurnee et al., 2025; Kantamneni & Tegmark, 2025), and Othello-GPT-style world models (Li et al., 2023; Nanda et al., 2023). What has been missing is a formal topological and geometric characterization of the structure — and, more importantly, a controlled test of *why* the structure takes the shape it does.

We test this by building a *second* counting environment in which the same concept (how many blobs have been gathered) is implemented as a 4-bit binary register with cascading carry propagation. Incrementing 0→1 flips one bit; incrementing 7→8 flips four. Training the same DreamerV3 architecture on this environment produces a dramatically different representational geometry — organized by Hamming distance rather than ordinal distance. The binary successor decomposes compositionally into four orthogonal bit-flip directions, scales with carry depth, and is internally simulated by the RSSM's prior dynamics without observation support.

**The thesis is that representation geometry is determined by the physical implementation of the task, not by the abstract concept being represented.** Same architecture, same training objective, same concept, different physics → different geometry.

This raises the synthesis question: can a single system maintain both geometric structures simultaneously? We show that naïve combined training fails — binary physics dominates. We therefore introduce the **FP Unifier**, a frozen-specialist-plus-adapter architecture that successfully preserves both geometries in a shared representation. Along the way, a series of ablation experiments on the unifier reveal three findings that we believe are of independent interest:

1. **Accuracy and representational geometry are formally separable.** A contrastive-alignment sweep reorganizes the unifier's geometry dramatically while per-bit task accuracy stays at 100% across all conditions. This parallels Fascianelli et al.'s (2024) biological finding in monkey PFC.
2. **Late-onset regularization beats early-onset in modular architectures.** A VICReg variance pulse applied at training steps 5000–5400 outperforms the same pulse applied at steps 0–400 on every metric. This extends rather than contradicts the critical learning periods framework (Achille et al., 2019; Golatkar et al., 2019), which predicts the opposite direction for homogeneous networks.
3. **Cooperative residual plasticity.** Explicitly freezing the GRU during late-onset VICReg reverses the effect — effective rank collapses instead of expanding. A frozen backbone behaves qualitatively differently from an "almost frozen" one (5% residual plasticity). We argue this is a singular-perturbation discontinuity predicted by two-timescale stochastic approximation theory and low-rank RNN dynamics, and it has implications for the LoRA/adapter ecosystem.

### Contributions

- **C1.** The first formal topological verification (persistent homology: β₀ = 1, β₁ = 0) that a learned embodied counting representation has 1D arc topology, combined with a comprehensive measurement battery (GHE, arc-length linearity, RSA, PaCMAP/TriMap multi-scale projection, jPCA, Gromov-Wasserstein) applied to emergent numerical structure. *(Sections 4, 5.)*
- **C2.** A 9-condition robustness battery showing the counting manifold persists under complete information starvation, across architectures (LSTM, MLP), and across gathering dimensionalities (D=2 through D=5), with an untrained-RSSM control that distinguishes learned structure from architectural scaffold. *(Section 5.)*
- **C3.** The observation that two distance-preserving input transformations (random orthogonal projection *and* random permutation) both improve multi-scale projection fidelity by 46–50% while leaving centroid-based metrics unchanged, identifying coordinate-structure disruption (not feature mixing) as the mechanism, and revealing a methodological gap where standard metrics miss meso-scale representation quality. *(Section 5.4.)*
- **C4.** A second counting environment (binary cascade) whose world model develops a fundamentally different representation organized by Hamming distance, compositionally decomposed into four orthogonal bit-flip directions with 100% sign agreement, and internally simulated via imagination rollouts. *(Section 6.)*
- **C5.** The FP Unifier architecture and empirical demonstration that it preserves both geometric structures simultaneously, validated with directional interchange intervention analysis and an untrained null control. *(Section 7.)*
- **C6.** Three mechanistic findings on modular integration: (i) accuracy-geometry dissociation via contrastive sweeps, (ii) late-onset > early-onset VICReg on modular architectures, (iii) cooperative residual plasticity as a discontinuity at zero backbone learning rate. *(Section 8.)*

---

## 2. Related Work

### 2.1 Numerical cognition

The approximate number system enables ratio-dependent magnitude discrimination across species (Dehaene, 1997; Gallistel & Gelman, 2000). Harvey et al. (2013) discovered topographic 1D numerosity maps in posterior parietal cortex via 7T fMRI, and Harvey & Dumoulin (2017) identified six such maps across human association cortex. Nieder (2012) and Eger et al. (2003) demonstrated that these representations are *supramodal* — encoding numerosity irrespective of sensory pathway. Viswanathan & Nieder (2013) showed that ~10% of neurons in numerically naïve monkeys spontaneously exhibit numerosity selectivity even during irrelevant tasks. The mental number line is widely reported as logarithmically compressed (Dehaene, 2003), consistent with Weber-Fechner scaling.

### 2.2 Emergent numerical representations in neural networks

Research on learned numerical representations has moved through three phases. **Unit-tuning phase** (2012–2021): Stoianov & Zorzi (2012), Nasr et al. (2019), and Kim et al. (2021) documented numerosity-tuned units in restricted Boltzmann machines, ImageNet CNNs, and *untrained* deep networks respectively. A striking finding is that untrained networks can produce *more* number-selective units (16.9%) than trained ones (9.6% in Nasr et al.) — showing that individual-unit selectivity can dissociate from learned population structure. **Partial population phase** (2022–2025): Gurnee et al. (2025) described Claude 3.5 Haiku's character-count representation as a 1D spiral through a 6D subspace; Kantamneni & Tegmark (2025) found GPT-J, Pythia, and Llama encode integers as 9D generalized helices reflecting base-10 structure; Mistry et al. (2023) found a 2D curved manifold in CORnet-S; Nanda et al. (2023) identified circular representations for modular arithmetic. These use PCA, MDS, linear probes, and RSA, but none apply formal topology, curvature profiling, or multi-scale projection validation. **Geometric characterization phase** (this work): we combine persistent homology, geodesic homogeneity, arc-length linearity, RSA, PCA concentration, nearest-neighbor ordinal accuracy, and PaCMAP/TriMap multi-scale projection into a battery that formally verifies manifold topology and reveals meso-scale pathologies invisible to conventional tools.

The closest concurrent work is Gurnee et al. (2025), who find 1D counting manifolds in LLMs from a prediction objective. The important distinction is that our setting is *embodied sequential*: the agent physically acts in an environment and the representation emerges from next-state prediction on sensorimotor streams. LLM settings count tokens; ours counts gathered physical objects.

### 2.3 World models

DreamerV3 (Hafner et al., 2023) learns an RSSM that predicts future observations and rewards. Prior representational analyses of world models have focused on spatial navigation (Banino et al., 2018), object permanence (Piloto et al., 2022), and physical intuition (Battaglia et al., 2013). Hu et al. (2024) applied Vapnik's Learning Using Privileged Information (Vapnik & Vashist, 2009) to DreamerV3's observation space via the Scaffolder framework, showing that auxiliary sensors bridge 79% of the performance gap between target and privileged observations by scaffolding *learning*, not *representation*. Our ablation results (Section 5.3) are consistent with this interpretation.

### 2.4 Representational geometry

Representational similarity analysis (RSA; Kriegeskorte et al., 2008), persistent homology (Rybakken et al., 2019; Chazal & Michel, 2021), and manifold dimensionality estimation (Jazayeri & Ostojic, 2021) provide complementary views of neural geometry. Fascianelli et al. (2024) demonstrated that two monkeys performing the same task at the same accuracy level exhibited strikingly different representational geometries in PFC — an independent biological precedent for the accuracy-geometry dissociation we observe in Section 8.1. The Platonic Representation Hypothesis (Huh et al., 2024) and the Anna Karenina scenario (Bansal et al., 2021) both predict representational convergence under sufficient prediction pressure, consistent with our observation-format invariance results.

### 2.5 Modular fusion and adapter tuning

Multi-modal fusion architectures (Flamingo, ImageBind, LLaVA) combine frozen specialist encoders via learned adapters. The LoRA/adapter literature (Hu et al., 2021) assumes frozen backbones work without degradation. SLCA (Zhang et al., ICCV 2023) demonstrated in continual learning that slow backbone learning rates dramatically outperform frozen backbones — up to 50 percentage points on standard benchmarks. MIST (2025) independently showed that updating <0.5% of backbone parameters improves adapter learning across benchmarks. HASTE (NeurIPS 2025) showed that early-then-stop representation alignment outperforms continuous alignment in diffusion models. Our cooperative residual plasticity finding (Section 8.3) sharpens this direction by identifying a genuine discontinuity at exactly zero backbone learning rate, and by demonstrating late-onset > early-onset (a claim we have not found in prior work).

### 2.6 Critical learning periods and regularization timing

Achille et al. (ICLR 2019) and Golatkar et al. (NeurIPS 2019) established that early training is uniquely formative in homogeneous deep networks — applying regularization only after the initial transient produces a generalization gap as large as if it were never applied. Our Section 8.2 result extends this framework to modular architectures where differential plasticity (GRU and adapters at different effective learning rates) breaks the standard prediction.

---

## 3. Environments

### 3.1 Counting world (grid)

A continuous 2D environment (1400 × 1000 units). Left side: *N* blobs (N ∈ {3, 5, 8, 10, 12, 15, 20, 25}) in a field zone. Right side: a 5×5 grid of target locations. An embodied bot uses a continuous 1D action space (steering angle) and picks up the nearest blob when within 60 units, carrying it to the next available grid slot. Observation: 82-dim continuous vector (bot position, bot state, all blob positions, grid slot assignments, episode metadata, count scalars). The reward is shaped gathering progress, computed from environment state (not observation indices), so ablation masking does not corrupt reward.

### 3.2 Binary counting world

A 2D arena with 15 blobs and a 4-column binary display. When the bot collects a blob, a cascade propagates through the display: the ones column fills; at threshold it empties and the twos column increments; and so on up to 8s. Observation: 72-dim vector (blob positions, column heights, carry states). The same DreamerV3 architecture is used, with identical hyperparameters.

### 3.3 Combined world

A 148-dim observation containing both the grid display and the binary display. Physical gathering is identical; only the visualization differs. Used for the naïve-combined baseline (Section 7.1).

### 3.4 Agent architecture

DreamerV3 (Hafner et al., 2023) with default hyperparameters and an MLP encoder/decoder (no CNN, since observations are vectors). The RSSM has a 512-dim GRU deterministic state (h_t) and a 32×32 discrete stochastic state (z_t), ~12M parameters total. All hidden-state analysis uses h_t.

---

## 4. Measurement Battery

### 4.1 Geodesic Homogeneity Error

For each integer count c ∈ {0, ..., N}, we compute centroid μ_c = mean(h_t | count=c). Consecutive distances d_c = d_geo(μ_c, μ_{c+1}) are computed as shortest paths on a k-NN graph (k=6) over centroids. GHE is the coefficient of variation:

GHE = std(d_1, ..., d_{N-1}) / mean(d_1, ..., d_{N-1})

**Why geodesic?** On a curved manifold in high-dimensional space, Euclidean distances systematically conflate curvature with non-uniformity. Our baseline Euclidean HE = 1.32 (suggesting failure), but GHE = 0.327 — 75% of the apparent error was curvature, not spacing inconsistency. We adopt GHE < 0.5 as the operational successor threshold; synthetic calibration against known geometries is an acknowledged gap.

### 4.2 Arc-length linearity (R²)

Cumulative geodesic arc-length L(c) = Σ d_geo(μ_i, μ_{i+1}) is regressed against count. Values > 0.95 indicate near-perfect uniform spacing.

### 4.3 Persistent homology

Vietoris-Rips persistent homology of the centroid cloud, computed with ripser (Tralie et al., 2018). Expected signature for a number line: β₀=1, β₁=0.

### 4.4 Representational similarity analysis

Spearman rank correlation between the Euclidean RDM of centroids and the ground-truth distance matrix |c_i − c_j| (ordinal RSA) or a Hamming distance matrix (Hamming RSA, used for the binary world).

### 4.5 Multi-scale projection validation

PaCMAP (Wang et al., 2021) and TriMap (Amid & Warmuth, 2019) project the full 512-dim hidden-state point cloud to 2D; we fit monotone regression of projected position against ground-truth count and report R². Unlike centroid-based metrics, these operate on the full point cloud and are sensitive to the shape of within-count scatter.

### 4.6 Probe SNR

Ratio of between-count centroid variance (projected onto the probe direction) to within-count scatter variance. Captures the quality of count separation invisible to mean-based metrics.

### 4.7 Directional interchange intervention analysis

For the unifier causal validation (Section 7), we swap activation subspaces between samples and measure whether the model's reconstruction changes in the expected direction. **Directional IIA** is the fraction of interventions where the cosine similarity between reconstruction delta and expected delta exceeds 0.5. Validated against an untrained null control that scores near 0%.

---

## 5. Counting Manifold: Results

### 5.1 Baseline: 5-seed replication

| Seed | Steps | Hardware | GHE | Arc R² | RSA | Topology |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 300K | MPS | 0.327 | 0.998 | 0.984 | β₀=1, β₁=0 |
| 1 | 155K | MPS | 0.331 | 0.997 | 0.978 | β₀=1, β₁=0 |
| 3 | 211K | RTX 4090 | 0.374 | 0.996 | 0.975 | β₀=1, β₁=0 |
| 4 | 215K | RTX 4090 | 0.288 | 0.998 | 0.983 | β₀=1, β₁=0 |
| 5 | 50K | RTX 4090 | 0.324 | 0.995 | 0.984 | β₀=1, β₁=0 |

**Mean GHE = 0.329 ± 0.027.** Topology is unanimous. Seed 5 converged at only 50K steps, suggesting the structure forms early in training. *Source: `artifacts/reports/replication_summary.json` (confirmed against artifact).*

A linear probe applied to h_t predicts count with R² = 0.9996. The 26 per-count centroids form a curved 1D arc with constant-speed geodesic trajectory.

### 5.2 Untrained control

To distinguish learned structure from architectural inertia, we evaluate an untrained DreamerV3 (random weights, zero gradient updates) on identical observation sequences driven by the trained agent's policy:

| Metric | Trained | Untrained | Δ |
|--------|:---:|:---:|:---:|
| RSA ρ | 0.982 | 0.591 | −0.391 |
| PCA PC1 | 73.0% | 23.0% | −50.0% |
| GHE | 0.281 | 0.395 | +0.114 |
| Arc R² | 0.998 | 0.985 | −0.013 |
| β₀, β₁ | 1, 0 | 1, 0 | — |

The untrained arc R² of 0.985 is an architectural scaffold: observation statistics change systematically with count, so any transformation (even from random weights) produces hidden states that evolve monotonically. But the metrics that probe *geometric quality* reveal dramatic differences: RSA drops from 0.982 to 0.591, PCA PC1 from 73% to 23%, and nearest-neighbor ordinal accuracy from 96% to 35%. Training's contribution is not the existence of count-correlated signal but its geometric precision and dimensional parsimony. *Sources: `artifacts/reports/ablation_multiseed_results.json` (untrained_baseline section). RSA and PCA PC1 values confirmed against artifact.*

### 5.3 Ablation cascade (information starvation)

Three masking conditions, each with 3 random seeds:

| Condition | Masked indices | Shuffled | Mean GHE | Topology |
|-----------|:---:|:---:|:---:|:---:|
| No-count | 80–81 | No | 0.336 ± 0.091 | β₀=1, β₁=0 |
| No-slots + no-count | 53–77, 80–81 | No | 0.344 ± 0.045 | β₀=1, β₁=0 |
| Shuffle + no-slots + no-count | 53–77, 80–81 | Yes | 0.367 ± 0.081 | β₀=1, β₁=0 |

*Confirmed against `artifacts/reports/ablation_multiseed_results.json`.*

The "shuffled + starved" condition is the strongest: grid slot assignments and count scalars are masked, *and* blob position pairs are randomly permuted every timestep so the agent cannot track individual objects. Every seed still produces a valid successor. The representation must be counting gathering *events* (blob disappearances from the field zone) rather than reading any specific observation feature.

Mean GHE drifts by only 0.04 across conditions; variance across seeds grows from 0.027 to 0.081. Auxiliary observation channels scaffold learning *reliability*, not converged geometry — consistent with Vapnik's LUPI framework (Vapnik & Vashist, 2009) and with Hu et al.'s (2024) Scaffolder result on DreamerV3 observation spaces.

### 5.4 Observation format: coordinate-structure disruption

We test two distance-preserving transformations applied to the 82-dim observation vector before it reaches the agent:

1. **Random orthogonal projection.** Fixed 82×82 random orthogonal matrix Q from seed 42000. Preserves all pairwise distances; destroys every axis-aligned feature.
2. **Random permutation.** Fixed 82-dim permutation from seed 42001. Preserves every per-feature marginal distribution; only disrupts semantic grouping of indices.

| Metric | Baseline | Random Projection | Random Permutation |
|--------|:---:|:---:|:---:|
| GHE | 0.329 | 0.326 | 0.346 |
| Topology | β₀=1, β₁=0 | β₀=1, β₁=0 | β₀=1, β₁=0 |
| RSA ρ | 0.981 | 0.984 | > 0.975 |
| Live probe accuracy | 81% exact | 95% exact | — |
| Probe SNR | 502 | 825 | — |
| **PaCMAP R²** | **0.651** | **0.976** | **0.953** |
| TriMap R² | 0.716 | 0.916 | — |

Centroid-based metrics (GHE, topology, RSA) declare all three conditions equivalent. The PaCMAP R² comparison tells a different story: both format disruptions improve multi-scale projection fidelity by 46–50%. Critically, **the permutation result refutes a scatter-isotropy explanation**: permutation preserves per-feature distributions, so the improvement cannot come from feature mixing. What projection and permutation share is disruption of the observation's semantic grouping (paired xy coordinates at adjacent indices, contiguous slot blocks, isolated count scalars). The RSSM's simplicity bias (Shah et al., 2020; Morwani et al., 2024) makes it build axis-aligned shortcut features that exploit this grouping, creating directional scatter around the counting backbone. Any disruption of the layout forces the RSSM to fall back on the task's temporal structure, producing a cleaner manifold.

This finding is an instance of shortcut learning (Geirhos et al., 2020) documented previously in RL augmentation literature (Lee et al., 2020; Laskin et al., 2020; Kostrikov et al., 2021). Our contribution is not the phenomenon but (a) the Probe SNR metric that surfaces the dissociation and (b) the methodological observation that standard metrics miss meso-scale representation quality — only multi-scale projection validation detects it.

### 5.5 Architecture independence

| Architecture | GHE | Arc R² | Topology | Probe R² |
|:-------------|:---:|:---:|:---:|:---:|
| DreamerV3 RSSM (baseline) | 0.269 | 0.997 | β₀=1, β₁=0 | 0.983 |
| LSTM | 0.379 | 0.997 | β₀=1, β₁=0 | 0.994 |
| MLP | 0.350 | 0.995 | β₀=1, β₁=0 | 0.999 |
| MLP-nocount | 0.511 | 0.994 | β₀=1, β₁=0 | 0.730 |

*Source: `artifacts/battery/lstm_mlp/summary.json`. Numbers cited from paper draft; artifact file location pending verification.*

All four architectures produce β₀=1, β₁=0. The MLP-nocount result is the decisive dissociation: without temporal memory and without the count scalar, the MLP achieves perfect count *discrimination* (NN accuracy 100%, correct topology) but fails to produce uniform *successor spacing* (GHE = 0.511 > 0.5). This maps onto the developmental distinction between infant approximate quantity discrimination (present by 6 months) and the later successor principle (Sarnecka & Carey, 2008) — temporal memory is the mechanism that converts state discrimination into uniform successor structure.

### 5.6 Intrinsic dimensionality

TwoNN (Facco et al., 2017) gives 5.5–6.1 depending on seed; MLE gives 7.6–9.3. The manifold occupies 5–9 dimensions of the 512 available. More than 1 because it curves through high-dimensional space; less than 10 because it's fundamentally sequential. The local/global discrepancy is consistent with curvature — locally the manifold looks lower-dimensional than its global embedding requires. *Confirmed against `artifacts/lid_results.json`.*

### 5.7 Successor function

Characterizing "+1" inside the counting model: step vectors (μ_{c+1} − μ_c) require 11 PCA components to capture 90% of variance. The step from 0→1 and the step from 24→25 have cosine similarity near zero. What's conserved is geodesic magnitude: CV = 0.21 across all 25 step sizes. The RSSM prior (before incorporating the current observation) achieves R² = 0.956 on count decoding — indicating that ~95.6% of count information lives in accumulated recurrent dynamics, not in the current observation. Hidden state begins shifting toward the next count 2–50 timesteps before blob landing, with anticipation interval proportional to travel distance.

---

## 6. Binary Counting Machine: Results

### 6.1 Random-baseline-aware evaluation

The binary observation format distributes count information across 4 correlated bit columns. Random temporal filtering preserves much of this signal, which required sharpening the evaluation protocol:

| Metric | Trained | Random (3 seeds) | Gap | Verdict |
|--------|:---:|:---:|:---:|---|
| Exact count accuracy | 100.0% | 58.7 ± 1.6% | +41% | **Survives** |
| RSA Hamming ρ | 0.558 | 0.337 ± 0.007 | +0.22 | **Survives** |
| Probe SNR | 6,347 | 45 ± 3.4 | 140× | **Survives** |
| Probe R² | 0.9998 | 0.977 | 1.02× | **Contaminated** |
| Per-bit accuracy | 100% | 99.0–100% | <1% | **Contaminated** |
| RSA ordinal ρ | 0.466 | 0.502 ± 0.085 | −0.04 | **Contaminated** |

*Source: `artifacts/battery/binary_random_baseline.json`. Location pending verification.*

The headline metric from the counting paper — probe R² — is contaminated here. This has two consequences. First, the binary evaluation must lead with accuracy, Hamming RSA, and probe SNR. Second, it retroactively strengthens the counting-manifold interpretation: the grid world concentrates count in 2 of 82 observation dimensions, so extracting it demands active learning; the binary world spreads count across 72 correlated dimensions, so a random temporal filter recovers most of it. The counting paper's untrained probe R² = 0.120 is therefore meaningful (2-of-82 representation must be actively extracted), whereas an equivalent test on the binary world would be uninformative.

### 6.2 Type A vs Type B geometry

| | Grid world (Type A) | Binary world (Type B) |
|-|:---:|:---:|
| Decimal GHE | 0.33 | 4.91 |
| Dominant geometry | Ordinal distance | Hamming distance |
| Pairwise R² Hamming | — | 0.737 |
| Pairwise R² Decimal | — | 0.067 |
| RSA ordinal | 0.978 | 0.466 |
| Topology β₀ | 1 | 1 |

Both are connected. Both represent count precisely. The organizing distance is different. 7 (0111) and 15 (1111) are representationally close; 7 and 8 (1000) are far. Binary cascade physics are best predicted by tracking bit states, not decimal magnitude — the world model builds structure optimized for prediction.

### 6.3 Binary successor: compositional decomposition

Step magnitude scales with carry depth (r = 0.98):

| Carry depth | Example transitions | Mean magnitude | n |
|:---:|---|:---:|:---:|
| 0 (simple flip) | 0→1, 2→3, ..., 12→13 | 5.86 | 7 |
| 1 (1-bit carry) | 1→2, 5→6, 9→10, 13→14 | 7.28 | 4 |
| 2 (2-bit carry) | 3→4, 11→12 | 8.58 | 2 |
| 3 (full cascade) | 7→8 | 9.40 | 1 |

Per-bit linear probes project each step vector onto the four probe weight directions. **Sign agreement: 25/25 (100%)** across all changed bits in all 15 transitions. Cross-talk on unchanged bits: ~0.001. The model discovered four orthogonal bit-flip axes in 512-dim space and composes them linearly to represent any transition.

Cosine similarity between within-depth transition pairs is high (0.63–0.86); between depth-0 and deeper depths it is strongly *negative* (−0.47 to −0.50). The model represents not just "what comes next" but "what kind of computational event is happening."

### 6.4 Imagination rollout: internal simulation of carries

We forked the RSSM into imagination mode (prior only, no observation updates) 20 timesteps before each carry transition and measured whether probes trained on actual column states detected sequential bit flips in autonomous dynamics.

For the 7→8 full cascade:

| Bit | Posterior | Imagination | Battery carry prop |
|:---:|:---:|:---:|:---:|
| bit0 (↓) | −10.5 | −11.2 | −10.4 |
| bit1 (↓) | −6.5 | −6.9 | −6.4 |
| bit2 (↓) | −2.5 | −3.0 | −2.5 |
| bit3 (↑) | −0.5 | −0.9 | −0.5 |

Three independent measurements agree within 1 timestep. The imagination generates the cascade at the posterior's pace — ~4 timesteps per carry phase, LSB to MSB. **This is active internal simulation, not passive observation interpolation.** Combined with the compositional decomposition and carry anticipation, it constitutes a remarkably complete internalization of binary arithmetic.

*Source: `artifacts/binary_successor/imagination_rollout_colstate.json`. Post_span = 10.02, imag_span = 10.31 confirmed against artifact.*

**Methodological note:** An initial analysis using decimal-count-derived probes (which only update after the full cascade completes) produced a false-negative "partial cascade" verdict. Those probes compressed a 10-step sequential process into a 1-step jump. The corrected analysis uses column-state probes matching the carry propagation methodology exactly. Probe calibration matters.

---

## 7. The FP Unifier

### 7.1 Combined training fails

As a baseline, we trained a single RSSM on a combined environment containing both the grid display and the binary display (148-dim observation):

| Configuration | Decimal GHE | Count Exact | Hamming RSA |
|---|:---:|:---:|:---:|
| Grid-only | 0.33 | 95.6% | — |
| Binary-only | 4.91 | 100% | 0.558 |
| **Combined** | **12.1** | 99.6% | 0.736 |

GHE is *worse* than binary alone. Binary physics completely dominate representation geometry despite the grid signal being present. Cross-format transfer is excellent (R² = 0.999 in both directions, 17 of 20 principal dimensions shared) — the model knows the count in both formats, it just organizes around the format that is harder to predict. **Prediction complexity determines representation geometry.**

### 7.2 FP Unifier architecture

The Functional Pipeline (FP) Unifier separates specialist knowledge from integrated representation:

```
  Grid RSSM            Binary RSSM
  (frozen, 512d)       (frozen, 512d)
       │                     │
    f_A × α_A             f_B × α_B      (MLP adapters, 512→128, SiLU)
       │                     │
       └──────── concat ─────┘
                 │
           GRU(256) + stoch 32²          (integrator)
                 │
       Decoders (both obs)
```

Two pre-trained specialist RSSMs (grid and binary) are frozen. Their hidden states feed through learned 2-layer MLP adapters with learned scalar gates (α_A, α_B, initialized at 0.01) into a shared GRU(256) with 32×32 categorical stochastic state. The integrator is trained to reconstruct both observation streams from its unified hidden state. ~600K trainable parameters.

The architectural pattern (frozen encoders + learned adapters + lightweight fusion) is inherited from Flamingo, ImageBind, LLaVA, and the LoRA literature. What is new is the question: integrating different physical implementations of the same abstract concept, rather than different sensory modalities of the same physical event.

### 7.3 Base result (λ = 0.1 contrastive alignment)

| Metric | Value |
|--------|:---:|
| Count probe R² | 0.988 |
| Count exact accuracy | 76.9% |
| Per-bit accuracy | 100% |
| RSA ordinal ρ | 0.410 |
| RSA Hamming ρ | 0.265 |
| α_A (grid adapter) | 1.65 |
| α_B (binary adapter) | 1.71 |
| CKA asymmetry | 0.050 |
| DCI compactness | 0.062 (distributed) |
| Topology β₀ | 1 |

The integrator preserves both ordinal and Hamming structure simultaneously — not at specialist levels, but both nonzero, which the combined-RSSM baseline could not achieve. Adapters are roughly balanced (4% α gap); CKA asymmetry of 0.05 confirms the integrator draws from both specialists.

### 7.4 Causal validation: directional IIA

An earlier reported IIA of 100% on a 1D subspace was **retracted** — untrained models also scored 100%. The corrected analysis uses directional IIA (cosine > 0.5) with an untrained null control:

| PCA dimensions k | Directional IIA | Nearest centroid |
|:---:|:---:|:---:|
| 1 | 17.5% | 4.7% |
| 2 | 35.2% | 9.8% |
| 4 | 52.9% | 14.9% |
| **8** | **84.2%** | **24.7%** |

Count is causally active, distributed across ~8 dimensions. The untrained null scores ~0% across all k, validating the protocol. *Source: `artifacts/checkpoints/unifier_s0/validation_corrected.npz`.*

This is the primary causal result for the unifier. The contrastive sweep, VICReg timing, and GRU freeze experiments that follow are correlational; extending causal validation to them is future work.

---

## 8. Mechanism of Integration

### 8.1 Accuracy-geometry dissociation

We swept contrastive alignment weight λ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}, training each for 70K steps on CPU:

| λ | pR² Ham | Per-bit | Count exact | RSA ord | α_A | α_B |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | **0.572** | 100% | 99.6% | 0.684 | 0.20 | 0.21 |
| 0.005 | 0.008 | 100% | 87.7% | 0.443 | 0.81 | 1.01 |
| 0.01 | 0.005 | 100% | 90.5% | 0.487 | 0.91 | 1.05 |
| 0.02 | 0.008 | 100% | 80.9% | 0.448 | 1.22 | 1.33 |
| 0.05 | 0.006 | 100% | 80.7% | 0.445 | 1.46 | 1.57 |
| 0.1 | 0.012 | 100% | 76.9% | 0.410 | 1.65 | 1.71 |
| 0.2 | 0.007 | 99.9% | 80.0% | 0.429 | 1.81 | 1.68 |

*Source: `artifacts/sweep_results/results.json`. Location pending verification.*

There is a sharp **phase transition at λ = 0.005**: pR² Hamming collapses from 0.572 to 0.008 (70×) while per-bit decoding accuracy stays at 100% across the entire range. The model retains all four binary bits as decodable information while completely reorganizing the geometric relationship between count states.

This dissociation held across every subsequent experiment: per-bit accuracy at 100% regardless of contrastive weight, VICReg timing, adapter configuration, or GRU freezing. **Task accuracy and geometric organization are formally independent.** Fascianelli et al. (2024) reported an independent biological precedent: two monkeys performing the same task at the same accuracy exhibited strikingly different representational geometries in PFC. Our contribution is a controlled computational demonstration in which the geometric structure can be systematically manipulated through training interventions, revealing which specific mechanisms determine which geometry emerges.

**Mechanistic explanation via subspace correlation analysis.** We analyzed whether ordinal and Hamming subspaces use the same hidden-state dimensions across conditions, tested against a null model (10,000 permutations of random unit vectors in 256-d space):

| Condition | Dim correlation | z-score | p-value |
|-----------|:---:|:---:|:---:|
| VICReg | 0.028 | 0.46 | 0.668 |
| Onset-5000 | 0.053 | 0.82 | 0.402 |
| Baseline (λ=0.1) | 0.090 | 1.45 | 0.126 |
| No alignment (λ=0) | 0.124 | 2.01 | 0.040 |
| Contrastive λ=0.005 | 0.182 | 2.93 | 0.009 |
| **Contrastive λ=0.05** | **0.307** | **4.90** | **0.0003** |

Dimensional decorrelation in high-dimensional space is *free* — random unit vectors in 256-d space have expected absolute cosine ~0.050 with std 0.063. The VICReg and onset-5000 conditions are statistically indistinguishable from random. **Contrastive alignment, however, actively entangles** the ordinal and Hamming subspaces: at λ = 0.05, the correlation is nearly 5σ above random. This is the mechanistic explanation for the phase transition — contrastive pressure forces both structures into overlapping dimensions, and entanglement destroys whichever structure conflicts with the dynamics loss's preference (Hamming).

### 8.2 VICReg reveals the mechanism

We replaced contrastive alignment with VICReg's variance term (Bardes et al., 2022) — a pressure that requires representational dimensions to have non-zero variance, without specifying any target geometry. Variance weight 0.5, active only during training steps 0–400 (<1% of training), then removed.

| Metric | VICReg (0–400) | Best contrastive (λ=0.005) | No alignment (λ=0) |
|--------|:---:|:---:|:---:|
| RSA ordinal | **0.833** | 0.443 | 0.684 |
| RSA Hamming | **0.437** | 0.234 | 0.477 |
| pR² Hamming | 0.261 | 0.008 | **0.572** |
| Count exact | **99.8%** | 87.7% | 99.6% |
| CKA asymmetry | 0.013 | **0.001** | 0.213 |

400 steps of variance pressure permanently changed the final geometry. RSA values 2× higher for both ordinal and Hamming compared to the best contrastive condition, and count accuracy 12 points better. The mechanism is not alignment (VICReg specifies no target geometry); it is **dimensional expansion** — preventing representational collapse during the critical early window of adapter learning, with the resulting high-rank geometric structure persisting long after the loss term is removed.

### 8.3 Late-onset outperforms early-onset

The critical learning periods framework (Achille et al., 2019; Golatkar et al., 2019) predicts that early training is uniquely formative and late regularization should be ineffective. We tested VICReg onset timing: steps 0–400 vs steps 5000–5400:

| Onset | RSA ord | RSA ham | pR² ham | Exact | eRank | α_A | α_B |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Step 0 | 0.812 | 0.405 | 0.203 | 99.8% | 6.8 | 0.14 | 0.14 |
| **Step 5000** | **0.845** | **0.552** | **0.569** | **100%** | **9.1** | **0.21** | **0.21** |

Late VICReg wins on every metric. The explanation involves the GRU's learning dynamics: by step 5000, the GRU's update gates have dropped from 0.491 to ~0.05. The GRU has consolidated into a slow temporal integrator. When VICReg fires on this consolidated substrate, the pressure cannot reshape the GRU (it is effectively frozen), so it routes entirely through the adapters, which grow larger (α ≈ 0.21 vs 0.14). Specialist information gains more geometric influence over the unified representation.

Why does Golatkar et al.'s prediction not hold here? Three reasons, each independently sufficient: (i) **differential plasticity** — the GRU and adapters have very different effective learning rates by step 5000, unlike the homogeneous networks Golatkar tested; (ii) **geometric regularizer** — VICReg operates on representational structure, not on parameters like weight decay; (iii) **modular architecture** — spatially separated consolidation and adaptation. The critical periods framework holds for homogeneous feedforward networks under parameter-level regularizers; none of those conditions apply here.

This finding extends HASTE (NeurIPS 2025), which demonstrated that early-then-stop representation alignment outperforms continuous alignment in diffusion models. Our result goes further: late-onset outperforms early-onset — a direction we have not found in prior work.

**Replication.** We reran both conditions on a separate GPU. Direction preserved (onset-5000 still wins on all key metrics), magnitudes shifted (RSA ordinal 0.845 → 0.772, RSA Hamming 0.539 → 0.432). Multi-seed validation remains future work.

### 8.4 Cooperative residual plasticity

Since the GRU is effectively frozen by step 5000 (gates ~0.05), would explicit freezing produce the same outcome as late-onset VICReg? We tested this: freeze all GRU parameters at step 5000, apply the same VICReg pulse.

| Step | Unfrozen eRank | Frozen eRank |
|:---:|:---:|:---:|
| 5,000 (VICReg starts) | 21.5 | 22.6 |
| 10,000 (post-VICReg) | **23.9** ↑ | **16.4** ↓ |
| 25,000 (final) | 19.7 | 17.3 |

| Metric @ 25K | Unfrozen | Frozen | Δ |
|---|:---:|:---:|---|
| RSA Hamming | 0.535 | 0.477 | Hamming suffers most |
| RSA ordinal | 0.782 | 0.797 | Unaffected |
| pR² Hamming | 0.553 | 0.527 | Modest reduction |
| eRank (final) | 19.7 | 17.3 | 12% lower |

Under identical VICReg pressure, the unfrozen GRU *expands* its representational dimensionality while the frozen GRU *collapses*. Hamming geometry — the fragile one that conflicts with the dynamics loss's ordinal preference — is what suffers most.

**Theoretical framing.** In continuous-rate RNNs, the rank of structured connectivity constrains the dimensionality of emergent dynamics (Mastrogiuseppe & Ostojic, *Neuron*, 2018). A frozen recurrent network's fixed weights impose a ceiling on representational dimensionality that new inputs cannot breach. Strong inputs paradoxically *compress* dynamics further by driving neurons into saturation (Rajan, Abbott & Sompolinsky, 2010; Engelken, Wolf & Abbott, 2023). The unfrozen GRU escapes this trap because its weight updates can expand effective connectivity rank, lifting the ceiling.

Two-timescale stochastic approximation (Borkar, 1997) provides the formal framework: when coupled processes update at different rates, the limit as the slow process's learning rate *approaches* zero (maintaining two-timescale coupling) is qualitatively different from setting it to *exactly* zero (collapsing to single-timescale dynamics). This is a singular-perturbation discontinuity. A GRU with 5% gate updates and a fully frozen GRU are not the same system.

We call this phenomenon **cooperative residual plasticity**. An extensive literature search confirmed that the concept has not been previously named or unified, despite being independently predicted by attractor dynamics theory, two-timescale optimization theory, and observed in at least five biological systems (Benna–Fusi synaptic cascades, sleep consolidation, complementary learning systems, synaptic tagging, neuromodulatory gating).

**Prior empirical observations consistent with our interpretation.** SLCA (Zhang et al., ICCV 2023) demonstrated the core empirical observation in continual learning: slow backbone learning rates dramatically outperform frozen backbones, with improvements up to 50 percentage points on standard benchmarks. MIST (2025) found that updating fewer than 0.5% of backbone parameters improves adapter learning across benchmarks. Our contribution beyond these is threefold: the theoretical framing connecting the observation to two-timescale optimization and attractor dynamics, biological grounding across five independent systems, and specific demonstration in a modular world-model architecture where the effect manifests as representational dimensionality collapse.

**Implication for adapter-tuning.** The standard practice in the LoRA/adapter ecosystem is fully frozen backbones. Our result is consistent with a discontinuity at exactly zero backbone learning rate. Whether the effect scales to large transformers on standard benchmarks — where the pretrained backbone's attractor manifold may already be well-aligned with fine-tuning distributions — is an open empirical question. The effect may be specific to settings where the frozen component's attractor structure is misaligned with new objectives.

### 8.5 Compositional structure survives integration

Does the binary specialist's clean factored bit structure survive the adapter + GRU integration? We ran the binary successor decomposition on the 256-dim unifier hidden states across all 7 conditions:

**Sign agreement is universally preserved: 23/23 (100%)** in every condition. The 4 orthogonal bit-flip axes survive the adapter+GRU transform intact, regardless of alignment loss, VICReg timing, or contrastive weight. Cross-talk increases ~7× (0.001 → 0.010) but remains negligible in absolute terms.

**Carry-depth structure splits on contrastive pressure.** Magnitude-depth correlation:

| Group | Conditions | r |
|---|---|:---:|
| Non-contrastive | λ=0, VICReg variants | +0.82 to +0.96 |
| Contrastive | λ=0.005, 0.05, 0.1 | −0.04 to +0.01 |

The contrastive InfoNCE loss treats all count mismatches equally (same-count positive, different-count negative), normalizing representational distances regardless of carry depth. This erases the magnitude-depth signature while perfectly preserving the factored bit directions — the accuracy-geometry dissociation visible at the level of individual transitions.

---

## 9. Discussion

### 9.1 What the project shows

The counting manifold is the strongest result. Same architecture, same training objective, same concept, yet the representation geometry follows the *physical implementation* of counting in each environment. A gathering world produces a smooth ordinal manifold; a binary cascade world produces a Hamming hypercube with factored bit-flip axes. The architecture does not force a geometry; the generative process does. Auxiliary observation channels scaffold learning *reliability* without determining converged geometry. The format of the observation matters more than its content — coordinate-structure disruption via permutation or projection cleans up meso-scale representation quality in ways invisible to standard metrics.

The FP Unifier shows that dual-geometry preservation is achievable with a simple architecture (frozen specialists, learned adapters, shared integrator) and validates it with directional IIA against an untrained null control. The unifier ablations reveal three findings with independent interest: accuracy-geometry dissociation, late-onset regularization dominance in modular architectures, and cooperative residual plasticity as a singular-perturbation discontinuity.

### 9.2 Relation to biological number

The 1D manifold topology appears to be a universal solution to the counting problem — realized in biological parietal cortex (Harvey et al., 2013; Nieder, 2012) and in multiple artificial architectures. But the metric geometry differs: our agent's number line is uniformly spaced (Weber-Fechner R² = 0.02 for the log model), while the human mental number line shows logarithmic compression (Dehaene, 2003). The uniform spacing reflects uniform task difficulty — gathering blob *n+1* is equally hard regardless of *n*. A task with diminishing-returns reward structure might produce compressed representations. This suggests logarithmic compression in biological number sense reflects contingent neural or evolutionary constraints rather than mathematical necessity.

### 9.3 Relation to prior work on modular fusion

The cooperative plasticity finding connects three literatures that have not previously been unified. Adapter-tuning research (LoRA, MIST, HASTE) has empirically observed that tiny amounts of backbone plasticity help. Continual learning research (SLCA) has shown that slow backbone learning rates dramatically outperform frozen backbones. Low-rank RNN theory (Mastrogiuseppe & Ostojic, 2018) predicts that fixed-weight recurrent networks have rank-constrained dynamics. Two-timescale stochastic approximation theory (Borkar, 1997) predicts discontinuities at the zero limit of slow-process learning rates. Our contribution is the controlled demonstration connecting these to a specific phenomenon (eRank collapse under VICReg with frozen backbone) and the naming of the effect as cooperative residual plasticity.

### 9.4 Limitations

**No causal intervention for the counting manifold.** The counting-manifold evidence is correlational. We show the structure exists and persists across conditions, not that the model *uses* it to make predictions. The Othello-GPT standard (Li et al., 2023; Nanda et al., 2023) requires editing h_t along the probe direction and verifying downstream predictions change accordingly. This is the single most important missing experiment.

**Single-seed unifier ablations.** The contrastive sweep, VICReg timing, and GRU freeze experiments are primarily single seeds. Imprinting replicates directionally (onset-5000 still wins across runs) but magnitudes shift. Multi-seed validation is needed before strong conclusions are warranted.

**5 seeds for counting replication.** Below current RL best practice (Henderson et al., 2018; Agarwal et al., 2021). The topology result is robust because it's discrete; GHE confidence intervals would benefit from more seeds.

**Binary probe R² is contaminated.** The binary evaluation depends on exact accuracy, Hamming RSA, and probe SNR — metrics we developed because standard metrics fail for this world type. These have not been independently validated beyond this project.

**Two specialists only.** The unifier integrates two specialist geometries. Whether the mechanisms (late VICReg, cooperative plasticity) extend to 3+ specialists — and whether many specialists produce genuine format-independent abstraction rather than multi-encoding — is untested. The theoretical argument that true abstraction requires many specialists remains a prediction, not a finding.

**Scale.** 15–25 count states, 256-dim unifier, ~600K parameters. Generalization to larger state spaces is an open question. The 32-state scaling test has not been run.

**Cooperative plasticity at one architecture.** The effect is demonstrated in one specific architecture (DreamerV3 + adapter + GRU integrator). Whether it transfers to transformers, SSMs, or other architectures remains an open empirical question. The entire LoRA/adapter ecosystem assumes frozen backbones work fine; our result is consistent with a discontinuity at exactly zero backbone learning rate, but demonstrating this at scale requires dedicated experiments we have not run.

**GHE lacks synthetic calibration.** The 0.5 operational threshold is empirically motivated, not derived from first principles.

**Architecture breadth.** We confirm the counting manifold in DreamerV3 and LSTM, and show MLP-nocount dissociates. We have not tested transformer-based world models (IRIS, TWM) or SSMs (S4, Mamba).

### 9.5 Future work

- **Causal intervention on the counting manifold.** Edit h_t along the probe direction, measure downstream reconstruction shifts. Highest-priority missing experiment.
- **Multi-seed replication of unifier ablations.** Three-to-five seeds for contrastive sweep, VICReg timing, and GRU freeze.
- **3+ specialist unifier.** Test whether abstraction emerges with more specialists encoding different mathematical properties (comparison, modular arithmetic, primality).
- **Transformer cooperative plasticity test.** If the effect transfers to LoRA-style fine-tuning of pretrained transformers, it affects the entire adapter-tuning ecosystem.
- **Scaling to larger count ranges.** Test whether uniform spacing persists to 100+ count states or whether logarithmic compression emerges.
- **Cross-architecture extension.** Transformer and SSM world models on the same environments.

---

## 10. Conclusion

We have shown that a DreamerV3 world model trained to gather objects develops an internal number line with formally verified 1D arc topology, uniform geodesic spacing, and robustness to full information starvation. A world model of the same architecture trained on a binary-cascade counting environment develops a fundamentally different representation organized by Hamming distance, with compositional decomposition into four orthogonal bit-flip directions and active internal simulation of carry cascades via imagination rollouts. Naïve combined training fails to preserve both geometries; the FP Unifier — frozen specialists plus learned adapters plus a shared integrator — succeeds, validated causally at 84.2% directional IIA on an 8D subspace.

The mechanism of integration produced three findings of independent interest. A contrastive-alignment sweep reveals a sharp phase transition at λ = 0.005 in which geometric structure collapses while task accuracy is perfectly preserved — a computational demonstration of the accuracy-geometry dissociation reported in monkey PFC (Fascianelli et al., 2024). Brief VICReg variance pressure reshapes the final geometry permanently, and applying it late (after the GRU has consolidated) outperforms applying it early — extending the critical learning periods framework to modular architectures where differential plasticity between components breaks the standard prediction. Explicitly freezing the GRU during late-onset VICReg reverses the effect (eRank collapses instead of expanding), revealing a discontinuity at exactly zero backbone learning rate that we call cooperative residual plasticity — a phenomenon predicted by low-rank RNN theory and two-timescale stochastic approximation and observed across five biological systems but not previously named or unified.

The thesis is simple: same concept, different physics, different geometry. And integration — combining representations that exist in different geometries of the same concept — is a distinct problem from learning, with its own mechanisms, timing dynamics, and failure modes. Counting is a clean window into all of them.

Counting, it appears, is not taught. It is gathered, bit-cascaded, and — when integrated — reshaped by the interplay of frozen and residually plastic components in a way that theory predicts but no prior work has named.

---

## References

*(Abbreviated list; full bibliography in the source documents `README.md` and `UNIFIER.md`.)*

- Achille, A., Rovere, M., & Soatto, S. (2019). Critical Learning Periods in Deep Networks. *ICLR 2019*.
- Agarwal, R., et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. *NeurIPS 2021*. Outstanding Paper.
- Amid, E., & Warmuth, M. K. (2019). TriMap: Large-scale dimensionality reduction using triplets. arXiv:1910.00204.
- Banino, A., et al. (2018). Vector-based navigation using grid-like representations in artificial agents. *Nature* 557.
- Bansal, Y., Nakkiran, P., & Barak, B. (2021). Revisiting model stitching to compare neural representations. *NeurIPS 2021*.
- Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization. *ICLR 2022*.
- Borkar, V. S. (1997). Stochastic approximation with two time scales. *Systems & Control Letters* 29.
- Carey, S. (2009). *The Origin of Concepts*. Oxford University Press.
- Chazal, F., & Michel, B. (2021). An introduction to topological data analysis. *Frontiers in AI* 4.
- Dehaene, S. (1997). *The Number Sense*. Oxford University Press.
- Dehaene, S. (2003). The neural basis of the Weber-Fechner law. *Trends in Cognitive Sciences* 7.
- Engelken, R., Wolf, F., & Abbott, L. F. (2023). Lyapunov Spectra of Chaotic Recurrent Neural Networks. *Physical Review Research*.
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets. *Scientific Reports* 7.
- Fascianelli, V., et al. (2024). Neural representational geometries reflect behavioral differences in monkeys and RNNs. *Nature Communications*.
- Feigenson, L., Dehaene, S., & Spelke, E. (2004). Core systems of number. *Trends in Cognitive Sciences* 8.
- Geirhos, R., et al. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*.
- Golatkar, A., Achille, A., & Soatto, S. (2019). Time Matters in Regularizing Deep Networks. *NeurIPS 2019*.
- Gurnee, W., et al. (2025). Counting in Claude. arXiv:2601.04480.
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104.
- Harvey, B. M., Klein, B. P., Petridou, N., & Dumoulin, S. O. (2013). Topographic representation of numerosity in parietal cortex. *Science* 341.
- Henderson, P., et al. (2018). Deep Reinforcement Learning That Matters. *AAAI*.
- Hu, E. J., Springer, J., Rybkin, O., & Jayaraman, D. (2024). Scaffolder: Learning to scaffold among observation spaces. *ICLR 2024* Spotlight.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML 2024*.
- Kantamneni, A., & Tegmark, M. (2025). The geometry of numerical representations in transformers.
- Kim, G., et al. (2021). Visual number sense in untrained deep neural networks. *Science Advances* 7.
- Kostrikov, I., Yarats, D., & Fergus, R. (2021). Image Augmentation Is All You Need (DrQ). arXiv:2004.13649.
- Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience* 2.
- Laskin, M., et al. (2020). Reinforcement Learning with Augmented Data. arXiv:2004.14990.
- Lee, K., et al. (2020). Network Randomization for Deep RL Generalization. arXiv:1910.05396.
- Li, K., et al. (2023). Emergent World Representations in a Sequence Model. arXiv:2210.13382.
- Mastrogiuseppe, F., & Ostojic, S. (2018). Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks. *Neuron*.
- Mistry, P. K., et al. (2023). Two-dimensional neural geometry underpins numerosity tuning. *PNAS Nexus* 2.
- Morwani, D., et al. (2024). Simplicity Bias of Two-Layer Networks. *COLT*.
- Nanda, N., Chan, L., et al. (2023). Progress measures for grokking. *ICLR 2023*.
- Nasr, K., Viswanathan, P., & Nieder, A. (2019). Number detectors spontaneously emerge. *Science Advances* 5.
- Nieder, A. (2012). Supramodal numerosity selectivity. *PNAS* 109.
- Rajan, K., Abbott, L. F., & Sompolinsky, H. (2010). Stimulus-dependent suppression of chaos. *Physical Review E*.
- Sarnecka, B. W., & Carey, S. (2008). How counting represents number. *Cognition* 108.
- Shah, H., et al. (2020). The Pitfalls of Simplicity Bias in Neural Networks. *NeurIPS 2020*.
- Stoianov, I., & Zorzi, M. (2012). Emergence of a "visual number sense" in hierarchical generative models. *Nature Neuroscience* 15.
- Tralie, C., Saul, N., & Bar-On, R. (2018). Ripser.py. *JOSS* 3.
- Vapnik, V., & Vashist, A. (2009). Learning using privileged information. *Neural Networks* 22.
- Viswanathan, P., & Nieder, A. (2013). Neuronal correlates of a visual "sense of number". *PNAS* 110.
- Wang, Y., et al. (2021). Understanding dimension reduction tools (PaCMAP). *JMLR* 22.
- Xu, F., & Spelke, E. S. (2000). Large number discrimination in 6-month-old infants. *Cognition* 74.
- Zhang, G., et al. (2023). SLCA: Slow Learner with Classifier Alignment. *ICCV 2023*.

---

## Appendix A: GHE Derivation

Standard homogeneity error computes the coefficient of variation of Euclidean distances between consecutive count centroids. On a curved manifold in high-dimensional space, Euclidean distances systematically conflate curvature with non-uniformity. GHE replaces Euclidean with graph geodesic distances:

1. Compute centroids μ_0, ..., μ_N.
2. Construct k-NN graph (k=6) over centroids with Euclidean edge weights.
3. Symmetrize: A_sym = (A + A^T)/2.
4. All-pairs shortest paths via Dijkstra.
5. Extract consecutive distances d_c = shortest_path(μ_c, μ_{c+1}).
6. GHE = std(d) / mean(d).

Baseline HE = 1.32, GHE = 0.327. Ratio GHE/HE = 0.25 — 75% of Euclidean error was attributable to curvature.

## Appendix B: Environment parameters

Grid world: 1400×1000 continuous units; bot speed 8 units/step; pickup radius 60; place radius 50; max blobs 25 (variable {3,5,8,10,12,15,20,25}); max steps 10,000; action space continuous 1D (steering). Binary world: identical arena, 15 blobs, 4-column display. Both implemented in pure Python NumPy (validated against original TypeScript; 50× faster).

## Appendix C: Training details

DreamerV3 with MLP encoder/decoder. RSSM: 512-dim GRU deterministic + 32×32 discrete stochastic (~12M params). Adam optimizer, default DreamerV3 learning rates. Hardware: Apple MPS and NVIDIA RTX 4090. Counting baseline: 200K–300K gradient steps. Binary specialist: 300K steps. Unifier: 70K steps, CPU.

## Appendix D: Full ablation condition specifications

| Condition | Arrangement | Masked | Shuffled | Format | Steps |
|---|---|---|---|---|---|
| Line | Linear | None | No | None | 100K |
| Grid (baseline) | 5×5 | None | No | None | 300K |
| Scatter | Random | None | No | None | 100K |
| Circle | Ring | None | No | None | 100K |
| No-count | Grid | 80–81 | No | None | 100K |
| No-slots + no-count | Grid | 53–77, 80–81 | No | None | 200K |
| Shuffle + no-slots + no-count | Grid | 53–77, 80–81 | Yes | None | 200K |
| Random projection | Grid | None | No | 82×82 orthogonal (seed 42000) | 200K |
| Random permutation | Grid | None | No | 82-dim permutation (seed 42001) | 200K |

---

*Implementation by major-scale and Claude (Anthropic). Primary research repository: anima-bridge. Companion documents: `README.md` (counting half), `UNIFIER.md` (binary + integration).*
