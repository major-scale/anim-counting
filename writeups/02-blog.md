# Same Concept, Different Shapes: How Physics Determines the Geometry of Number Inside a World Model

*A technical walk-through of the anima-bridge project — from the emergent counting manifold, to the binary counting machine, to the dual-geometry integration problem.*

*Initial release: this post focuses on the two results with the strongest empirical support (the counting manifold and the binary world's dual geometry). The integration architecture (FP Unifier) is sketched at the end; full numerical tables for the unifier experiments are pending re-export from cloud runs and will accompany the formal paper.*

---

## The question

Can an artificial system develop a representation of number without being taught one?

The question isn't new — Stoianov & Zorzi (2012) showed numerosity detectors emerging in restricted Boltzmann machines; Nasr et al. (2019) found them in ImageNet CNNs; Kim et al. (2021) found them in *untrained* networks; Gurnee et al. (2025) found 1D counting manifolds in Claude 3.5 Haiku; Kantamneni & Tegmark (2025) found 9D helical integer representations in transformers. The existence of emergent numerical structure is well-established at this point.

What's been missing is a clean answer to a narrower question: **what shape does a learned counting representation take, and why does it take that shape?**

This project starts from a simple hypothesis — that counting-like geometry should emerge from embodied sequential prediction, because the environment's generative process provides the inductive bias — and spends most of its energy testing that hypothesis from every angle it can think of. In the process, it ends up somewhere unexpected: with two counting worlds that produce two very different geometries from the same architecture. The thesis of the project is that **representation geometry is determined by the physical implementation of the task, not by the abstract concept being represented**.

This post walks through the two halves. It assumes you're comfortable with RL, representation learning, and basic topology, but nothing more specialized than that.

---

## Part 1: The Counting Manifold

### Setup

We train a DreamerV3 world model (Hafner et al., 2023) on a minimal gathering environment. A bot navigates a 2D field, picks up one of up to 25 blobs, and places it in the next available slot of a 5×5 counting grid. Episodes terminate when all blobs are placed. The observation is an 82-dimensional vector: bot position (2), bot state (1), all blob positions (50), grid slot assignments (25), episode metadata (4). The agent gets a shaped gathering reward derived from environment state — critically, the reward is independent of observation indices, so we can mask observation dimensions without corrupting the reward.

The world model has an RSSM with a 512-dimensional GRU deterministic state (h_t) and a 32×32 discrete stochastic state (z_t) — roughly 12M parameters. All analysis targets h_t.

We then ask: does count appear as a geometrically structured feature of h_t?

### The core finding

**A linear probe applied to h_t predicts count with R² = 0.9996.** But decodability is weak evidence — it confirms the signal is present, not that it's structured. So we build count-conditioned centroids μ_c = mean(h_t | count=c) for c ∈ {0, ..., 25} and analyze the resulting point cloud.

**Topology.** Vietoris–Rips persistent homology (via ripser) on the centroids gives β₀ = 1, β₁ = 0. One connected component, no loops. This rules out rings, tori, disconnected clusters — but doesn't uniquely identify a line (any contractible space gives the same signature). We supplement with:

**Intrinsic dimensionality.** TwoNN (Facco et al., 2017) gives 5.5–6.1; MLE gives 7.6–9.3. The manifold lives in roughly 5–9 dimensions of the 512 available. More than 1 because it curves; less than 10 because it's fundamentally sequential.

**Arc-length linearity.** Cumulative geodesic distance along the centroids (computed via k-NN graph shortest paths, k=6) is near-perfectly linear with count: R² = 0.998. Uniform ruler.

**Geodesic Homogeneity Error.** We computed the coefficient of variation of consecutive geodesic distances. GHE = std(d_c) / mean(d_c). The baseline GHE is 0.329 ± 0.027 across 5 seeds. The same data under *Euclidean* distances gave HE = 1.32 — about 75% of the apparent non-uniformity was curvature, not spacing error. This is why GHE needed inventing.

**RSA.** Spearman correlation between the centroid RDM and |c_i − c_j|: ρ = 0.982. Ordinal structure intact.

Five seeds, 50K–300K training steps, different hardware (MPS and RTX 4090). Every single one: mean GHE 0.329, topology unanimous, RSA > 0.975. *Source: `artifacts/reports/replication_summary.json`.*

### The ablation table

The manifold's robustness is the strongest evidence that it isn't an accident. Every row below is a separate training run:

| Condition | GHE | Topology | RSA |
|-----------|:---:|:--------:|:---:|
| Grid baseline (5 seeds) | 0.329 ± 0.027 | β₀=1, β₁=0 | 0.982 |
| Line arrangement | 0.288 | β₀=1, β₁=0 | 0.982 |
| Circle arrangement | 0.394 | β₀=1, β₁=0 | 0.978 |
| **No count signal** (3 seeds) | 0.336 ± 0.091 | β₀=1, β₁=0 | 0.981 |
| **No slots + no count** (3 seeds) | 0.344 ± 0.045 | β₀=1, β₁=0 | 0.981 |
| **Shuffled + starved** (3 seeds) | 0.367 ± 0.081 | β₀=1, β₁=0 | 0.985 |
| Random projection (3 seeds) | 0.326 ± 0.063 | β₀=1, β₁=0 | 0.983 |
| Multi-dim D=3 | 0.325 | β₀=1, β₁=0 | 0.954 |

The "information starvation" row (shuffled + starved) is the punchline: we masked grid slot assignments and the count scalar, *and* we permuted blob identities every frame so the model couldn't track individual objects. The number line still emerged. The only thing left for the model to detect was *that a gathering event had happened* — and that was enough.

Notably, the mean GHE barely moves across starvation conditions (0.329 → 0.367). What *does* change is the variance across seeds (0.027 → 0.081). The interpretation: auxiliary observation channels scaffold the *reliability* of learning, not the *geometry* of what gets learned — consistent with Vapnik & Vashist's Learning Using Privileged Information framework, and with Hu et al. (2024)'s Scaffolder result on DreamerV3 specifically.

*Sources: `artifacts/reports/ablation_multiseed_results.json`.*

### The untrained baseline

To distinguish architecture from learning, we ran an untrained DreamerV3 (random weights, zero gradient updates) on identical observation sequences:

| Metric | Trained | Untrained |
|--------|:-------:|:---------:|
| RSA ρ | 0.982 | 0.591 |
| PCA PC1 | 73.0% | 23.0% |
| GHE | 0.281 | 0.395 |
| β₀, β₁ | 1, 0 | 1, 0 |
| Arc R² | 0.998 | 0.985 |

The untrained baseline's arc R² = 0.985 deserves careful interpretation. The observation changes systematically with count — more blobs on the grid, bot in different positions — so *any* function of the observation, even one from random weights, will produce hidden states that evolve monotonically with count. Arc R² captures this architectural scaffold.

But the metrics that probe *representational quality* tell a different story. RSA drops from 0.982 to 0.591: the untrained representations are only weakly ordinal. PCA PC1 drops from 73% to 23%: the trained agent concentrates count into a dominant axis; the untrained agent spreads variance diffusely. The interpretation is that architecture provides a weak monotonic scaffold, which training transforms into a precise 1D manifold with uniform spacing, ordinal structure, and dimensional parsimony.

This parallels Kim et al. (2021), who showed untrained networks can have more individual number-selective units than trained ones (16.9% vs 9.6%). Unit-level selectivity and population-level geometry can dissociate — and manifold-level analysis captures what single-unit tuning misses.

*Source: `artifacts/reports/ablation_multiseed_results.json` (untrained_baseline section).*

### The random projection surprise

This was the most unexpected finding. We multiplied the 82-dim observation by a fixed random orthogonal matrix Q before passing it to the agent, preserving all pairwise distances but destroying every axis-aligned feature. The standard centroid-based metrics (GHE, topology, RSA) declared the two models equivalent. We almost stopped there.

Then we built a real-time visualization, watched the probe output wobble on the baseline and snap cleanly on the random projection, and realized the summary statistics were hiding something. We developed a new metric — **probe SNR**, the ratio of between-count centroid variance to within-count scatter — and found a real gap:

| Metric | Baseline | Random Projection |
|--------|:--------:|:-----------------:|
| GHE | 0.329 | 0.326 |
| Probe SNR | 502 | 825 |
| Live probe accuracy | 81% exact | 95% exact |
| PaCMAP R² | 0.651 | 0.976 |

Crucially, we also ran a **random permutation** condition — reordering observation dimensions without mixing them. Permutation preserves each feature's marginal distribution; it only disrupts *which index carries which feature*. If the mechanism were scatter isotropy from feature mixing, permutation should have no effect. Instead, permutation produced nearly the full benefit (PaCMAP R² = 0.953).

The mechanism isn't feature mixing. It's **coordinate-structure disruption**. The baseline observation format has strong semantic grouping: paired xy coordinates at adjacent indices, contiguous blocks of slot assignments, isolated count scalars. The RSSM's simplicity bias (Shah et al., 2020; Morwani et al., 2024) makes it build axis-aligned features that exploit this grouping, creating directional scatter around the counting backbone. Permutation and projection both disrupt the grouping, forcing the RSSM to extract count from the underlying temporal dynamics instead. The scaffold matters; the format is not neutral.

This is an instance of a well-documented phenomenon (Geirhos et al., 2020; Lee et al., 2020; Laskin et al., 2020), but it has an interesting methodological twist: **standard metrics missed it entirely**. Only multi-scale projection fidelity — sensitive to within-count scatter geometry — surfaced the difference. Summary statistics on centroids average the scatter away. This is probably a general lesson, not a counting-specific one: high-dimensional representation quality has scale-dependent structure that single-scale metrics cannot see.

### The successor function

"+1" isn't a single direction. When we compute per-count step vectors (μ_{c+1} − μ_c) for the counting manifold, we find it takes 11 PCA components to capture 90% of variance. The step from 0→1 and the step from 24→25 have cosine similarity near zero — they point in nearly opposite directions in 512-d space. What's conserved is the *geodesic magnitude*: the manifold maintains constant speed along a curving high-dimensional trajectory.

We also found the model *anticipates* transitions. Its hidden state begins shifting toward the next count 2–50 timesteps before a blob lands on the grid, with anticipation interval proportional to travel distance. And the RSSM prior (before incorporating the current observation) decodes count at R² = 0.956 — meaning the counting signal lives primarily in accumulated recurrent dynamics, not in the current observation. Predictive processing in miniature.

### What's strong, what's not

The counting manifold is the strongest result in the project. Five-seed replication with unanimous topology; nine-condition ablation with mean GHE within 0.04 of baseline under full information starvation; untrained control with dramatic RSA gap (0.982 → 0.591); cross-dimensional replication from D=2 through D=5 with matching Gromov-Wasserstein geometry across different neurons. This is probably publication-ready.

What's missing is **causal intervention**. We've shown a counting-correlated manifold exists and persists across conditions — not that the model *uses* it to make predictions. The Othello-GPT standard (Li et al., 2023; Nanda et al., 2023) requires editing the hidden state along the putative representation and verifying that downstream predictions change accordingly. The R² = 0.956 prior decoding is indirect evidence of functional integration, not a substitute.

Five seeds is also below modern RL best practice (Henderson et al., 2018; Agarwal et al., 2021). Topology is discrete so it's robust, but GHE intervals would benefit from more seeds.

---

## Part 2: The Binary Counting Machine

Same architecture. Same training objective. Different physics.

### Setup

The binary world replaces the 5×5 grid with a 4-bit binary register. Fifteen blobs, 2D arena. When the bot collects a blob, it triggers a cascade: the ones column fills, and when it reaches threshold it empties and the twos column increments, and so on up to 8s. The observation is 72-dimensional (blob positions, column heights, carry states). The world model is structurally identical — DreamerV3 RSSM, 512-dim h_t.

Incrementing 0→1 flips one bit: 0000→0001. Incrementing 7→8 flips four: 0111→1000. The physical process is fundamentally different from smooth gathering.

### A note on random baselines in this setting

One thing worth flagging up front: the binary observation format distributes count information across 4 correlated bit columns, and random temporal filtering — what an untrained GRU effectively is — preserves a lot of that signal just from the observation structure itself. This changed how we evaluate this half of the project. The headline metric from the counting paper (probe R²) is contaminated here: an untrained RSSM can score nearly as well as a trained one, not because it has learned anything, but because the binary display essentially contains the answer and random temporal filtering carries it through.

So for the binary world we lead with metrics that do survive untrained controls: exact count accuracy, Hamming RSA (organization around bit-flip distance, which random networks don't produce), and probe SNR. The binary evaluation is designed around the observation that **different observation statistics demand different metric protocols** — and this retroactively strengthens the counting-manifold interpretation, because the grid world concentrates count in 2 of 82 observation dimensions, meaning extracting it *requires* active learning.

### Type A vs Type B

The central comparison with the counting manifold:

| | Grid world (Type A) | Binary world (Type B) |
|-|:---:|:---:|
| Decimal GHE | 0.33 (smooth manifold) | 4.91 (not a manifold) |
| Dominant geometry | Ordinal distance | Hamming distance |
| Pairwise R² Hamming | — | 0.737 |
| Pairwise R² Decimal | — | 0.067 |
| RSA ordinal | 0.978 | 0.466 |
| Topology β₀ | 1 | 1 |

Both are connected. Both represent count precisely (exact accuracy 96% vs 100%). But the organizing distance is different: in the grid world, counts that are numerically close are representationally close; in the binary world, counts that differ by one *bit* are representationally close. 7 (0111) and 15 (1111) are close; 7 and 8 (1000) are far.

The representation didn't choose this for philosophical reasons. The world model builds the structure that's best for predicting next-state observations, and binary cascade physics are best predicted by tracking bit states, not decimal magnitude.

### The binary successor function

The counting-world successor was a smooth 11-PC rotation. The binary successor is something else entirely — a compositional decomposition into four orthogonal bit-flip directions.

**Step magnitude scales with carry depth** (r = 0.98):

| Carry depth | Example transitions | Mean magnitude |
|:-----------:|---------------------|:--------------:|
| 0 (simple flip) | 0→1, 2→3, ..., 12→13 | 5.86 |
| 1 (1-bit carry) | 1→2, 5→6, 9→10, 13→14 | 7.28 |
| 2 (2-bit carry) | 3→4, 11→12 | 8.58 |
| 3 (full cascade) | 7→8 | 9.40 |

Every simple flip looks identical (CV = 0.025).[^sampling] Every carry of a given depth looks identical. The representational displacement scales almost exactly with the number of bits that physically flip.

**Step vectors decompose linearly into four orthogonal bit-flip directions.** We trained per-bit linear probes and projected each step vector onto the four probe weight directions. Sign agreement across all 26 changed bits in all 15 transitions: **100%**. Cross-talk on unchanged bits: ~0.001. The model discovered four orthogonal "bit-flip" axes in 512-dim space and composes them linearly to represent any transition. The step vectors themselves require 5 PCA components to capture 90% of variance (cf. 11 PCs for the counting-world successor); the four bit-flip directions account for 16% of that variance. They are a small, clean, generalizing subspace within a larger step-vector structure whose dominant components lie orthogonal to the bit-flip axes and likely encode shared count-magnitude and transition-type information. What the model factored out cleanly was binary arithmetic, embedded in a representation that is mostly about other things.

[^sampling]: Count=15 is the terminal state and is sampled once per episode (~15 stable samples versus 500+ for other counts), which undermines centroid precision. Magnitude-based statistics (CV, step magnitude) therefore exclude 14→15; directional statistics (sign agreement, decomposition) include it.

**The model anticipates carries from the bit state.** We measured how early the hidden state begins shifting toward the next count:

| Carry depth | Anticipation onset |
|:-----------:|:------------------:|
| 0 (simple flip) | 18.3 ± 9.6 steps |
| 1 (1-bit carry) | 2.8 ± 0.5 |
| 2 (2-bit carry) | 5.5 ± 0.5 |
| 3 (full cascade) | 8.4 ± 0.5 |

The anticipation pattern is cascade-specific: deeper cascades start earlier (8 steps out for a full cascade). The model looks at the current bit state 0111, predicts that the next increment will require a 4-bit flip, and starts preparing before the transition.

### Imagination rollout: does the model simulate carries internally?

The strongest test of internal mechanism: can the model generate the sequential cascade without any observations? We forked the RSSM into imagination mode (prior only, no observation updates) 20 timesteps before each carry transition and measured whether probes trained on actual column states detected sequential bit flips in the autonomous dynamics.

For the 7→8 full cascade:

| Bit | Posterior (crossing time) | Imagination | Carry propagation (battery) |
|:---:|:---:|:---:|:---:|
| bit0 (↓) | −10.5 | −11.2 | −10.4 |
| bit1 (↓) | −6.5 | −6.9 | −6.4 |
| bit2 (↓) | −2.5 | −3.0 | −2.5 |
| bit3 (↑) | −0.5 | −0.9 | −0.5 |

Three independent measurements (posterior, imagination, battery carry propagation) agree within one timestep. The imagination generates the cascade at the same pace as the posterior — approximately 4 timesteps per carry phase, LSB to MSB, every depth sequential. The model doesn't just predict the endpoint of a carry; it internally simulates the *process*, bit by bit, in physical order.

A methodological note worth preserving: an initial imagination analysis used probes trained on decimal-count-derived labels, which only update after the full cascade completes. Those probes were blind to intermediate cascade states and compressed a 10-step sequential process into a 1-step jump, producing a false-negative "partial cascade" verdict. The corrected analysis used column-state probes matching the carry propagation methodology exactly. Probe calibration matters; the cost of getting this wrong is misdiagnosing active simulation as passive interpolation.

*Source: `artifacts/binary_successor/imagination_rollout_colstate.json`. Posterior span 10.02, imagination span 10.31, confirmed against raw data.*

### The thesis

The number seven is not a single geometry. In the gathering grid, seven is a position on a smooth curve. In the binary machine, seven is a vertex of a hypercube. Both are correct. Both are complete. Neither is "the" representation of seven.

What number *is*, in this framing, is whatever is common across all the geometries that different physical implementations produce. To find that commonality, you need to integrate across implementations — which brings us to the open question.

---

## Part 3: The Integration Problem (Preview)

The natural next question is whether a single system can maintain both geometric structures simultaneously. We explored this with two approaches, and the findings will appear in full in the formal paper. Here is a short preview of what we're seeing.

**Naïve combined training fails.** If you train one RSSM on a combined environment that shows both the grid display and the binary display, the world model allocates its representational capacity to whichever physics is harder to predict. Binary cascades, with their non-local carry dependencies, completely dominate — the combined model preserves near-perfect count accuracy and near-perfect cross-format transfer, but its representation geometry is organized around Hamming distance and the grid's smooth-manifold structure is effectively absent. Prediction complexity determines representation geometry.

**A frozen-specialist architecture (the "FP Unifier") succeeds.** We built a small integrator that freezes the two pre-trained specialist world models and passes their hidden states through learned adapters into a shared GRU that must reconstruct both observation streams. The pattern borrows from multi-modal fusion (Flamingo, ImageBind, LLaVA). What's different is the question being asked: rather than integrating different sensory modalities of the same physical event, the unifier integrates different physical implementations of the same abstract concept. The integrator preserves both ordinal and Hamming structure simultaneously — not at specialist quality, but both nonzero, which the combined-RSSM baseline cannot achieve.

**The central finding from the integration experiments is an accuracy-geometry dissociation.** Across a wide range of training interventions, per-bit task accuracy stays at 100% while the geometric organization of the unified representation reshapes dramatically. You can push the integrator toward ordinal-dominant, Hamming-dominant, or balanced regimes — and the model always knows the count, it just organizes that knowledge differently. This parallels Fascianelli et al.'s (*Nature Communications*, 2024) finding that two monkeys performing the same task at the same accuracy level exhibited strikingly different representational geometries in PFC. Our contribution is a controlled computational demonstration in which the geometric structure is systematically manipulated through training interventions, letting us identify which mechanisms determine which geometry emerges.

We also observed a surprising pattern in *timing*: a brief regularization pulse applied late in training (after the integrator's GRU had consolidated) outperformed the same pulse applied early, and explicitly freezing the GRU during the late pulse reversed the effect. This is a direction the established critical-learning-periods framework (Achille et al., 2019; Golatkar et al., 2019) does not predict — early training should be uniquely formative, not late. Our working explanation connects this to low-rank RNN dynamics (Mastrogiuseppe & Ostojic, 2018) and two-timescale stochastic approximation (Borkar, 1997), which together predict a genuine discontinuity at exactly zero backbone learning rate. A GRU with 5% residual plasticity and a fully frozen GRU are not the same system. We suspect this generalizes beyond counting, possibly to the LoRA/adapter ecosystem, and the formal paper will treat it as its own finding.

*Full numerical tables for the unifier experiments — contrastive-alignment sweep, VICReg timing comparison, GRU freeze test, subspace null model — are pending re-export from the cloud runs that produced them and will accompany the formal paper.*

---

## What the project found so far

1. **A number line emerges from gathering alone.** Five-seed replication, unanimous topology (β₀=1, β₁=0), RSA > 0.975, robust to full information starvation, architecture-independent in every condition we tested. *Causal intervention remains the main gap.*

2. **The random projection / permutation finding sharpens the story.** Coordinate-structure disruption — not feature mixing — cleans up meso-scale representation quality in ways invisible to standard centroid-based metrics. Probable general lesson about representation evaluation: summary statistics miss scale-dependent structure.

3. **Different physics produce different geometries, in the same architecture.** Grid world → smooth ordinal manifold. Binary world → Hamming hypercube with factored bit-flip axes. Same DreamerV3 RSSM, same training objective. This is the anima-bridge thesis.

4. **The binary successor decomposes compositionally.** Four orthogonal bit-flip directions, 100% sign agreement across all transitions, cross-talk ~0.001. The model cleanly factored binary arithmetic out of a larger step-vector structure.

5. **The binary model internally simulates carry cascades.** Imagination mode (no observations) reproduces the sequential LSB→MSB cascade at the same pace as the posterior. Active simulation, not passive interpolation.

6. **Preview of integration results.** Naïve combined training loses one geometry; a frozen-specialist-plus-adapter architecture preserves both. Across integration experiments, accuracy and geometric organization are formally separable — the same facts can live in the same representation at many different geometries. Full numerical results pending formal paper.

## Limitations worth stating clearly

- **No causal interventions for the counting manifold.** Everything in Part 1 is correlational. The Othello-GPT standard requires editing the hidden state along the number-line direction and verifying downstream predictions change. Not yet done. This is the single most important missing experiment.
- **Five seeds is below modern RL best practice.** Topology is discrete so it's robust; GHE confidence intervals would benefit from more seeds.
- **The binary evaluation needed its own metric protocol.** Standard metrics (probe R², per-bit accuracy, ordinal RSA) are contaminated by the observation structure. We lead with exact accuracy, Hamming RSA, and probe SNR, which survive untrained controls. These have not been independently validated beyond this project.
- **Integration results are preliminary.** Part 3 is a preview. Most unifier ablations are single-seed; the full tables and multi-seed replication will appear with the formal paper.
- **Scale.** Everything here is 0–25 count states. Generalization to larger state spaces is an open question.

## What comes next

- **Causal intervention on the counting manifold.** Edit h_t along the probe direction, measure downstream prediction shifts. Top priority.
- **Multi-seed replication of the integration experiments.** Directional replication exists for some conditions; magnitude stability across seeds remains to be established.
- **The formal paper.** Full unifier numerical tables, cooperative residual plasticity theory, and a broader discussion of the accuracy-geometry dissociation.

The thesis stands: same concept, different physics, different geometry. Counting turns out to be a surprisingly clean window into a general question about what it means for a learned representation to know something.

---

*Implementation by major-scale and Claude (Anthropic). Primary repo: [anima-bridge](../README.md). Paper draft (counting half): [counting_from_observation.md](../artifacts/paper/counting_from_observation.md). Extended draft covering the binary world and integration: [UNIFIER.md](../UNIFIER.md) (note: numerical tables in the binary/integration sections cite analysis outputs that are being re-exported from cloud runs; treat those specific numbers as preliminary until the formal paper).*
