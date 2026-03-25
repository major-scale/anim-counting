# A World Model as Mental Abacus: Autonomous Simulation of Binary Arithmetic from Prediction Alone

**major-scale**
Independent Researcher

## Abstract

A DreamerV3 world model trained on next-state prediction in a physical binary counting environment discovers the compositional structure of binary arithmetic and autonomously simulates carry cascades in imagination, passing through correct intermediate states at 1.03–1.07× the speed of the physical process. Systematic investigation (13 claims systematically tested, 7 alternative explanations eliminated) reveals a mildly expansive dynamical regime stabilized by sensory input — paralleling cortical dynamics — where observation removal produces immediate desynchronization rather than gradual degradation. The model learns a correct arithmetic algorithm embedded in a conditionally stable dynamical system: the computation is flawless, but the substrate requires continuous sensory grounding.

## The Binary Counting Specialist

A DreamerV3 RSSM (~12M parameters; 512-d GRU hidden state, 32×32 categorical stochastic state) observes a 2D world where a bot delivers objects to a physical 4-bit binary counter with visible carry cascades. The environment uses 15 blobs (counts 0–14) and propagates carries over multiple timesteps (~2 steps per column), making the sequential mechanism directly observable. The model learns purely from next-state prediction with no informative reward signal (reward is constant zero throughout).

**Compositional structure emerges without explicit instruction.** The hidden state develops four orthogonal bit-flip axes (one per counter column). Transition step vectors decompose as exact linear combinations of these axes: each bit contributes approximately ±1.5 units along its probe direction when it participates in a transition, and a mean of 0.010 when it does not — a mean participation ratio of 152:1. Step magnitude scales with carry depth at r=0.98 (Spearman). Mutual information analysis independently confirms this factoring: LSB and MSB encode through completely disjoint hidden dimensions (Jaccard overlap = 0.000), and count-level MI decomposes as the sum of bit-level MI (r=0.950). Five PCA components capture 90% of successor function variance — near the theoretical minimum for a 4-bit binary code.

All metrics survive a three-tier random-RSSM baseline protocol (3 untrained seeds on identical observation streams). Standard probe R² is contaminated — untrained RSSMs achieve R²=0.977 on count prediction because the 72-d binary observation distributes count information across correlated dimensions. Surviving metrics: exact count accuracy (100% trained vs 58.7% random), Hamming RSA (0.558 vs 0.337), probe SNR (6,347 vs 45). Without the random baseline protocol, per-bit accuracy (100% trained vs 99%+ random) would falsely suggest learned structure where architecture alone suffices.

**Autonomous simulation in imagination.** When the RSSM is forked to prior-only mode (no observations), it reproduces the full sequential carry cascade in correct physical order (LSB→MSB). Across 30 episodes:

|   Carry depth    |  N  | Posterior span | Imagination span | Ratio |
| :--------------: | :-: | :------------: | :--------------: | :---: |
| 0 (simple flip)  | 209 |   0.0 ± 0.0    |    0.0 ± 0.0     |   —   |
| 1 (1-bit carry)  | 90  |  2.00 ± 0.02   |   2.12 ± 0.10    | 1.06  |
| 2 (2-bit carry)  | 60  |  6.02 ± 0.04   |   6.41 ± 0.28    | 1.07  |
| 3 (full cascade) | 30  |  10.02 ± 0.05  |   10.31 ± 0.22   | 1.03  |

Posterior spans are extremely tight (std 0.02–0.05), reflecting observation-driven precision. Imagination spans show modest variability (std 0.10–0.28) but preserve sequential LSB→MSB ordering in every instance. Non-participating bits show <3.4% maximum deviation during imagined cascades — the model computes carry scope purely from internal state.

From count 7, all seven remaining transitions are imagined correctly across all 10 tested rollouts (single training seed). A reservoir computing control (Echo State Network: random fixed dynamics, trained readout only) achieves 0.988 probe accuracy on binary states but completely fails at autonomous imagination, drifting to chance within 2 steps — confirming that learned dynamics, not generic recurrent processing, underlie the RSSM's autonomous arithmetic. This mirrors Frank & Barner's (2012) finding that mental abacus experts pass through the same intermediate states as the physical device, exhibiting computational isochrony — mental simulation proceeding at approximately physical speed (Decety & Jeannerod, 1995).

**A methodological contribution: probe-representation alignment.** Initial probes trained on the decimal count variable (which updates only after cascade completion) showed an apparent "Outcome C: partial cascade — the model tracks but cannot simulate." Probes retrained on actual column states (which change step-by-step during cascades) revealed the full sequential simulation. Same model, same imagination — different probes, opposite conclusions. This parallels the Othello-GPT ontology lesson (Li et al., 2022; Nanda et al., 2023): probing world models requires matching the probe's target variable to the model's representational primitives, not derived summary quantities.

## Dynamical Characterization

**The controlled explosion.** Jacobian eigenvalue analysis at all 15 count centroids reveals spectral radius 1.05–1.31 everywhere except the terminal state (count 14, the last reachable state in this environment: spectral radius 0.998, the sole genuine fixed-point attractor). The system operates in a mildly expansive regime held on-manifold by continuous observation correction — consistent with trained recurrent networks converging toward the edge of chaos (Mastrovito et al., 2024; Schoenholz et al., 2017). This parallels cortical dynamics: excitatory recurrence is mildly supercritical, stabilized by inhibitory feedback and ongoing sensory input. Sensory deprivation in humans produces hallucinations within minutes (Ganzfeld effect); the RSSM's hidden state drifts off its learned manifold within steps of observation removal.

**A novel failure mode: the off-manifold fixed point.** With continuous observations: 96.2% step accuracy. With any interruption (peeks every 10 steps): 16.9%. This is not gradual compound error (Ross & Bagnell, 2011) — it is a binary phase transition consistent with desynchronization in driven chaotic systems (Pecora & Carroll, 1990; where observations serve as the coupling signal maintaining synchronization). Investigation reveals the mechanism is NOT gate closure (GRU update gates: 0.273–0.277 across normal, blind, and peek conditions — identical). The posterior computes corrections from out-of-distribution hidden state context after drift, producing small, wrong updates that converge to a stable off-manifold equilibrium: distance plateaus at 6.4–6.5 units from the correct centroid even after 20 consecutive observations. The model converges to a confident but wrong internal state from which observation-driven correction cannot recover. To our knowledge, this specific failure mode — convergence to a stable off-manifold fixed point from which observation-driven correction cannot recover — has not been previously characterized in the world model literature.

**Anticipatory destabilization.** Representational variance during idle periods scales with upcoming cascade depth (Spearman r=0.923, p<0.0001), concentrated along the LSB axis (the trigger dimension for every transition). This is NOT classical critical slowing down (Scheffer et al., 2009): autocorrelation tracks count position rather than depth (r=0.129, p=0.66), and perturbation recovery shows no significant depth dependence (r=0.145, p=0.62). The partial correlation controlling for idle duration remains significant (r=0.798, p=0.0006), confirming a depth-specific signal. The model's dynamics encode upcoming transition complexity through representational stability along the fastest-cycling dimension.

## Systematic Evaluation of Candidate Explanations

The mechanistic portrait — stable attractor → anticipatory destabilization → sequential cascade → scope-bounded termination → observation correction → terminal attractor — was developed through systematic testing of 13 candidate explanations, eliminating 7 alternative accounts in the spirit of Platt's (1964) Strong Inference:

| Claim tested                                  | Method                                                              |                          Verdict                           |
| :-------------------------------------------- | :------------------------------------------------------------------ | :--------------------------------------------------------: |
| RSSM simulates carry cascades in imagination  | Column-state probe comparison, prior-only rollout (N=30 episodes)   |                       **Confirmed**                        |
| Learned dynamics required for simulation      | Reservoir computing control (ESN: random dynamics, trained readout) | **Confirmed** (ESN decodes at 0.988, fails at imagination) |
| Bit axes encode through disjoint dimensions   | Mutual information analysis (512-d × 4-bit)                         |        **Confirmed** (Jaccard = 0.000 for LSB/MSB)         |
| Variance increases before deep transitions    | Idle-period variance vs carry depth across 15 count states          |                  **Confirmed** (r=0.923)                   |
| Count 14 is a terminal attractor              | Jacobian eigenvalue analysis at all 15 centroids                    |                  **Confirmed** (ρ=0.998)                   |
| Off-manifold drift causes observation cliff   | Drift trajectory + multi-peek recovery + GRU gate analysis          |                       **Confirmed**                        |
| AR1 autocorrelation tracks cascade depth      | AR1 vs carry depth correlation                                      |            Killed (tracks count order, r=0.129)            |
| Spectral radius predicts cascade depth        | Jacobian analysis at all 15 centroids                               |                Killed (uniformly 1.05–1.31)                |
| GRU gates close during blind periods          | Gate value comparison across 3 conditions                           |              Killed (0.273–0.277 everywhere)               |
| Multi-peek recovers on-manifold state         | Sequential peek recovery test (1–20 peeks)                          |                Killed (plateaus at 6.4–6.5)                |
| Non-normal amplification explains variance    | Henrici departure from normality vs depth                           |                 Killed (r=0.459, p=0.099)                  |
| Variance aligns with all flipping bits        | Per-bit directional variance projection                             |                   Killed (LSB dominates)                   |
| Variance-depth correlation is timing artifact | Partial correlation controlling idle duration                       |                 Killed (r=0.798 survives)                  |

## Discussion

The DreamerV3 RSSM discovers exact digital structure from analog physical experience — four orthogonal bit axes, sequential carry dynamics, scope-bounded imagination — through prediction alone. This instantiates the embodied cognition thesis for mathematics (Lakoff & Núñez, 2000) with mechanistic precision. The finding validates predictive coding's strongest claims (Clark, 2016; Friston, 2010): prediction error minimization yields discrete algorithmic structure, not just smooth dynamics or feature detectors. The autonomous imagination capability — running the counter forward without input — demonstrates a core mechanism of Clark's "Imaginarium" — autonomous forward simulation from learned dynamics — and connects to LeCun's (2022) proposal that world models are the foundation of autonomous intelligence. The fragility (observation cliff) adds essential nuance: the model's understanding is real but sensory-dependent, mirroring the brain's dependence on ongoing perception for stable cognition.

The 4-bit counter serves as a model organism — simple enough for complete characterization, generating general principles about how prediction-trained world models implement computation. The off-manifold fixed point applies to any RSSM-class world model. The controlled-explosion regime (mildly expansive, input-stabilized, with one terminal attractor) appears consistent with the natural operating point of trained recurrent networks.

**Limitations.** Single training seed, single architecture (GRU-based RSSM), 4-bit state space (15 states). The 5-bit generalization test — whether carry mechanisms extend to cascade depths never seen during training — and multi-seed replication are in progress.

## References

Belinkov (2022). _Computational Linguistics._ · Chughtai et al. (2023). _ICML._ · Clark (2016). _Surfing Uncertainty._ · Decety & Jeannerod (1995). _Behavioural Brain Research._ · Frank & Barner (2012). _J Exp Psych: General._ · Friston (2010). _Nat Rev Neurosci._ · Hafner et al. (2023). _JMLR._ · Lakoff & Núñez (2000). _Where Mathematics Comes From._ · LeCun (2022). _A Path Towards Autonomous Machine Intelligence._ · Li et al. (2022). _ICLR._ · Machamer, Darden & Craver (2000). _Philosophy of Science._ · Mastrovito et al. (2024). _bioRxiv._ · Nanda et al. (2023). _ICLR._ · Pecora & Carroll (1990). _Phys Rev Lett._ · Platt (1964). _Science._ · Ross & Bagnell (2011). _AISTATS._ · Scheffer et al. (2009). _Nature._ · Schoenholz et al. (2017). _ICLR._
