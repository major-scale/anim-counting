# Positioning Reference Reads

Verification pass on 7 papers cited in the DR positioning-and-novelty analysis for the DreamerV3/RSSM representation-geometry paper. For each entry: canonical source confirmed, abstract and (where needed) key methodology sections fetched, and any mismatch against the DR characterization flagged explicitly. The framing context is our compositional-coordinate-system result in a 4-bit counter RSSM (orthogonal per-bit directions; rowspace-only intervention matches full-direction effect under a Makelov-style test), contrasted with a smooth 1D manifold for decimal counting.

Verification summary: **6 verified, 1 mischaracterized (paper 3, author name error), 0 not found.** Several secondary hedges noted inline (DR over-sharpens MetaOthello's "rule-conflict" contrast; DR's "ICLR 2024" attribution for Quirke & Barez points at the shorter original paper, not the extended 2402.02619 arxiv version).

---

## 1. MetaOthello: A Controlled Study of Multiple World Models in Transformers (Chawla, Hall, Lovato, arXiv:2602.23164, Feb 2026)

**Status**: verified (with one hedge)

Chawla, Hall, and Lovato train small GPT models on a controlled suite of Othello variants that share syntax but differ in rules or token assignments, then probe how a single transformer organizes multiple world models. Headline findings: (a) transformers do not partition capacity into isolated sub-models — they converge on a mostly shared board-state representation that transfers causally across variants (linear probes from one variant successfully intervene on another's internal state); (b) for isomorphic (token-remapped) games the representations are "equivalent up to a single orthogonal rotation that generalizes across layers"; (c) for partially rule-overlapping variants, early layers stay game-agnostic, a middle layer identifies game identity, and later layers specialize. DR characterized (c) as "different organizations for rule-conflict variants" — this is directionally right but slightly over-sharp; the actual finding is a layer-depth-indexed divergence rather than wholesale different organizations. Contemporary controlled-comparison paper on representation geometry across task variants; inverse framing to ours (they hold architecture fixed and vary task; we hold architecture fixed and vary task continuum-vs-discrete).

**How to cite in our paper**: As contemporary controlled-comparison work — cite for the "orthogonal-rotation-between-isomorphic-tasks" finding and as the nearest transformer-side analog to our RSSM decimal-vs-binary contrast.

---

## 2. Revisiting the Platonic Representation Hypothesis: An Aristotelian View (Gröger, Wen, Brbić, arXiv:2602.14486, Feb 2026)

**Status**: verified

The paper argues that existing global representational-similarity metrics are confounded by network scale, such that larger models systematically inflate apparent cross-model convergence. They introduce a permutation-based null-calibration framework with statistical guarantees. After calibration, **apparent global spectral convergence across modalities largely disappears**, but **local neighborhood similarity** (though not local distances) retains significant cross-modality agreement. They pitch this as an "Aristotelian" refinement of the Platonic Representation Hypothesis: representations converge on shared local neighborhood structure rather than a unified global geometry. EPFL affiliation confirmed via Brbić's MLBio Lab page at brbiclab.epfl.ch; this matches the DR's noted proximity to Sobotka. Title, authorship, and key claims all match the DR summary. Useful for defensive framing: if a reviewer pushes "your binary-vs-decimal geometric difference could just be two routes to the same abstract quantity," Gröger et al. sharpens the tool to argue global-geometry claims need calibration before they ground abstraction-convergence arguments.

**How to cite in our paper**: Defensive citation against strong abstraction-driven-geometry-convergence readings of our binary-vs-decimal contrast.

---

## 3. Task structure and nonlinearity jointly determine learned representational geometry (Alleman, Lindsey, Fusi, arXiv:2401.13558, ICLR 2024)

**Status**: mischaracterized (author error)

**Correction**: The third author is **Stefano Fusi** (Columbia Neuroscience), NOT SueYeon Chung as DR asserts. The paper was published at ICLR 2024 (matches DR date). The actual thesis is narrower than DR implies: in one-hidden-layer networks, the activation function has an "unexpectedly strong impact" on representational geometry — Tanh networks learn representations reflecting target-output structure (more disentangled when targets are low-dim), while ReLU networks retain more raw-input structure, driven by ReLU's asymmetric saturation causing feature neurons to specialize regionally. So the title slogan "task structure and nonlinearity jointly determine..." is literally near-verbatim to our thesis, but the paper's actual mechanism is about activation-function choice shaping a Tanh/ReLU dichotomy in shallow feedforward nets — not the RSSM-scale claim that task physics (discrete vs continuous quantity) shapes recurrent-state geometry. The title collision is a real risk; the mechanistic distance is large.

**How to cite in our paper**: Must-cite-and-differentiate. The title is a near-miss; the contribution regime (one-hidden-layer feedforward, Tanh vs ReLU) is disjoint from ours (RSSM deterministic recurrent state, deep stochastic world model). Frame as complementary evidence that representational geometry has multiple orthogonal determinants.

---

## 4. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs (Driscoll, Shenoy, Sussillo, Nature Neuroscience 2024)

**Status**: verified

Driscoll, Shenoy, and Sussillo (Stanford) train RNNs on multi-task suites and identify recurring dynamical primitives — "dynamical motifs" — including ring attractors, point attractors, decision boundaries, and rotations, which the network reuses across tasks with shared computational demands. Concrete claims: (a) units cluster modularly by computational role (stimulus processing, memory, response); (b) tasks requiring identical sub-computations (e.g., circular-variable memory) share the same underlying dynamical landscape (same fixed points); (c) lesioning a cluster selectively impairs tasks that use that cluster's motif. This is the canonical "shared dynamical motifs mirror task structure" result as DR states, with a stronger compositional-reuse angle than DR flagged — the motif story is explicitly about reusable primitives for rapid transfer, which actually aligns well with our "compositional coordinate system" reframing. Relationship to our work: they identify motifs at the dynamics level (attractor topology); we identify an orthogonal linear coordinate system at the representation level. Different abstraction levels of a compatible claim.

**How to cite in our paper**: Must-cite-and-differentiate. Cite as the canonical precedent for "task structure shapes learned neural computation in RNNs," differentiate on granularity (motifs vs linear coordinate axes) and on evidential standard (they show dynamical-system structure; we show linear-probing + Makelov-clean orthogonal decomposition).

---

## 5. Not All Language Model Features Are One-Dimensionally Linear (Engels, Michaud, Liao, Gurnee, Tegmark, arXiv:2405.14860, ICLR 2025)

**Status**: verified

Engels et al. push back on the implicit one-dimensional linear-feature assumption behind much mech-interp work. They give a formal definition of irreducible multi-dimensional features (features that cannot be decomposed into independent lower-dimensional ones), use sparse autoencoders on GPT-2 and Mistral 7B to automatically discover candidate multi-dim features, find interpretable **circular** features for days of the week and months of the year, and run causal intervention experiments on Mistral 7B and Llama 3 8B showing these circles are actually used for modular-arithmetic-over-days/months tasks (not epiphenomenal). Methodology stack: SAE-based feature discovery plus activation-patching-style causal tests to establish functional role. This is genuinely the nearest methodological neighbor for our work as DR states: both do compositional causal features, both test functional role (not just decoding accuracy). Distinction to emphasize: their composition is a 2D circle per concept; ours is a 4-axis orthogonal frame that survives a Makelov rowspace-only test — a stronger causal-decomposition guarantee.

**How to cite in our paper**: Nearest methodological neighbor for compositional causal features. Differentiate on (a) orthogonal decomposition across axes, (b) Makelov-style rowspace isolation.

---

## 6. Understanding Addition in Transformers (Quirke & Barez, ICLR 2024, arXiv:2310.13121; extended as Quirke, Neo, Barez, arXiv:2402.02619)

**Status**: verified (with citation caveat)

The ICLR 2024 poster is by Philip Quirke and Fazl Barez (matches DR's attribution). The extended arxiv paper 2402.02619 "Understanding Addition and Subtraction in Transformers" adds Clement Neo as coauthor. The DR's "ICLR 2024" date pins to the two-author original. Content: one-layer three-head transformer model performs 5-digit integer addition via parallel per-digit streams with position-specific sub-algorithms; the authors mechanistically identify carry circuits — crucially, the 1-layer model can cascade a carry across one digit but fails when carries cascade through three or more positions, exposing the circuit's algorithmic depth. The expanded version unifies addition and subtraction under a cascading-carry/borrow-circuit account and achieves 99.999% accuracy with systematic ablations and node-level constraints validating the mechanism. This is the correct closest prior art for carry-as-internal-variable and feedforward-carry-circuit interpretability.

**How to cite in our paper**: Closest prior art for carry-as-internal-variable in a trained transformer. Differentiate on (a) architectural context (we're in an RSSM, they're in a 1-layer transformer), (b) orthogonality claim (we demonstrate orthogonal frame across four independent bits), (c) Makelov-test passage.

---

## 7. A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments (Wu, Geiger, Huang, Arora, Icard, Potts, Goodman, arXiv:2401.12631, Jan 2024)

**Status**: verified

Wu et al. defend distributed-alignment-search-style subspace interchange intervention methods against Makelov et al.'s "interpretability illusion" critique. Their central technical claim, directly relevant to our Makelov-test framing: a so-called illusion in Makelov's definition arises **essentially only where the distributed interchange intervention induces representations not orthogonal to the nullspace** — but this non-orthogonality is expected and unavoidable because "the space of possible inputs to the network leads to variation within a given set of neurons that covers some data-induced submanifold [which] need not, and generally will not, span the whole activation space." So null-direction causal efficacy is a simple fact about how networks use the data-induced submanifold, not a pathology. They additionally show their own intuitively-correct direction is classified as illusory by Makelov's criteria, arguing the detection framework is overbroad. DR's characterization ("argues null-space effects are expected") is accurate. For our paper this means we must carefully articulate why our rowspace-only result is evidentially strong *despite* Wu et al.'s point — our argument runs the other direction: we are not defending that a null-space-touching direction is real, we are showing the rowspace-only intervention already fully recovers the effect, which is strictly stronger than what Makelov's test requires and immune to the Wu et al. counterpoint.

**How to cite in our paper**: Defensive citation in the Makelov-passage section. Acknowledge Wu et al.'s point that null-space effects can be expected, then argue our test is evidentially stronger because rowspace-only intervention alone recovers the full-direction effect — we are not leaning on the ambiguous "null effect is expected" defense.
