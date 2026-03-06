# Anim: Multi-Dreamer Cognitive Architecture — Vision Document

**Status:** Future work. Not for immediate implementation. Captures concepts, principles, and experimental designs to guide development after counting paper (CCN 2026) and multi-sensory experiments are complete.

**Last updated:** March 2026

---

## The Core Idea

Train separate small DreamerV3 agents that each learn to see one thing. The classifier sees groups. The counter sees quantities. The adder sees totals. Each one dreams its own simplified version of reality through its trained lens. Those dreams flow between agents — the counter looks at the classifier's dream and counts what it sees there.

The real world stays unchanged. The agents project interpreted versions of reality. Mathematical cognition emerges from the composition of simple specialists, not from one monolithic model.

Nobody has built this. A comprehensive search (150+ targeted queries across academic and non-academic sources) confirmed the architecture is genuinely novel. Fragments exist — shared workspaces between modules (Global Latent Workspace), chained policies (CompoNet), inter-Dreamer communication (CoDreamer) — but nobody has composed multiple world models specialized by cognitive competency into a pipeline where Agent A's dreamed world becomes Agent B's observation.

---

## The Functional Programming Analogy

Each Dreamer agent is like a pure function from observation to dreamed world. Typed inputs, typed outputs, no hidden side effects, composable because each one transforms data cleanly.

```
classifier(messy_world)  → sorted_world
counter(sorted_world)    → quantities_per_group
adder(quantities)        → grand_total
```

The composition is function composition: `adder(counter(classifier(world)))`. The typed interfaces (latent state format, structured outputs with uncertainties) are literally type signatures ensuring output of one agent matches expected input of the next.

---

## Self-Organization: Why We May Not Need a Coordinator

The key insight: you don't need a manager directing traffic. You need a principle. The principle is prediction error minimization — each agent attends to whichever input reduces its own prediction error.

**How it works in practice:**

1. All agents receive the raw messy environment as their primary observation
2. Each agent also receives every other agent's output as optional additional observation channels
3. Early in training, upstream agents produce garbage latent states — downstream agents learn to ignore them
4. The classifier succeeds first because grouping is the easiest prediction problem
5. Once the classifier's dreamed world is a useful signal, the counter notices that attending to it produces lower prediction error than parsing the raw mess
6. The counter naturally upweights the classifier's channel — not because anyone told it to, but because pre-organized input is more predictable
7. The hierarchy assembles itself through prediction error minimization

**Why this should work:** This is exactly what happened in our ablation experiments. The agent learned which observation channels to attend to without being told. Removing channels didn't break counting — the agent adapted. The multi-agent version is the same principle one level up. Each agent is an observation channel for every other agent.

**Why circular dependencies are unlikely:** The competencies have a natural causal order. Grouping doesn't require counting. Counting doesn't require adding. The dependencies are directional because the mathematical concepts are directional. The DAG structure isn't imposed — it's inherent in the mathematics.

**The most exciting possibility:** Agents discover dependencies we didn't anticipate. Maybe the adder finds the classifier's output useful directly, skipping the counter. Maybe the classifier benefits from the counter's output — "I should regroup because the counts don't make sense." Emergent top-down feedback without engineering it. We'd discover the cognitive architecture rather than design it.

---

## Communication Between Agents

### The Sequential Presentation Hypothesis

The counter learned to count by tracking a sequential process — blobs gathered one by one. Its manifold is a 1D line parameterized by accumulation over time. That architecture is optimized for sequential input.

For sub-type counting (count the red ones, count the blue ones, count the green ones), dumping all three groups at once requires the counter to do something it's never learned — parse a structured multi-part message. But if the classifier presents one group at a time, the counter does exactly what it already knows, three times.

This is compositional reuse in the most literal sense. The counter function gets called three times with different inputs rather than once with a complex input. And it's how humans naturally count multiple groups.

**The manifold traversal view:** The composed system `adder(counter(classifier(world)))` isn't doing anything conceptually new. The counter walks its existing 1D manifold — the arc parameterized by accumulation over time. It just walks it three times: once for the red group, once for blue, once for green. Each walk is the same learned skill applied to a different input. The classifier's only job is to provide clean inputs for each walk. The adder then combines the endpoints of three manifold traversals. This is function reuse, not transfer learning — the exact same geometric structure serves all three counting tasks.

### Three Communication Options (Escalating Complexity)

**Option 1 — Dumbest possible (try first):**
Classifier outputs its full latent state every timestep. Counter receives it as concatenated observation channel. No scheduling, no turn-taking, no coordinator. See what happens. If the counter uses the classifier's signal at all, that's the first win. If temporal structure emerges in how it attends — that's evidence of spontaneous coordination.

**Option 2 — Simple scheduler:**
Round-robin presentation. The classifier filters its output to show one group at a time. Group A for N steps, then group B, then group C. One line of code. The counter processes each group sequentially using its existing counting skill unchanged.

**Option 3 — Emergent turn-taking:**
The classifier learns when to switch groups based on the counter's state. When the counter's hidden state stabilizes (finished counting current group), the classifier presents the next group. Neither agent is programmed to take turns — the dynamics naturally produce sequential processing because it minimizes both agents' prediction error.

### The Intermediary Coordinator (Only If Needed)

If simple options fail, consider a small coordinator Dreamer whose reward signal is effective communication between the other agents. But try without it first — emergent coordination is both simpler and more scientifically interesting.

**Principle: Escalate complexity only when simpler approaches fail.** Start with the dumbest version, measure, understand, add complexity where measurements say you need it.

---

## Communication Mechanisms (from DRQ findings)

### Ranked Options

1. **Latent state + learned alignment layers** (CoDreamer/LatentMAS pattern): Highest bandwidth, lowest loss. Requires alignment training. LatentMAS showed 70-84% token reduction vs text-based communication. A single latent step carries information equivalent to hundreds of tokens.

2. **Structured intermediate representations:** Attention maps, object features, typed outputs (class labels with confidences, integer counts with uncertainty). Inspired by Neural Module Networks. Most interpretable.

3. **Decoded dream + conditioning augmentation:** Agent B literally watches Agent A's reconstruction. Most interpretable. Significant information loss at each step. Requires mixed training (real observations + imperfect reconstructions). Cascaded diffusion model research says conditioning augmentation is essential to prevent error accumulation.

4. **Information-bottleneck-optimized messages:** Principled but limits expressiveness. IMAC showed bandwidth-constrained agents learn maximally informative protocols.

**Practical recommendation:** Use latent transfer between jointly trained agents. Use decoded dreams as universal fallback interface for agents that were never trained together.

---

## The Hardest Problem: Representation Alignment

Independently trained agents develop incompatible internal representations even with shared observation formats. The classifier's latent state for "three groups" means nothing to the counter unless their representations are aligned.

### Training Stability During Composition

The core risk: when Agent B starts attending to Agent A's output, Agent A's gradients change (its output now matters downstream), which shifts its representations, which breaks Agent B's learned attention patterns. This is the co-adaptation instability problem.

**Staged composition training (recommended first approach):**

1. **Phase 1 — Train agents independently.** Each agent masters its competency in isolation.
2. **Phase 2 — Freeze Agent A, train Agent B.** Agent B learns to use Agent A's frozen output as an additional observation channel. No co-adaptation — A's representations are static targets.
3. **Phase 3 — Unfreeze and fine-tune jointly** with a lower learning rate. Both agents adapt to each other, but from a stable starting point where B already knows how to read A's output.

This is the same staged approach used when fine-tuning any pretrained model — freeze the base, train the head, then optionally unfreeze end-to-end. Simple, proven, avoids instability entirely in the critical first phase. The DEQ fixed-point iteration for top-down correction is elegant theory but unnecessary complexity for the first experiment. Try the dumb version first.

### Mitigations (implement from the start)

1. **Shared frozen encoder:** All agents use the same perceptual front-end. Downstream representations diverge as agents specialize, but the input representation is anchored.

2. **Standardized RSSM format:** 32x32 categorical stochastic state + 512-dim GRU hidden state across all agents. This is the DreamerV3 default — locking it in is free.

3. **Typed interfaces:** Each agent outputs:
   - Full latent state (h_t, z_t) for high-bandwidth communication
   - Structured typed output (class labels, counts, uncertainties) for interpretability
   - Uncertainty/precision signal for downstream weighting

4. **Composition rehearsal during individual training:** Occasionally feed agent's output through a frozen composition pathway to maintain interface compatibility.

---

## Uncertainty Propagation

DreamerV3's 32x32 categorical latent state naturally represents multimodal uncertainty. When the classifier is unsure whether objects form 3 or 4 groups, the relevant categorical dimensions can be bimodal. The 1% uniform mixing (unimix) prevents overconfidence.

This connects to probabilistic population coding in neuroscience (Ma et al., 2006): uncertainty is encoded in population response gain. Two population codes combine optimally through addition weighted by inverse variance. Multi-Dreamer analog: compose agent outputs by concatenating or adding categorical distributions weighted by entropy-derived confidence.

A flat categorical from the classifier tells the counter "I'm not confident about this grouping" without any additional uncertainty channel.

---

## Top-Down Correction

"Your grouping must be wrong because the numbers don't add up."

This is precisely what predictive coding describes (Rao & Ballard, 1999). Higher areas predict lower-level activity. Discrepancies propagate downward as prediction errors.

### Implementation via Deep Equilibrium Models

Formulate the entire pipeline as a fixed-point iteration:

```
z = [classifier_output, counter_output, adder_output]
z_new = f(z_old, observation)
```

Iterate until convergence (3-5 iterations with damped updates). Convergence is guaranteed under contraction mapping if each agent's Jacobian spectral norm < 1.

This mirrors the brain: 2-5 recurrent loops for visual processing, feedback connections outnumber feedforward connections (Rockland, 1997), feedforward and feedback use separate frequency bands.

---

## Design Decisions to Lock In Now

These are free — just standardization choices that enable future composition.

1. **RSSM format:** 32x32 categorical + 512-dim GRU for all future agents
2. **DAG composition, not linear pipeline:** Every agent can access raw observations (skip connections), not just upstream outputs
3. **Typed interfaces from the start:** Latent state + structured output + uncertainty signal
4. **Shared encoder backbone:** Freeze during individual training, fine-tune during composition
5. **Diagnostic probes per agent:** Verify each agent's competency independently before attempting composition

---

## Developmental Curriculum

Validated by both developmental psychology and computation:

1. **Classification/Sorting** — perceptual grouping by color, size, proximity
2. **Counting** — sequential enumeration (ACHIEVED)
3. **Conservation** — quantity invariance under rearrangement
4. **Addition** — merging groups, accumulating totals
5. **Subtraction** — removing from groups, computing differences

Each competency is prerequisite for the next (Sarnecka & Carey, 2008; Stanford DREME project). Build agents in this order. Each produces an independently publishable result while building toward the composed architecture.

---

## Minimum Viable Experiment

**Environment:** Colored blobs (red, blue, green) scattered in 2D. Task involves sorting by color and counting per group.

**Agents:** Two DreamerV3 instances — classifier and counter.

**Setup:** Shared frozen encoder. Each agent receives raw observation + the other agent's latent state as additional observation channel. Train simultaneously.

**Measurement:**
- Does the counter develop dependency on the classifier's output? (Ablate classifier output and measure counting degradation)
- Does any temporal structure emerge in how the counter attends to the classifier? (Attention analysis over time)
- Does the classifier benefit from the counter's output? (Unexpected top-down dependency)
- What does the classifier's manifold look like? (Discrete clusters vs continuous structure?)

**Estimated cost:** Under $5 on RunPod. One environment, two agents, 300K training steps.

**Infrastructure reuse:** This experiment does NOT require a new environment from scratch. Add a color attribute to blobs in the existing counting environment (`counting_env_pure.py`). Train a classifier Dreamer on a color-sorting variant of the same environment. Everything else — the pure Python env, RunPod training pipeline, GHE evaluation, the full measurement battery — is ready. The infrastructure investment from the counting paper pays dividends here.

**Connection to random projection ablation (March 2026):** If the counting manifold survives orthogonal scrambling of the observation vector (preserving distances, destroying spatial semantics), that's evidence the RSSM extracts counting from trajectory dynamics, not spatial features. That same property makes latent-state communication between agents more plausible — if Agent A's output doesn't need to be spatially interpretable, just geometrically structured, then representation alignment is easier than feared.

**Either outcome is informative:**
- Counter uses classifier's dream → emergent cognitive composition (genuine first)
- Counter ignores classifier → raw world is sufficient even for complex environments (extends robustness finding)

---

## Scaling Argument

400 specialized 12M-parameter modules = 4.8B total parameters, potentially covering far more capability than a single 4.8B monolithic model. This is MoE (Mixture of Experts) applied at a higher level of abstraction. Mixtral and Switch Transformer already validate that sparse modular computation scales better than dense monolithic computation.

Difference: MoE selects experts (sparse activation). Multi-Dreamer composes all agents simultaneously (dense composition). The compositional scaling argument favors modularity as competencies multiply — a monolithic model must represent all competency combinations internally, while modules combine combinatorially.

---

## The Neuroscience Case

The brain does not build one unified world model. It maintains multiple parallel representations:

- V1 sees edges, V2 sees shapes, V4 sees objects, IT sees categories
- All run simultaneously, all feed each other
- More connections go backward (top-down) than forward (bottom-up)
- Different levels operate at different timescales
- Binding happens through mechanisms as simple as superposition (linear addition)

The multi-Dreamer architecture mirrors this directly. Each agent is a cortical area specialized on one aspect of perception. The self-organization through prediction error minimization is how the brain assembles its hierarchy. The top-down correction through iterative convergence is how the brain resolves conflicts between levels.

---

## Connection to Other Anim Workstreams

**Multi-sensory modalities:** Each sense (spatial, auditory, graph-based) is an independent observation channel. Multi-Dreamer composition is multi-sensory integration at the cognitive level rather than the perceptual level.

**Symbol emergence:** A society of specialized agents that each understand one mathematical operation creates the combinatorial relational structure that the compositional horizon literature says is necessary for compositional symbols to emerge. Counting alone produces holistic labels. Classification + counting + addition creates the compositional pressure.

**Linguistic bootstrapping:** The multi-agent architecture provides natural communication pressure. Agents that need to share information develop shared protocols. Those protocols are proto-symbols grounded in geometric representations.

**Predictive processing framework:** Every component traces back to prediction error minimization. The self-organization, the communication, the top-down correction, the uncertainty propagation — all are consequences of each agent minimizing its own variational free energy. DreamerV3's ELBO is formally equivalent to negative free energy, making this not analogy but mathematical identity.

---

## Summary

The multi-Dreamer architecture is the capstone vision for Anim: mathematical cognition emerging from the composition of embodied specialists, organized by prediction error minimization, communicating through dreamed worlds, with the hierarchy assembling itself because abstraction reduces surprise. Every component has been independently validated. The composition is the novel contribution. Start with two agents, test the principle, and let the science tell you what to build next.
