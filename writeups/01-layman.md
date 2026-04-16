# Counting Without Being Taught

## An AI learned what numbers are by watching a robot pick up blocks

**What if you could watch a mind discover numbers for the first time — and see exactly how it happened?**

That's what this project is. We built a tiny artificial world, put a simple AI inside it, gave it one job — predict what happens next — and waited. No one taught it to count. No one told it what a number was. No labels, no lessons, no vocabulary. Just the visual experience of a robot picking up objects one at a time.

And then, inside the AI's memory, something appeared.

A number line.

Not a metaphor. A literal, measurable, geometrically precise number line — a curve in the AI's internal representation space where each point corresponds to a count, and where the distance between "3" and "4" is exactly the same as the distance between "17" and "18." Like marks on a ruler.

We didn't put it there. The AI built it on its own, from raw experience.

---

## The setup

The world is almost comically simple. A flat 2D field contains 25 colored blobs. A small robot roams around, picks them up one at a time, and drops each one onto a 5×5 grid of target slots. When all 25 blobs are placed, the episode ends.

Watching the world is an AI system called a "world model." Its only goal is to predict what the world will look like a moment from now. If the robot is about to grab a blob, the world model should predict where the blob will be next frame. If a blob is about to land on the grid, the world model should predict which slot.

That's it. There's no reward for counting. There's no lesson that says "this is three, this is four." There's no symbolic instruction of any kind. The world model is just trying to get good at predicting pixels — well, not pixels exactly, but a vector of numbers describing positions.

And yet.

Inside its memory — a 512-dimensional internal workspace where it keeps track of what's happening — we found that one particular shape had emerged. When we plotted the AI's memory states colored by how many blobs had been placed, the states lined up into a smooth curve. A curve where position 0 was at one end, position 25 was at the other, and every integer in between sat neatly between its neighbors.

A line. An internal number line, self-assembled, in an AI that had never been told numbers exist.

## Measuring what it built

"The AI kind of learned counting" is the sort of claim that's easy to make and hard to defend. So we threw the full toolkit at it.

- **Topology** (the branch of math that studies shapes in the abstract): we asked whether the structure was really a line, or a circle, or a blob, or a disconnected mess. Verdict: one connected piece, no loops. Consistent with a line.
- **Geometry** (the branch of math that studies distances): we asked whether the spacing was uniform — whether the distance from "5" to "6" matched the distance from "20" to "21." Yes, very closely. Our measurement said the arc was 99.8% linear.
- **Ordering**: we asked whether the curve preserved numerical order — whether "3" was between "2" and "4." Yes, with 98% fidelity.

And we did this across five separate training runs with different random starting conditions. Every single one built the same structure. The AI wasn't finding a lucky accident. It was reliably constructing a number line.

We also tried to break it. We masked out the part of the observation that literally said how many blobs were on the grid. The number line still emerged. We also hid which grid slot each blob went to. Still emerged. We even scrambled the blob identities so the AI couldn't track individual objects from frame to frame. Number line still emerged. The only thing the AI could still sense was *that something had been gathered* — and that was enough.

Counting, it turned out, doesn't need to be taught. It just needs to be gathered.

## The strangest finding

Here's the thing we didn't expect.

We tried another test where we mixed up the observation vector — multiplied it by a random rotation matrix that preserves all the information but scrambles which number means what. Now, instead of "the first dimension is the robot's x-position," every dimension was a random blend. From the AI's perspective, its senses had been thrown into a blender.

The number line got *better*.

Not worse. Better. The AI's count tracking jumped from 81% accuracy to 95%. When we watched it work in real time, the version with scrambled senses snapped cleanly from one number to the next, while the original version wobbled at transitions.

Why? Because the original observation format came with shortcuts. The AI could tell a single number like "position X" apart from "the count" by its index in the input vector, and it exploited that shortcut to do a half-lazy job. When we removed the shortcut — when every piece of information was mixed into every dimension — the AI was forced to do the real work of extracting the count from the dynamics of the scene. And that forced it to build a cleaner number line.

This turns out to be a well-known phenomenon in machine learning (it's called "shortcut learning"), but seeing it show up here, in counting, with a clean measurement that let us quantify it — that was a small, pleasing surprise.

## The second world

Then we asked: what if the physics were different?

We built a second world where counting worked not by a continuous gathering process but by binary cascading. Same kind of world model, same kind of training. Same concept — "how many" — but implemented as a 4-bit register where each blob triggers a carry propagation. Going from 7 to 8 means flipping all four bits at once: 0111 → 1000. Going from 0 to 1 means flipping just one.

Inside this new world model's memory, a structure emerged. But it was completely different.

Where the first AI built a smooth curve, this one built something closer to a hypercube — a structure organized by **Hamming distance**, which is just a fancy way of saying "how many bits are different." In its internal representation, 7 (0111) and 15 (1111) were *closer* than 7 and 8 (1000), because 7 and 15 differ in only one bit while 7 and 8 differ in all four.

Same architecture. Same training objective. Same concept. Totally different shape.

We took it further. We tracked the "+1 operation" inside this second AI's memory and found that it had independently invented a coordinate system for binary arithmetic: four orthogonal axes inside the 512-dimensional space, one for each bit. When the AI internally simulated going from 7 to 8, it flipped all four bits in the correct order — the bottom bit first, then the next, then the next, then finally the top bit — just like a physical cascade propagating through a register. The AI had built a model of how binary addition *mechanically works* inside its own head.

We even tested whether the AI was just reading these bits from the observation or actually simulating them. We cut off the observation entirely and asked the AI to imagine what would happen next, based purely on its internal dynamics. It produced the same cascade — in the same order, at the same pace, within a single timestep of the real thing. The model wasn't reacting to binary arithmetic. It was *doing* binary arithmetic.

## Same concept, different shapes

This is the central finding of the project, and it's a strange one: the number seven is not a single shape.

In the gathering world, seven is a point on a smooth curve. In the binary world, seven is a vertex of a hypercube. Both are correct. Both are complete. Both were built by the same architecture trying to do the same job. The only thing that differed was the physics of *how* counting happened in that world.

The geometry followed the physics. That's the thesis.

Which raises a question: if a single mind encountered both worlds, could it build a representation that held both shapes at once?

We tried. Just training one AI on both worlds at once didn't work — the binary physics, being harder to predict, completely dominated and overwrote the smooth-curve structure. The AI learned both, but organized itself around only one.

So we built a separate architecture — two pre-trained specialists (one from each world) feeding into a small shared integrator. The integrator's job was to reconstruct *both* worlds from a single unified memory state. And it worked. The final representation preserved both the ordinal structure of the gathering world and the Hamming structure of the binary world simultaneously.

Along the way we discovered some things we didn't expect:

- **Accuracy and geometry are separable.** You can dramatically reshape the internal geometry — making it more ordinal, more binary, or more balanced — and the AI's counting accuracy doesn't budge. It knows the count either way. It just organizes that knowledge differently. (A parallel finding, in monkey brains, was published in *Nature Communications* in 2024 — two monkeys doing the same task at the same accuracy had very different neural geometries.)

- **Timing matters more than strength.** A brief geometric nudge applied late in training — after the integrator had already settled — reshaped the final representation more effectively than the same nudge applied early or continuously. This contradicts standard "critical learning period" theory, which predicts early interventions dominate.

- **You need a little residual flexibility.** When we fully froze the integrator and tried to reshape its geometry from outside, the whole thing collapsed instead of expanding. But when we left it 5% flexible, it expanded beautifully. The "almost frozen" state is qualitatively different from "completely frozen." We named this phenomenon **cooperative residual plasticity** — it had been predicted by several branches of math and observed in several biological systems, but no one had given it a name.

## What it means (cautiously)

A few things this is, and a few things it isn't.

It *is* evidence that structured internal representations of number can emerge from embodied experience alone, without symbolic instruction. This is consistent with a long tradition in cognitive science — Dehaene's *Number Sense*, Lakoff and Núñez's *Where Mathematics Comes From*, Spelke's core knowledge theory — that says mathematical cognition is rooted in physical experience, not in abstract symbol manipulation. The AI's story is a clean, controlled version of that theory.

It *isn't* evidence that our AI understands numbers the way a human does. The representation it builds is uniformly spaced, while human number sense is logarithmically compressed (bigger numbers feel closer together). Our AI's counting is tied to its specific physical setup and doesn't transfer zero-shot to new counting tasks. It's more like a very early, very grounded proto-number-sense than the full thing.

And there's a scientific caveat: everything we've shown is *correlational*. We've shown a counting-shaped structure exists in the hidden state, reliably, across many conditions. We haven't yet shown that the AI *uses* this structure to make predictions (as opposed to merely *containing* it). Doing that requires a more invasive experiment called a causal intervention — reaching into the model's memory, tweaking it along the number-line direction, and checking whether the downstream predictions change. That's next.

## Why it's interesting

Because counting is one of those things humans take for granted until you try to explain where it comes from. It feels symbolic. It feels taught. It feels like something that arrives through language and culture. But here's a machine that never had language or culture, that was never rewarded for counting, that didn't even have a word for "number," and it built a number line anyway — just from the experience of watching objects be gathered one at a time.

More than that: it built *different* number lines depending on how the gathering worked. And when forced to hold two of them at once, it revealed that the shape of a concept and the knowledge of a concept are separable things — the same facts can live in the same mind in many different geometries.

That's not a statement about AI. It's a statement about what it might mean, computationally, to know something.

---

*The counting work is described in the [main repo README](../README.md). The binary world and the FP Unifier are described in [UNIFIER.md](../UNIFIER.md). A formal paper draft is in [artifacts/paper/counting_from_observation.md](../artifacts/paper/counting_from_observation.md).*
