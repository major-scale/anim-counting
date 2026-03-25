#!/usr/bin/env python3
"""
Symbolic Binary Environment — Token-Based 4-Bit Counter
=========================================================
A passive-observer Gymnasium environment that emits symbolic token
observations of a 4-bit binary counter (states 0–15).

Observation: 4 integer tokens from vocabulary {0, 1}, one per bit position.
    Bit ordering: LSB first (index 0 = ones), matching the physical binary
    specialist's column ordering.

Action space: Discrete(1) — zero-action, passive observer.
Reward: Always 0.0 — pure world modeling.

Episode structure: The counter increments every `steps_per_count` timesteps,
cycles through 0–15, and the episode ends after `n_cycles` full cycles.
This matches the physical specialist's ~20 steps per delivery, 15 deliveries
per episode (the physical env counts 0-15 with ~20 observation steps per
count transition).

The environment also exposes metadata (decimal count, bits, carry depth)
for evaluation, but these are NOT part of the observation.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants (matching physical binary specialist)
# ---------------------------------------------------------------------------

NUM_BITS = 4
VOCAB_SIZE = 2  # {0, 1}
MAX_COUNT = 2 ** NUM_BITS  # 16 (states 0-15, wraps at 16)

# Timing: physical specialist takes ~20 steps per delivery (phases of carry
# cascade), so we replicate that pacing to give the RSSM similar temporal
# structure to learn from.
DEFAULT_STEPS_PER_COUNT = 20
DEFAULT_N_CYCLES = 1  # 1 full cycle = 16 counts = 320 steps
DEFAULT_MAX_STEPS = DEFAULT_STEPS_PER_COUNT * MAX_COUNT * DEFAULT_N_CYCLES


def int_to_bits(n, num_bits=NUM_BITS):
    """Convert integer to LSB-first bit array."""
    return np.array([(n >> i) & 1 for i in range(num_bits)], dtype=np.int64)


def carry_depth(n):
    """Number of trailing 1-bits in n (= number of carries when incrementing n).

    Examples: carry_depth(0)=0, carry_depth(1)=1, carry_depth(3)=2, carry_depth(7)=3
    """
    if n == 0:
        return 0
    depth = 0
    while n & 1:
        depth += 1
        n >>= 1
    return depth


class SymbolicBinaryEnv:
    """Symbolic 4-bit binary counter environment.

    Compatible with DreamerV3's dict-observation API.
    """

    def __init__(self, steps_per_count=DEFAULT_STEPS_PER_COUNT,
                 n_cycles=DEFAULT_N_CYCLES, seed=0):
        self.steps_per_count = steps_per_count
        self.n_cycles = n_cycles
        self.max_steps = steps_per_count * MAX_COUNT * n_cycles
        self._rng = np.random.RandomState(seed)
        self._step = 0
        self._count = 0
        self._done = True

    @property
    def observation_space(self):
        """4 tokens from vocab {0, 1}."""
        # Use a simple box space for compatibility
        import gym
        import gym.spaces
        return gym.spaces.Dict({
            "tokens": gym.spaces.MultiDiscrete([VOCAB_SIZE] * NUM_BITS),
            "is_first": gym.spaces.Box(0, 1, shape=(), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=bool),
        })

    @property
    def action_space(self):
        import gym
        import gym.spaces
        return gym.spaces.Discrete(1)

    def _get_obs(self, is_first=False, is_last=False):
        tokens = int_to_bits(self._count)
        return {
            "tokens": tokens,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_last,
        }

    @property
    def metadata(self):
        """Non-observation metadata for evaluation."""
        return {
            "decimal_count": self._count,
            "bits": int_to_bits(self._count).tolist(),
            "carry_depth": carry_depth(self._count),
            "step": self._step,
        }

    def reset(self):
        self._step = 0
        self._count = 0
        self._done = False
        return self._get_obs(is_first=True)

    def step(self, action=None):
        if self._done:
            return self.reset(), 0.0, False, {}

        self._step += 1

        # Increment count at the boundary of each count window
        if self._step % self.steps_per_count == 0:
            self._count = (self._count + 1) % MAX_COUNT

        done = self._step >= self.max_steps
        self._done = done

        obs = self._get_obs(is_last=done)
        info = self.metadata
        return obs, 0.0, done, info

    def close(self):
        pass


# ---------------------------------------------------------------------------
# DreamerV3-compatible wrapper
# ---------------------------------------------------------------------------

class SymbolicBinaryWorld:
    """DreamerV3-compatible wrapper around SymbolicBinaryEnv."""

    def __init__(self, task="symbolic_binary", seed=0, **kwargs):
        import os
        steps_per_count = int(os.environ.get("SYMBOLIC_STEPS_PER_COUNT",
                                              DEFAULT_STEPS_PER_COUNT))
        n_cycles = int(os.environ.get("SYMBOLIC_N_CYCLES", DEFAULT_N_CYCLES))
        self._env = SymbolicBinaryEnv(
            steps_per_count=steps_per_count,
            n_cycles=n_cycles,
            seed=seed,
        )
        self._done = True

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        if isinstance(action, dict):
            action = action.get("action", 0)
        if self._done:
            return self._reset_obs(), 0.0, False, {}
        obs, reward, done, info = self._env.step(action)
        self._done = done
        return obs, reward, done, info

    def reset(self):
        return self._reset_obs()

    def _reset_obs(self):
        obs = self._env.reset()
        self._done = False
        return obs

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SymbolicBinaryEnv — Self-test")
    print("=" * 60)

    env = SymbolicBinaryEnv(steps_per_count=5, n_cycles=1)
    obs = env.reset()
    print(f"\nReset: tokens={obs['tokens']}, is_first={obs['is_first']}")

    transitions = []
    prev_count = 0
    step = 0
    done = False

    while not done:
        obs, reward, done, info = env.step(0)
        step += 1
        curr_count = info["decimal_count"]
        if curr_count != prev_count:
            transitions.append((step, prev_count, curr_count, info["carry_depth"]))
            prev_count = curr_count

    print(f"\nTotal steps: {step}")
    print(f"Transitions ({len(transitions)}):")
    for s, old, new, cd in transitions:
        bits = int_to_bits(new)
        print(f"  step {s:3d}: {old:2d} -> {new:2d}  bits={bits}  carry_depth={cd}")

    # Verify wrapping
    assert transitions[-1][2] == 0, f"Expected wrap to 0, got {transitions[-1][2]}"
    assert len(transitions) == MAX_COUNT, f"Expected {MAX_COUNT} transitions, got {len(transitions)}"

    # Verify bit encoding
    for c in range(MAX_COUNT):
        bits = int_to_bits(c)
        reconstructed = sum(b * (2 ** i) for i, b in enumerate(bits))
        assert reconstructed == c, f"Bit encoding failed: {c} -> {bits} -> {reconstructed}"

    print(f"\nAll {MAX_COUNT} transitions verified. Bit encoding correct.")
    print("=" * 60)
