#!/usr/bin/env python3
"""
Rich Symbolic Binary Environment — Token-Based 4-Bit Counter with Carry Dynamics
==================================================================================
Variant B of the symbolic binary specialist experiment.

Like symbolic_binary_env.py (Variant A), this emits 4-token observations from
vocabulary {0, 1}. The difference: during count transitions, individual bit flips
are sequenced over multiple timesteps LSB→MSB, mirroring the physical specialist's
carry cascade timing.

Physical specialist carry timing (CARRY_STEPS_PER_PHASE=2):
  - Simple flip (depth 0, e.g. 0→1): ~4 observation steps (entering + merging)
  - 1-bit carry (depth 1, e.g. 1→2): ~8 steps (entering + merge + carry + merge)
  - 2-bit carry (depth 2, e.g. 3→4): ~12 steps
  - 3-bit carry (depth 3, e.g. 7→8): ~14-16 steps (full cascade)

We map this to: each bit flip takes 2 steps (matching CARRY_STEPS_PER_PHASE),
plus 2 steps for the initial "entering" phase where no bits have changed yet.
So carry depth d → 2 + 2*d steps of transition, with the remaining steps
showing the stable new count.

For 7→8 (carry depth 3, 4 bits flip):
  steps_per_count = 20
  transition_steps = 2 + 2*4 = 10 steps
  stable_old = 20 - 10 = 10 steps of [1,1,1,0]
  then: 2 steps [1,1,1,0] (entering, no change yet)
        2 steps [0,1,1,0] (bit 0 flips)
        2 steps [0,0,1,0] (bit 1 flips)
        2 steps [0,0,0,0] (bit 2 flips)
        2 steps [0,0,0,1] (bit 3 flips → count 8)
  next count starts: 20 steps of [0,0,0,1]

For 0→1 (carry depth 0, 1 bit flips):
  transition_steps = 2 + 2*1 = 4 steps
  stable_old = 20 - 4 = 16 steps of [0,0,0,0]
  then: 2 steps [0,0,0,0] (entering)
        2 steps [1,0,0,0] (bit 0 flips → count 1)
  next count starts: 20 steps of [1,0,0,0]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BITS = 4
VOCAB_SIZE = 2
MAX_COUNT = 2 ** NUM_BITS  # 16

DEFAULT_STEPS_PER_COUNT = 20
DEFAULT_N_CYCLES = 1

# Carry timing: matches physical specialist CARRY_STEPS_PER_PHASE = 2
STEPS_PER_PHASE = 2  # steps per bit flip
ENTERING_STEPS = 2   # initial phase before first flip


def int_to_bits(n, num_bits=NUM_BITS):
    """Convert integer to LSB-first bit array."""
    return np.array([(n >> i) & 1 for i in range(num_bits)], dtype=np.int64)


def carry_depth(n):
    """Trailing 1-bits in n = carries when incrementing n."""
    if n == 0:
        return 0
    depth = 0
    while n & 1:
        depth += 1
        n >>= 1
    return depth


def bits_that_flip(old_count, new_count):
    """Return list of bit indices that change, ordered LSB first."""
    old_bits = int_to_bits(old_count)
    new_bits = int_to_bits(new_count)
    return [i for i in range(NUM_BITS) if old_bits[i] != new_bits[i]]


def compute_transition_schedule(old_count, new_count, steps_per_count):
    """Compute the timestep-by-timestep token observation during a transition.

    Returns list of (step_offset, bits) pairs for the full count window,
    where step_offset is 0..steps_per_count-1.

    The schedule fills the window as:
      [0, stable_end): stable old count
      [stable_end, stable_end + ENTERING_STEPS): entering phase (still old bits)
      [entering_end, ...): one bit flips every STEPS_PER_PHASE steps, LSB first
      [..., steps_per_count): stable new count (if any steps remain)
    """
    flipping = bits_that_flip(old_count, new_count)
    n_flips = len(flipping)

    # Total transition steps: entering + flips
    transition_steps = ENTERING_STEPS + n_flips * STEPS_PER_PHASE
    # Clamp to available steps
    transition_steps = min(transition_steps, steps_per_count)

    stable_old_steps = steps_per_count - transition_steps

    schedule = []
    current_bits = int_to_bits(old_count).copy()

    # Phase 1: Stable old count
    for s in range(stable_old_steps):
        schedule.append(current_bits.copy())

    # Phase 2: Entering (still old bits)
    entering_end = min(stable_old_steps + ENTERING_STEPS, steps_per_count)
    for s in range(stable_old_steps, entering_end):
        schedule.append(current_bits.copy())

    # Phase 3: Sequential bit flips
    flip_idx = 0
    step = entering_end
    while flip_idx < n_flips and step < steps_per_count:
        # Flip this bit
        bit = flipping[flip_idx]
        current_bits[bit] = 1 - current_bits[bit]
        # Hold for STEPS_PER_PHASE steps
        for _ in range(STEPS_PER_PHASE):
            if step >= steps_per_count:
                break
            schedule.append(current_bits.copy())
            step += 1
        flip_idx += 1

    # Phase 4: Remaining steps at new count (if any)
    new_bits = int_to_bits(new_count)
    while len(schedule) < steps_per_count:
        schedule.append(new_bits.copy())

    assert len(schedule) == steps_per_count
    return schedule


class RichSymbolicBinaryEnv:
    """Rich symbolic 4-bit counter with carry cascade dynamics.

    Variant B: bit flips are sequenced over timesteps during transitions.
    """

    def __init__(self, steps_per_count=DEFAULT_STEPS_PER_COUNT,
                 n_cycles=DEFAULT_N_CYCLES, seed=0):
        self.steps_per_count = steps_per_count
        self.n_cycles = n_cycles
        self.max_steps = steps_per_count * MAX_COUNT * n_cycles
        self._rng = np.random.RandomState(seed)

        # Precompute all transition schedules for the full episode
        self._precompute_schedules()

        self._step = 0
        self._count = 0
        self._done = True

    def _precompute_schedules(self):
        """Build the full observation sequence for one cycle."""
        self._cycle_obs = []
        for c in range(MAX_COUNT):
            next_c = (c + 1) % MAX_COUNT
            schedule = compute_transition_schedule(c, next_c, self.steps_per_count)
            self._cycle_obs.extend(schedule)
        # Full episode = n_cycles repetitions
        self._episode_obs = self._cycle_obs * self.n_cycles

    @property
    def observation_space(self):
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
        tokens = self._episode_obs[self._step]
        return {
            "tokens": tokens.copy(),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_last,
        }

    @property
    def metadata(self):
        """Non-observation metadata for evaluation."""
        # Decimal count is the count at the START of this count window
        count_window = self._step // self.steps_per_count
        decimal_count = count_window % MAX_COUNT
        # Position within current count window
        window_pos = self._step % self.steps_per_count
        # Are we in the carry cascade portion?
        next_count = (decimal_count + 1) % MAX_COUNT
        flipping = bits_that_flip(decimal_count, next_count)
        transition_steps = ENTERING_STEPS + len(flipping) * STEPS_PER_PHASE
        stable_steps = self.steps_per_count - transition_steps
        in_cascade = window_pos >= stable_steps

        return {
            "decimal_count": decimal_count,
            "bits": self._episode_obs[self._step].tolist(),
            "carry_depth": carry_depth(decimal_count),
            "step": self._step,
            "in_cascade": in_cascade,
            "window_pos": window_pos,
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
        done = self._step >= self.max_steps
        self._done = done

        # Clamp step for final observation access
        if self._step >= len(self._episode_obs):
            self._step = len(self._episode_obs) - 1

        obs = self._get_obs(is_last=done)
        info = self.metadata
        return obs, 0.0, done, info

    def close(self):
        pass


# ---------------------------------------------------------------------------
# DreamerV3-compatible wrapper
# ---------------------------------------------------------------------------

class RichSymbolicBinaryWorld:
    """DreamerV3-compatible wrapper."""

    def __init__(self, task="rich_symbolic_binary", seed=0, **kwargs):
        import os
        steps_per_count = int(os.environ.get("SYMBOLIC_STEPS_PER_COUNT",
                                              DEFAULT_STEPS_PER_COUNT))
        n_cycles = int(os.environ.get("SYMBOLIC_N_CYCLES", DEFAULT_N_CYCLES))
        self._env = RichSymbolicBinaryEnv(
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
    print("RichSymbolicBinaryEnv — Self-test")
    print("=" * 60)

    env = RichSymbolicBinaryEnv(steps_per_count=20, n_cycles=1)
    obs = env.reset()

    # Print transitions with carry detail
    print(f"\nFull episode ({env.max_steps} steps):")
    print(f"{'step':>4}  {'tokens':>12}  {'count':>5}  {'cascade':>7}  {'wpos':>4}")
    print("-" * 45)

    prev_tokens = None
    interesting_steps = set()

    # Collect all steps
    steps_data = []
    steps_data.append((0, obs["tokens"].tolist(), env.metadata))

    done = False
    while not done:
        obs, _, done, info = env.step(0)
        steps_data.append((env._step, obs["tokens"].tolist(), info))

    # Print only steps where tokens change, plus context
    prev_tok = None
    for step, tokens, meta in steps_data:
        tok_str = str(tokens)
        changed = prev_tok is not None and tokens != prev_tok
        if changed or step < 3 or step >= len(steps_data) - 2 or step % 20 < 2:
            marker = " ←" if changed else ""
            print(f"{step:4d}  {tok_str:>12}  {meta['decimal_count']:5d}  "
                  f"{'YES' if meta.get('in_cascade') else 'no':>7}  "
                  f"{meta.get('window_pos', 0):4d}{marker}")
        prev_tok = tokens

    # Verify specific transitions
    print("\n--- Transition verification ---")

    # Check 7→8 (the deep cascade)
    # Count 7 window starts at step 7*20=140, ends at 160
    # Bits should flip sequentially
    window_start = 7 * 20
    print(f"\nCount 7→8 (carry depth 3, 4 bits flip):")
    for i in range(20):
        s = window_start + i
        tokens = steps_data[s][1]
        print(f"  step {s}: {tokens}")

    # Check 0→1 (simple flip)
    print(f"\nCount 0→1 (carry depth 0, 1 bit flip):")
    for i in range(20):
        s = i
        tokens = steps_data[s][1]
        print(f"  step {s}: {tokens}")

    # Verify episode length matches Variant A
    # steps_data includes reset obs (step 0) + max_steps step() calls
    assert len(steps_data) == 20 * 16 + 1, f"Expected {20*16+1} entries, got {len(steps_data)}"

    # Verify final state wraps to 0
    final_tokens = steps_data[-1][1]
    assert final_tokens == [0, 0, 0, 0], f"Expected [0,0,0,0] at end, got {final_tokens}"

    print(f"\nTotal steps: {len(steps_data) - 1} env steps (matches Variant A: {20 * 16})")
    print("All verifications passed!")
    print("=" * 60)
