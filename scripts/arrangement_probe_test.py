#!/usr/bin/env python3
"""Quick per-arrangement probe accuracy — ultra-fast version.
Uses only 500 RSSM steps per arrangement (no full episode needed).
"""
import numpy as np
import json, sys, time
from pathlib import Path

MODELS_DIR = Path('/workspace/bridge/models/robust_varied')

import export_deter_centroids as edc
edc.BIN_PATH = MODELS_DIR / 'dreamer_weights.bin'
edc.MANIFEST_PATH = MODELS_DIR / 'dreamer_manifest.json'
weights = edc.load_weights()

with open(MODELS_DIR / 'embed_probe.json') as f:
    pd = json.load(f)
probe_w = np.array(pd['weights'], dtype=np.float32)
probe_b = float(pd['bias'])

from scipy.stats import ortho_group
proj = ortho_group.rvs(82, random_state=np.random.RandomState(42_000)).astype(np.float32)

from counting_env_pure import CountingWorldEnv, _ALL_TYPES
import counting_env_pure as cep

MAX_STEPS = 800  # ~800 steps captures counting phase for 10 blobs

results = {}
t0 = time.time()
for arr_type in _ALL_TYPES:
    rssm = edc.FastRSSM(weights)
    exact = within1 = total = 0
    orig = cep._random_arrangement
    cep._random_arrangement = lambda varied=False, t=arr_type: t
    env = CountingWorldEnv(blob_count_min=10, blob_count_max=10, conservation=True)
    obs = env.reset()
    rssm.reset()
    cep._random_arrangement = orig
    for step in range(MAX_STEPS):
        obs_p = (proj @ obs.astype(np.float32))[:82]
        deter = rssm.step(obs_p, 0.0)
        raw = float(deter @ probe_w + probe_b)
        pred = int(np.clip(np.round(raw), 0, 25))
        gt = int(obs[81])
        if env._state.phase == 'counting':
            total += 1
            if pred == gt: exact += 1
            if abs(pred - gt) <= 1: within1 += 1
        obs, _, done, _ = env.step(0)
        if done:
            break
    env.close()
    e_pct = 100 * exact / max(total, 1)
    w_pct = 100 * within1 / max(total, 1)
    results[arr_type] = (e_pct, w_pct, total)
    elapsed = time.time() - t0
    print(f'{arr_type:20s}  exact={e_pct:5.1f}%  within1={w_pct:5.1f}%  n={total:4d}  [{elapsed:.0f}s]', flush=True)

vals = [r[0] for r in results.values()]
print(f'\n--- Summary ---')
print(f'Mean exact:  {np.mean(vals):.1f}%   Std: {np.std(vals):.1f}%')
print(f'Min: {min(vals):.1f}% ({min(results, key=lambda k: results[k][0])})')
print(f'Max: {max(vals):.1f}% ({max(results, key=lambda k: results[k][0])})')

base4 = ['scattered', 'clustered', 'grid-like', 'mixed']
new = [t for t in _ALL_TYPES if t not in base4]
bm = np.mean([results[t][0] for t in base4])
nm = np.mean([results[t][0] for t in new])
print(f'\nBase 4 mean: {bm:.1f}%')
print(f'New 16 mean: {nm:.1f}%')
print(f'Delta:       {nm - bm:+.1f}%')
print(f'\nTotal time: {time.time() - t0:.0f}s')
