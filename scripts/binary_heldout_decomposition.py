"""
Held-out transition test for binary successor decomposition.

Tests whether the compositional bit-flip decomposition generalizes to
transitions not seen during probe training, or is a tautological
artifact of probe construction.
"""

import json
import numpy as np
from sklearn.linear_model import Ridge
from pathlib import Path
from itertools import combinations

BATTERY_PATH = Path(__file__).parent.parent / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
OUTPUT_JSON = Path(__file__).parent.parent / 'results' / 'binary_heldout_decomposition.json'
OUTPUT_MD = Path(__file__).parent.parent / 'results' / 'binary_heldout_decomposition.md'

TRANSITIONS = [{'from': i, 'to': i+1} for i in range(15)]

CARRY_DEPTHS = {}
for t in TRANSITIONS:
    n = t['from']
    xor = n ^ (n + 1)
    depth = bin(xor).count('1') - 1
    CARRY_DEPTHS[n] = depth


def bits_for_count(n):
    return [(n >> b) & 1 for b in range(4)]


def get_bit_changes(n_from, n_to):
    changes = {}
    for b in range(4):
        bf = (n_from >> b) & 1
        bt = (n_to >> b) & 1
        if bf != bt:
            changes[b] = 1 if bt > bf else -1
    return changes


def train_probes(h_t, bits, mask=None):
    if mask is not None:
        h_t = h_t[mask]
        bits = bits[mask]
    probe_weights = []
    probe_accuracies = []
    for b in range(4):
        probe = Ridge(alpha=1.0)
        probe.fit(h_t, bits[:, b])
        w = probe.coef_ / np.linalg.norm(probe.coef_)
        probe_weights.append(w)
        acc = (np.round(probe.predict(h_t)).astype(int) == bits[:, b]).mean()
        probe_accuracies.append(float(acc))
    return probe_weights, probe_accuracies


def compute_centroids(h_t, counts):
    centroids = {}
    for c in sorted(set(counts)):
        centroids[c] = h_t[counts == c].mean(axis=0)
    return centroids


def evaluate_decomposition(step_vectors, transitions, probe_weights):
    results = []
    total_correct = 0
    total_changes = 0
    all_crosstalks = []

    for i, t in enumerate(transitions):
        sv = step_vectors[i]
        projections = [float(np.dot(sv, w)) for w in probe_weights]
        changes = get_bit_changes(t['from'], t['to'])
        depth = CARRY_DEPTHS[t['from']]

        correct = 0
        n_changes = 0
        crosstalk_vals = []

        for b in range(4):
            if b in changes:
                n_changes += 1
                expected_sign = changes[b]
                actual_sign = 1 if projections[b] > 0 else -1
                if expected_sign == actual_sign:
                    correct += 1
            else:
                crosstalk_vals.append(abs(projections[b]))

        flipping_magnitudes = [abs(projections[b]) for b in changes]
        mean_flip_mag = np.mean(flipping_magnitudes) if flipping_magnitudes else 1.0
        crosstalk = np.mean(crosstalk_vals) / mean_flip_mag if crosstalk_vals and mean_flip_mag > 0 else 0.0

        reconstructed = sum(p * w for p, w in zip(projections, probe_weights))
        sv_norm = np.linalg.norm(sv)
        recon_cos = float(np.dot(sv, reconstructed) / (sv_norm * np.linalg.norm(reconstructed) + 1e-12)) if sv_norm > 0 else 0.0

        total_correct += correct
        total_changes += n_changes
        all_crosstalks.append(crosstalk)

        results.append({
            'transition': f'{t["from"]}->{t["to"]}',
            'carry_depth': depth,
            'sign_agreement': correct / n_changes if n_changes > 0 else 1.0,
            'crosstalk': float(crosstalk),
            'reconstruction_cosine': recon_cos,
            'projections': projections
        })

    agg = {
        'aggregate_sign_agreement': total_correct / total_changes if total_changes > 0 else 0.0,
        'aggregate_crosstalk_mean': float(np.mean(all_crosstalks)),
        'aggregate_reconstruction_cosine': float(np.mean([r['reconstruction_cosine'] for r in results]))
    }

    return results, agg


def run_scheme_a(h_t, counts, bits):
    """Stratified holdout: one transition per carry depth."""
    heldout = [
        {'from': 2, 'to': 3},    # depth 0
        {'from': 5, 'to': 6},    # depth 1
        {'from': 11, 'to': 12},  # depth 2
        {'from': 7, 'to': 8},    # depth 3
    ]
    heldout_set = {(t['from'], t['to']) for t in heldout}
    train_transitions = [t for t in TRANSITIONS if (t['from'], t['to']) not in heldout_set]

    train_counts_set = set()
    for t in train_transitions:
        train_counts_set.add(t['from'])
        train_counts_set.add(t['to'])

    train_mask = np.isin(counts, list(train_counts_set))

    probe_weights_heldout, probe_accs_heldout = train_probes(h_t, bits, train_mask)
    probe_weights_full, probe_accs_full = train_probes(h_t, bits)

    centroids = compute_centroids(h_t, counts)

    heldout_steps = [centroids[t['to']] - centroids[t['from']] for t in heldout]
    heldout_results, heldout_agg = evaluate_decomposition(heldout_steps, heldout, probe_weights_heldout)

    train_steps = [centroids[t['to']] - centroids[t['from']] for t in train_transitions]
    nonheldout_results, nonheldout_agg = evaluate_decomposition(train_steps, train_transitions, probe_weights_full)

    full_steps = [centroids[t['to']] - centroids[t['from']] for t in TRANSITIONS]
    full_results, full_agg = evaluate_decomposition(full_steps, TRANSITIONS, probe_weights_full)

    return {
        'heldout_transitions': [f'{t["from"]}->{t["to"]}' for t in heldout],
        'training_transitions': [f'{t["from"]}->{t["to"]}' for t in train_transitions],
        'probe_accuracies_heldout_trained': probe_accs_heldout,
        'probe_accuracies_full_trained': probe_accs_full,
        'nonheldout_baseline': nonheldout_agg,
        'full_baseline': full_agg,
        'heldout_results': {
            'per_transition': [{k: v for k, v in r.items() if k != 'projections'} for r in heldout_results],
            **heldout_agg
        },
        'full_results_per_transition': [{k: v for k, v in r.items() if k != 'projections'} for r in full_results]
    }


def run_scheme_b(h_t, counts, bits):
    """Leave-one-carry-depth-out."""
    depth_groups = {0: [], 1: [], 2: [], 3: []}
    for t in TRANSITIONS:
        d = CARRY_DEPTHS[t['from']]
        depth_groups[d].append(t)

    results_per_depth = []
    for heldout_depth in range(4):
        heldout = depth_groups[heldout_depth]
        train_transitions = [t for t in TRANSITIONS if CARRY_DEPTHS[t['from']] != heldout_depth]

        train_counts_set = set()
        for t in train_transitions:
            train_counts_set.add(t['from'])
            train_counts_set.add(t['to'])
        train_mask = np.isin(counts, list(train_counts_set))

        probe_weights, _ = train_probes(h_t, bits, train_mask)
        centroids = compute_centroids(h_t, counts)

        heldout_steps = [centroids[t['to']] - centroids[t['from']] for t in heldout]
        heldout_results, heldout_agg = evaluate_decomposition(heldout_steps, heldout, probe_weights)

        results_per_depth.append({
            'heldout_depth': heldout_depth,
            'n_heldout_transitions': len(heldout),
            'heldout_transitions': [f'{t["from"]}->{t["to"]}' for t in heldout],
            **heldout_agg,
            'per_transition': [{k: v for k, v in r.items() if k != 'projections'} for r in heldout_results]
        })

    return {'per_depth': results_per_depth}


def compute_orthogonality(h_t, counts, bits):
    """Probe weight orthogonality under full vs held-out training."""
    probe_weights_full, _ = train_probes(h_t, bits)

    train_counts_set = set()
    for t in TRANSITIONS:
        if CARRY_DEPTHS[t['from']] != 3:
            train_counts_set.add(t['from'])
            train_counts_set.add(t['to'])
    mask = np.isin(counts, list(train_counts_set))
    probe_weights_heldout, _ = train_probes(h_t, bits, mask)

    def pairwise_cosines(weights):
        pairs = {}
        for i, j in combinations(range(4), 2):
            cos = float(np.dot(weights[i], weights[j]))
            pairs[f'w{i}_w{j}'] = cos
        return pairs

    full_cos = pairwise_cosines(probe_weights_full)
    heldout_cos = pairwise_cosines(probe_weights_heldout)

    return {
        'full_training': {
            'pairwise_cosines': full_cos,
            'max_absolute_cosine': max(abs(v) for v in full_cos.values())
        },
        'heldout_training_no_depth3': {
            'pairwise_cosines': heldout_cos,
            'max_absolute_cosine': max(abs(v) for v in heldout_cos.values())
        }
    }


def classify_outcome(heldout_agg):
    sa = heldout_agg['aggregate_sign_agreement']
    rc = heldout_agg['aggregate_reconstruction_cosine']
    ct = heldout_agg['aggregate_crosstalk_mean']

    if sa >= 0.9 and rc >= 0.85:
        return 'G1'
    elif sa >= 0.6 or rc >= 0.6:
        return 'G2'
    else:
        return 'G3'


def generate_markdown(results):
    sa = results['scheme_A_stratified_holdout']
    sb = results['scheme_B_leave_depth_out']
    outcome = results['outcome_classification']

    md = []
    md.append('# Held-Out Transition Test: Binary Successor Decomposition\n')
    md.append(f'**Outcome: {outcome}**\n')
    md.append(results.get('notes', ''))
    md.append('')

    md.append('## Scheme A: Stratified Holdout\n')
    md.append(f'Held out: {", ".join(sa["heldout_transitions"])}')
    md.append(f'Trained on: {", ".join(sa["training_transitions"])}\n')

    md.append('### Full-data baseline (all 14 transitions, probes trained on all data)\n')
    fb = sa['full_baseline']
    md.append(f'| Metric | Value |')
    md.append(f'|--------|:-----:|')
    md.append(f'| Sign agreement | {fb["aggregate_sign_agreement"]:.1%} |')
    md.append(f'| Mean cross-talk | {fb["aggregate_crosstalk_mean"]:.4f} |')
    md.append(f'| Reconstruction cosine | {fb["aggregate_reconstruction_cosine"]:.4f} |')
    md.append('')

    md.append('### Held-out results (probes trained WITHOUT held-out transitions)\n')
    md.append('| Transition | Carry depth | Sign agreement | Cross-talk | Recon cosine |')
    md.append('|:---:|:---:|:---:|:---:|:---:|')
    for r in sa['heldout_results']['per_transition']:
        md.append(f'| {r["transition"]} | {r["carry_depth"]} | {r["sign_agreement"]:.0%} | {r["crosstalk"]:.4f} | {r["reconstruction_cosine"]:.4f} |')
    hr = sa['heldout_results']
    md.append(f'| **Aggregate** | | **{hr["aggregate_sign_agreement"]:.1%}** | **{hr["aggregate_crosstalk_mean"]:.4f}** | **{hr["aggregate_reconstruction_cosine"]:.4f}** |')
    md.append('')

    md.append('### Comparison\n')
    md.append('| Metric | Full-data baseline | Held-out |')
    md.append('|--------|:---:|:---:|')
    md.append(f'| Sign agreement | {fb["aggregate_sign_agreement"]:.1%} | {hr["aggregate_sign_agreement"]:.1%} |')
    md.append(f'| Mean cross-talk | {fb["aggregate_crosstalk_mean"]:.4f} | {hr["aggregate_crosstalk_mean"]:.4f} |')
    md.append(f'| Recon cosine | {fb["aggregate_reconstruction_cosine"]:.4f} | {hr["aggregate_reconstruction_cosine"]:.4f} |')
    md.append('')

    md.append('## Scheme B: Leave-One-Carry-Depth-Out\n')
    md.append('| Held-out depth | N transitions | Sign agreement | Mean cross-talk | Recon cosine |')
    md.append('|:---:|:---:|:---:|:---:|:---:|')
    for d in sb['per_depth']:
        md.append(f'| {d["heldout_depth"]} | {d["n_heldout_transitions"]} | {d["aggregate_sign_agreement"]:.1%} | {d["aggregate_crosstalk_mean"]:.4f} | {d["aggregate_reconstruction_cosine"]:.4f} |')
    md.append('')

    orth = results['orthogonality_heldout']
    md.append('## Orthogonality Diagnostic\n')
    md.append('| Pair | Full training | Held-out (no depth 3) |')
    md.append('|------|:---:|:---:|')
    for pair in orth['full_training']['pairwise_cosines']:
        fc = orth['full_training']['pairwise_cosines'][pair]
        hc = orth['heldout_training_no_depth3']['pairwise_cosines'][pair]
        md.append(f'| {pair} | {fc:.4f} | {hc:.4f} |')
    md.append(f'| **Max |cos|** | **{orth["full_training"]["max_absolute_cosine"]:.4f}** | **{orth["heldout_training_no_depth3"]["max_absolute_cosine"]:.4f}** |')
    md.append('')

    return '\n'.join(md)


def main():
    print('Loading battery data...')
    data = np.load(str(BATTERY_PATH), allow_pickle=True)
    h_t = data['h_t']
    counts = data['counts']
    bits = data['bits']

    print(f'h_t: {h_t.shape}, counts: {counts.shape} (range {counts.min()}-{counts.max()}), bits: {bits.shape}')

    samples_per_count = {int(c): int((counts == c).sum()) for c in sorted(set(counts))}
    print(f'Samples per count: min={min(samples_per_count.values())}, max={max(samples_per_count.values())}')
    if min(samples_per_count.values()) < 30:
        print('WARNING: Some counts have fewer than 30 samples!')

    print('\n=== Scheme A: Stratified Holdout ===')
    scheme_a = run_scheme_a(h_t, counts, bits)
    print(f'Full baseline sign agreement: {scheme_a["full_baseline"]["aggregate_sign_agreement"]:.1%}')
    print(f'Held-out sign agreement: {scheme_a["heldout_results"]["aggregate_sign_agreement"]:.1%}')
    print(f'Held-out recon cosine: {scheme_a["heldout_results"]["aggregate_reconstruction_cosine"]:.4f}')

    if scheme_a['full_baseline']['aggregate_sign_agreement'] < 0.98:
        print(f'KILL: Full baseline sign agreement {scheme_a["full_baseline"]["aggregate_sign_agreement"]:.1%} < 98%. Environmental drift.')
        return

    print('\n=== Scheme B: Leave-One-Carry-Depth-Out ===')
    scheme_b = run_scheme_b(h_t, counts, bits)
    for d in scheme_b['per_depth']:
        print(f'  Depth {d["heldout_depth"]}: sign={d["aggregate_sign_agreement"]:.1%}, recon_cos={d["aggregate_reconstruction_cosine"]:.4f}')

    print('\n=== Orthogonality Diagnostic ===')
    orth = compute_orthogonality(h_t, counts, bits)
    print(f'Full training max |cos|: {orth["full_training"]["max_absolute_cosine"]:.4f}')
    print(f'Held-out max |cos|: {orth["heldout_training_no_depth3"]["max_absolute_cosine"]:.4f}')

    outcome = classify_outcome(scheme_a['heldout_results'])

    if outcome == 'G1':
        notes = ('The compositional decomposition generalizes cleanly to held-out transitions. '
                 'Sign agreement, cross-talk, and reconstruction fidelity on transitions never seen '
                 'during probe training are comparable to the full-data baseline. The "independently '
                 'invented coordinate system" framing is supported — the four bit-flip directions are '
                 'a stable property of the representation, not an artifact of probe fitting.')
    elif outcome == 'G2':
        notes = ('The decomposition partially generalizes. The four bit-flip directions capture most '
                 'of the step vector structure on held-out transitions, but with reduced fidelity compared '
                 'to the full-data baseline. The writeup should soften from "independently invented a coordinate '
                 'system" to "approximately decomposes into four primary bit-flip directions, with residual '
                 'transition-specific structure."')
    else:
        notes = ('The decomposition does not generalize to held-out transitions. The compositional claim '
                 'is likely an artifact of probe construction. The writeup should be substantially reframed: '
                 'bit states are linearly decodable, but the "coordinate system" framing is not supported by '
                 'held-out evidence.')

    results = {
        'scheme_A_stratified_holdout': scheme_a,
        'scheme_B_leave_depth_out': scheme_b,
        'orthogonality_heldout': orth,
        'outcome_classification': outcome,
        'notes': notes
    }

    with open(str(OUTPUT_JSON), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_JSON}')

    md = generate_markdown(results)
    with open(str(OUTPUT_MD), 'w') as f:
        f.write(md)
    print(f'Writeup saved to {OUTPUT_MD}')

    print(f'\n=== OUTCOME: {outcome} ===')
    print(notes)


if __name__ == '__main__':
    main()
