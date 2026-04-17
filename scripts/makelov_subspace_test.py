"""Makelov subspace-illusion test for binary successor bit-flip directions.

Tests whether the four bit-flip probe directions w_i are causally faithful or
partially illusory per Makelov, Lange & Nanda 2023. Decomposes each w_i into
w_i^null (projected onto the approximate nullspace of the decoder's h_t
projection) and w_i^row (the complement). Compares M1 (decoder-flip rate) for
intervention along the full direction vs the rowspace-only component at
α = α_nat.

α_nat(c, i) = |(μ_{c+1} − μ_c) · ŵ_i| for source counts c where bit i flips.
Per-bit α for Step 3: mean across flipping transitions.

Count=15 excluded from all source-count cells (terminal-state sampling
asymmetry, per the split-convention).
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

BRIDGE_SCRIPTS = Path("/Users/petermurphy/cc-sandbox/projects/jamstack-v1/bridge/scripts")
CHECKPOINT_DIR = Path("/Users/petermurphy/cc-sandbox/projects/jamstack-v1/bridge/artifacts/checkpoints/binary_baseline_s0/exported")
BATTERY_PATH = Path("/Users/petermurphy/anima-bridge/artifacts/battery/binary_baseline_s0/battery.npz")
OUT_JSON = Path("/Users/petermurphy/anima-bridge/results/makelov_subspace_test.json")
OUT_MD = Path("/Users/petermurphy/anima-bridge/results/makelov_subspace_test.md")

sys.path.insert(0, str(BRIDGE_SCRIPTS))
from quick_ghe_binary import load_exported_weights, STOCH_FLAT, STOCH_DIM, STOCH_CLASSES, DETER_DIM

LN_EPS = 1e-3
N_SAMPLES_PER_CELL = 100
BIT_OBS_INDICES = [49, 53, 57, 61]  # obs[base + 0] = col.occupied for the 4 bit columns
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)


def _ln(x, w, b, eps=LN_EPS):
    mu = x.mean()
    var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _argmax_one_hot(logits):
    out = np.zeros(STOCH_FLAT, dtype=np.float32)
    idx = logits.argmax(axis=-1)
    for i, j in enumerate(idx):
        out[i * STOCH_CLASSES + j] = 1.0
    return out


def prior_stoch(deter, w):
    """Compute prior stoch deterministically from deter (=h_t)."""
    h = _silu(_ln(w["img_out_w"] @ deter, w["img_out_norm_w"], w["img_out_norm_b"]))
    logits = (w["imgs_stat_w"] @ h + w["imgs_stat_b"]).reshape(STOCH_DIM, STOCH_CLASSES)
    return _argmax_one_hot(logits)


def decode(stoch, deter, w):
    """Forward pass through the decoder. Returns predicted obs (symlog'd)."""
    x = np.concatenate([stoch, deter])
    h = _silu(_ln(w["dec_linear0_w"] @ x, w["dec_norm0_w"], w["dec_norm0_b"]))
    h = _silu(_ln(w["dec_linear1_w"] @ h, w["dec_norm1_w"], w["dec_norm1_b"]))
    h = _silu(_ln(w["dec_linear2_w"] @ h, w["dec_norm2_w"], w["dec_norm2_b"]))
    obs_pred_symlog = w["dec_out_w"] @ h + w["dec_out_b"]
    return obs_pred_symlog


def symexp(y):
    return np.sign(y) * (np.exp(np.abs(y)) - 1.0)


def decode_bits(stoch, deter, w):
    """Decode obs, return 4-bit prediction (thresholded at 0.5 in raw space)."""
    obs_symlog = decode(stoch, deter, w)
    obs = symexp(obs_symlog)
    return np.array([1 if obs[i] > 0.5 else 0 for i in BIT_OBS_INDICES], dtype=np.int32)


def bit_flips_on_transition(c):
    """Which bits flip going c → c+1."""
    xor = c ^ (c + 1)
    return [b for b in range(4) if (xor >> b) & 1]


def bca_ci(samples, statistic_func, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """BCa bootstrap confidence interval."""
    samples = np.asarray(samples)
    n = len(samples)
    if n == 0:
        return (float("nan"), float("nan"))
    theta_hat = statistic_func(samples)
    boot_thetas = np.array([
        statistic_func(samples[RNG.integers(0, n, size=n)])
        for _ in range(n_bootstrap)
    ])
    # Bias correction
    frac = np.mean(boot_thetas < theta_hat)
    if frac == 0.0 or frac == 1.0:
        return (float(np.percentile(boot_thetas, 2.5)), float(np.percentile(boot_thetas, 97.5)))
    from scipy.stats import norm
    z0 = norm.ppf(frac)
    # Acceleration via jackknife
    jack = np.array([statistic_func(np.delete(samples, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5
    a = num / denom if denom > 0 else 0.0
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)
    alpha_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    return (float(np.percentile(boot_thetas, 100 * alpha_lo)),
            float(np.percentile(boot_thetas, 100 * alpha_hi)))


def main():
    print("Loading battery + weights...")
    d = np.load(BATTERY_PATH)
    h_t, counts, bits_actual, carry_active = d["h_t"], d["counts"], d["bits"], d["carry_active"]
    stable = carry_active == 0
    w = load_exported_weights(CHECKPOINT_DIR)

    # Refit probes on stable samples
    print("Refitting probes...")
    probe_weights_raw = []
    for b in range(4):
        pr = Ridge(alpha=1.0).fit(h_t[stable], bits_actual[stable][:, b])
        probe_weights_raw.append(pr.coef_.astype(np.float64))
    probe_weights = [w_ / np.linalg.norm(w_) for w_ in probe_weights_raw]  # unit-normalized

    # ─── Step 1: decoder projection SVD ───────────────────────────────────
    W_dec_full = w["dec_linear0_w"].astype(np.float64)  # (512, 1536)
    W_h = W_dec_full[:, STOCH_FLAT:]  # (512, 512) — multiplies h_t
    U, S, Vt = np.linalg.svd(W_h, full_matrices=False)
    sv_spectrum = S.tolist()

    thresholds = {
        "strict_1e-4_rel": 1e-4 * S.max(),
        "relaxed_1e-3_rel": 1e-3 * S.max(),
        "relaxed_1e-2_rel": 1e-2 * S.max(),
    }
    rank_by_thr = {}
    null_basis_by_thr = {}
    for name, thr in thresholds.items():
        mask = S <= thr
        null_basis = Vt[mask]  # rows of Vt = right singular vectors → h_t-space directions
        rank_by_thr[name] = int((~mask).sum())
        null_basis_by_thr[name] = null_basis  # (null_dim, 512)
        print(f"  threshold {name} (SV ≤ {thr:.6f}): rank={rank_by_thr[name]}, "
              f"null_dim={null_basis.shape[0]}")

    # ─── α_nat: per-bit, per-flipping-transition ──────────────────────────
    print("\nComputing α_nat per bit per flipping transition...")
    unique_counts = sorted(set(counts[stable].tolist()))
    centroids = {c: h_t[stable][counts[stable] == c].mean(0).astype(np.float64) for c in unique_counts}
    step_vectors = {c: centroids[c + 1] - centroids[c] for c in range(15) if c + 1 in centroids}

    alpha_nat_per_trans = {b: {} for b in range(4)}  # bit → {source_count: α}
    for c, sv in step_vectors.items():
        if c == 14:
            continue  # exclude terminal 14→15 per split-convention
        flipped = bit_flips_on_transition(c)
        for b in flipped:
            alpha_nat_per_trans[b][c] = float(abs(np.dot(sv, probe_weights[b])))

    alpha_nat_mean = {b: float(np.mean(list(alpha_nat_per_trans[b].values())))
                      for b in range(4) if alpha_nat_per_trans[b]}
    alpha_nat_spread = {b: {"min": float(min(alpha_nat_per_trans[b].values())),
                            "max": float(max(alpha_nat_per_trans[b].values())),
                            "std": float(np.std(list(alpha_nat_per_trans[b].values())))}
                        for b in range(4) if alpha_nat_per_trans[b]}

    for b in range(4):
        trans_str = ", ".join(f"{c}→{c+1}: {v:.3f}" for c, v in alpha_nat_per_trans[b].items())
        print(f"  bit {b}: mean α_nat = {alpha_nat_mean[b]:.3f}, per-transition: {trans_str}")

    # ─── Step 2: decomposition (using primary relaxed_1e-2_rel threshold) ──
    primary_thr = "relaxed_1e-2_rel"
    null_basis = null_basis_by_thr[primary_thr]  # (null_dim, 512)
    # Projector onto nullspace: P_null = N^T N where N rows are orthonormal null directions
    P_null = null_basis.T @ null_basis  # (512, 512)

    decomp = {}
    for b in range(4):
        w_raw = probe_weights_raw[b]
        w_unit = probe_weights[b]
        w_null = P_null @ w_unit
        w_row = w_unit - w_null
        null_norm = float(np.linalg.norm(w_null))
        row_norm = float(np.linalg.norm(w_row))
        decomp[b] = {
            "w_null": w_null,
            "w_row": w_row,
            "w_null_hat": w_null / null_norm if null_norm > 0 else np.zeros_like(w_null),
            "w_row_hat": w_row / row_norm if row_norm > 0 else np.zeros_like(w_row),
            "frac_null": null_norm,  # since w_unit is unit-norm, ratio is the norm itself
            "frac_row": row_norm,
            "pythag_check": null_norm**2 + row_norm**2,  # should be ~1
        }

    print(f"\n[Step 2] Decomposition at primary threshold ({primary_thr}, null_dim={null_basis.shape[0]}):")
    for b in range(4):
        print(f"  bit {b}: ‖w_null‖/‖w‖ = {decomp[b]['frac_null']:.4f}, "
              f"‖w_row‖/‖w‖ = {decomp[b]['frac_row']:.4f}, "
              f"Pythag sum = {decomp[b]['pythag_check']:.4f}")

    # Also compute decomposition at other thresholds for reference
    decomp_all_thresholds = {}
    for thr_name, basis in null_basis_by_thr.items():
        P = basis.T @ basis if basis.shape[0] > 0 else np.zeros((512, 512))
        decomp_all_thresholds[thr_name] = {
            "null_dim": int(basis.shape[0]),
            "per_bit_frac_null": [float(np.linalg.norm(P @ probe_weights[b])) for b in range(4)]
        }

    # Random-direction baseline for frac_null
    n_random = 1000
    random_dirs = RNG.standard_normal((n_random, 512))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    rand_frac_null = np.linalg.norm(random_dirs @ P_null.T, axis=1)
    print(f"  random-unit baseline frac_null: mean={rand_frac_null.mean():.4f}, "
          f"std={rand_frac_null.std():.4f}, range=[{rand_frac_null.min():.4f}, "
          f"{rand_frac_null.max():.4f}]")

    # ─── Test 2: class separation on w_i^null vs w_i^row projections ──────
    print("\n[Test 2] Class separation on projections (stable samples, counts 0-14):")
    test2_results = {}
    mask_stable_0_14 = stable & (counts < 15)
    h_stable = h_t[mask_stable_0_14].astype(np.float64)
    bits_stable = bits_actual[mask_stable_0_14]

    def cohens_d(x0, x1):
        pooled_std = np.sqrt((x0.var(ddof=1) * (len(x0) - 1) + x1.var(ddof=1) * (len(x1) - 1)) /
                             (len(x0) + len(x1) - 2))
        return float((x1.mean() - x0.mean()) / pooled_std) if pooled_std > 0 else 0.0

    def auc_separation(x0, x1):
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
        x = np.concatenate([x0, x1])
        return float(roc_auc_score(y, x))

    for b in range(4):
        if decomp[b]["frac_null"] < 1e-10:
            test2_results[b] = {"note": "frac_null ~ 0, no null-projection separation to report"}
            continue
        proj_null = h_stable @ decomp[b]["w_null_hat"]
        proj_row = h_stable @ decomp[b]["w_row_hat"]
        bit_cur = bits_stable[:, b]
        pn0, pn1 = proj_null[bit_cur == 0], proj_null[bit_cur == 1]
        pr0, pr1 = proj_row[bit_cur == 0], proj_row[bit_cur == 1]
        test2_results[b] = {
            "null_cohens_d": cohens_d(pn0, pn1),
            "row_cohens_d": cohens_d(pr0, pr1),
            "null_auc": auc_separation(pn0, pn1),
            "row_auc": auc_separation(pr0, pr1),
            "null_mean_bit0": float(pn0.mean()), "null_mean_bit1": float(pn1.mean()),
            "row_mean_bit0": float(pr0.mean()), "row_mean_bit1": float(pr1.mean()),
        }
        print(f"  bit {b}: null_d={test2_results[b]['null_cohens_d']:+.3f} "
              f"(AUC {test2_results[b]['null_auc']:.3f}), "
              f"row_d={test2_results[b]['row_cohens_d']:+.3f} "
              f"(AUC {test2_results[b]['row_auc']:.3f})")

    # ─── Step 3: intervention comparison ──────────────────────────────────
    print("\n[Step 3] Intervention comparison at α_nat (mean per bit)...")
    step3 = {}
    for b in range(4):
        if b not in alpha_nat_mean:
            continue
        alpha = alpha_nat_mean[b]
        w_full = probe_weights[b]          # unit-normalized
        w_row_unit = decomp[b]["w_row_hat"]  # unit-normalized

        source_counts = sorted(alpha_nat_per_trans[b].keys())
        per_cell_A, per_cell_B = [], []
        flip_outcomes_A, flip_outcomes_B = [], []  # individual sample binary outcomes

        for c in source_counts:
            c_mask = stable & (counts == c)
            h_c = h_t[c_mask][:N_SAMPLES_PER_CELL].astype(np.float64)
            current_bit_b = (c >> b) & 1
            target_bit_b = ((c + 1) >> b) & 1
            # Patch sign: increase h_t · ŵ_b to flip 0→1, decrease to flip 1→0
            sign = +1.0 if current_bit_b == 0 else -1.0
            n = len(h_c)
            if n == 0:
                continue

            flip_A = np.zeros(n, dtype=np.int32)
            flip_B = np.zeros(n, dtype=np.int32)
            for i, h in enumerate(h_c):
                stoch = prior_stoch(h.astype(np.float32), w)
                # Also measure baseline (unpatched) decoded bit
                base_bits = decode_bits(stoch, h.astype(np.float32), w)

                h_A = h + sign * alpha * w_full
                bits_A = decode_bits(stoch, h_A.astype(np.float32), w)
                h_B = h + sign * alpha * w_row_unit
                bits_B = decode_bits(stoch, h_B.astype(np.float32), w)

                # Did patch flip bit b to the target value?
                flip_A[i] = 1 if bits_A[b] == target_bit_b and base_bits[b] != target_bit_b else 0
                flip_B[i] = 1 if bits_B[b] == target_bit_b and base_bits[b] != target_bit_b else 0

            M1_A_c = float(flip_A.mean())
            M1_B_c = float(flip_B.mean())
            per_cell_A.append(M1_A_c)
            per_cell_B.append(M1_B_c)
            flip_outcomes_A.extend(flip_A.tolist())
            flip_outcomes_B.extend(flip_B.tolist())
            print(f"    bit {b}, source c={c}: n={n}, "
                  f"M1_A={M1_A_c:.3f}, M1_B={M1_B_c:.3f}")

        # Aggregate M1 across all samples (not cells) — more statistical power
        M1_A = float(np.mean(flip_outcomes_A))
        M1_B = float(np.mean(flip_outcomes_B))
        ci_A = bca_ci(np.array(flip_outcomes_A), np.mean)
        ci_B = bca_ci(np.array(flip_outcomes_B), np.mean)
        ratio = M1_B / M1_A if M1_A > 0 else float("nan")
        gap = M1_A - M1_B
        step3[b] = {
            "alpha_nat_mean": alpha,
            "source_counts": source_counts,
            "per_cell_M1_A": per_cell_A,
            "per_cell_M1_B": per_cell_B,
            "M1_A": M1_A, "M1_A_ci": ci_A,
            "M1_B": M1_B, "M1_B_ci": ci_B,
            "ratio_B_over_A": ratio,
            "gap_A_minus_B": gap,
            "n_total_samples": len(flip_outcomes_A),
        }
        print(f"  bit {b}: α={alpha:.3f}, M1_A={M1_A:.3f} [{ci_A[0]:.3f}, {ci_A[1]:.3f}], "
              f"M1_B={M1_B:.3f} [{ci_B[0]:.3f}, {ci_B[1]:.3f}], "
              f"ratio={ratio:.3f}, gap={gap:+.3f}")

    # ─── Verdict ──────────────────────────────────────────────────────────
    verdict_per_bit = {}
    for b in step3:
        ratio = step3[b]["ratio_B_over_A"]
        if np.isnan(ratio):
            verdict_per_bit[b] = "degenerate_M1_A=0"
        elif ratio >= 0.80:
            verdict_per_bit[b] = "pass"
        elif ratio < 0.50:
            verdict_per_bit[b] = "fail"
        else:
            verdict_per_bit[b] = "ambiguous"

    overall_verdict = ("pass" if all(v == "pass" for v in verdict_per_bit.values())
                       else "fail" if any(v == "fail" for v in verdict_per_bit.values())
                       else "ambiguous")

    print(f"\n─── Verdict ───")
    for b, v in verdict_per_bit.items():
        print(f"  bit {b}: {v}")
    print(f"  overall: {overall_verdict}")

    # ─── Save results ─────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "step1_decoder_projection": {
            "matrix_used": "dec_linear0_w[:, 1024:] (h_t columns, shape 512x512)",
            "rationale": (
                "Dreamer decoder takes [stoch(1024), deter(512)] concatenated. "
                "The h_t-specific projection is the last 512 columns of dec_linear0_w. "
                "This is the direct analog of Makelov's W_out (the matrix through "
                "which the patched vector leaves the layer)."
            ),
            "max_sv": float(S.max()),
            "min_sv": float(S.min()),
            "sv_ratio_min_over_max": float(S.min() / S.max()),
            "sv_spectrum": sv_spectrum,
            "rank_by_threshold": rank_by_thr,
            "null_dim_by_threshold": {k: 512 - v for k, v in rank_by_thr.items()},
            "primary_threshold_used": primary_thr,
        },
        "step2_decomposition": {
            "primary_threshold": primary_thr,
            "primary_null_dim": int(null_basis.shape[0]),
            "per_bit": {
                str(b): {
                    "frac_null_norm": decomp[b]["frac_null"],
                    "frac_row_norm": decomp[b]["frac_row"],
                    "pythag_sum": decomp[b]["pythag_check"],
                } for b in range(4)
            },
            "all_thresholds": decomp_all_thresholds,
            "random_direction_baseline_primary_threshold": {
                "mean_frac_null": float(rand_frac_null.mean()),
                "std_frac_null": float(rand_frac_null.std()),
                "range": [float(rand_frac_null.min()), float(rand_frac_null.max())],
                "n_samples": n_random,
            },
        },
        "alpha_nat": {
            "definition": "alpha_nat(c, i) = |(mu_{c+1} - mu_c) · w_hat_i| for source counts c where bit i flips",
            "per_bit_per_transition": {
                str(b): {f"{c}->{c+1}": v for c, v in alpha_nat_per_trans[b].items()}
                for b in range(4)
            },
            "per_bit_mean": {str(b): alpha_nat_mean[b] for b in alpha_nat_mean},
            "per_bit_spread": {str(b): alpha_nat_spread[b] for b in alpha_nat_spread},
        },
        "test2_class_separation": {str(b): test2_results[b] for b in test2_results},
        "step3_intervention": {
            str(b): {k: v for k, v in step3[b].items()} for b in step3
        },
        "verdict": {
            "per_bit": {str(b): v for b, v in verdict_per_bit.items()},
            "overall": overall_verdict,
        },
        "methodology_notes": {
            "stoch_source": "prior_stoch computed deterministically from h_t via img_out+imgs_stat heads; avoids re-running episodes",
            "bit_obs_indices": BIT_OBS_INDICES,
            "bit_decode_threshold": "symexp(decoder_output[idx]) > 0.5",
            "samples_per_cell": N_SAMPLES_PER_CELL,
            "n_bootstrap": N_BOOTSTRAP,
            "terminal_state_excluded": "source count c=15 excluded; transition 14->15 excluded from α_nat per split-convention",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    return results


if __name__ == "__main__":
    main()
