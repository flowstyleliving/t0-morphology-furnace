#!/usr/bin/env python3
"""Contract tests — research-candidate #10 (shadow-ambiguity).

This file is the self-contained numpy reference for the proposed
delta-h-independent centered-Fisher readout statistics. It does not load a
model and does not require MLX for checks 1-5 or 7.

Checks:

  1. Definitional spectrum identities:
     flat, spiked, scale-invariant effective-rank/entropy behavior, plus
     participation_ratio as a separate statistic.
  2. Temperature Fisher identity:
     Hessian_z KL(softmax(z/T) || softmax((z+δ)/T))
     == (1/T²)(diag(p_T) - p_T p_Tᵀ).
  3. Temperature flattening direction:
     as T increases over [0.5, 2], effective rank of the vocab bracket is
     non-decreasing on a deterministic synthetic example.
  4. Degeneracy guard:
     the near-one-hot effective-rank -> 1 claim is asserted only on a
     spiked/rank-one limiting spectrum; guarded log-volume remains finite.
  5. h-space-vs-vocab trap:
     the V-1 uniform-bracket limit is vocab-space only; h-space F_c depends
     on W and can have a very different effective rank.
  6. Production cross-check (fails hard, does not skip):
     with fake mlx + sklearn modules injected before import, compare reference
     F_c eigenspectrum energy fractions AND null-ratio against the inherited
     centered-Fisher readout core.
  7. Cheap pre-check demo, print-only:
     synthetic temperature sweep prints Spearman rho between surprise and
     fisher_eff_rank(F_c), with no hard threshold.

Location: t0-morphology-furnace/exploratory/shadow-ambiguity/ -- the FORWARD
(unsealed) morphology lab, kept out of the sealed tests/ suite (pytest
`testpaths = tests` does not collect it).

Run with any numpy-capable python (scipy optional), e.g.:
    python3 exploratory/shadow-ambiguity/test_shadow_ambiguity.py

Exit 0 on pass, 1 on fail. Check 6 imports the inherited centered-Fisher
core after installing fake mlx/sklearn modules; it FAILS HARD (does not SKIP)
on import error, since it is the only production-drift guard.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
from typing import Callable, Iterable

import numpy as np


THIS = os.path.dirname(os.path.abspath(__file__))
# This file lives under <repo>/exploratory/shadow-ambiguity/; walk up to the
# repo root (the directory holding pri_runtime.py) so the guarded production
# cross-check (check 6) can import it regardless of nesting depth.
ROOT = THIS
while ROOT != os.path.dirname(ROOT):
    if os.path.exists(os.path.join(ROOT, "pri_runtime.py")):
        break
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

# Relative tolerance for numerical rank in spectrum-only statistics. A mode is
# active when lambda_i > REL_TOL * lambda_max. This keeps exact/roundoff zeros
# out of entropy normalization while retaining scale invariance.
REL_TOL = 1e-12

# Epsilon used only for eps-guarded pseudo-log-volume. It is deliberately much
# smaller than synthetic nonzero eigenvalues, so it regularizes zeros without
# moving ordinary spectra.
LOGVOL_EPS = 1e-12

PRODUCTION_STATUS = "not run"


# ─────────────────────────────────────────────────────────────────────────────
#  Reference statistics
# ─────────────────────────────────────────────────────────────────────────────


def clean_spectrum(lam: Iterable[float]) -> np.ndarray:
    """Return nonnegative eigenvalues sorted descending."""
    arr = np.asarray(lam, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    return np.sort(arr)[::-1]


def active_spectrum(
    lam: Iterable[float],
    rel_tol: float = REL_TOL,
) -> np.ndarray:
    """Return numerically active nonnegative eigenvalues sorted descending.

    NOTE (design tradeoff): thresholding the spectrum at rel_tol * lambda_max
    makes the active-set statistics below (fisher_spectral_entropy and
    fisher_eff_rank) DISCONTINUOUS as an eigenvalue crosses rel_tol * lambda_max
    -- a mode flips in/out of the active set at that boundary. This is a
    deliberate numerical-rank floor for this contract: it stops arbitrarily many
    roundoff-scale modes from moving entropy / effective rank. The production
    statistic should revisit threshold-vs-soft (e.g. a smooth floor) once it is
    fed real, roundoff-tailed F_c spectra.
    """
    vals = clean_spectrum(lam)
    if vals.size == 0 or vals[0] <= 0.0:
        return vals[:0]
    return vals[vals > rel_tol * vals[0]]


def fisher_spectral_entropy(lam: Iterable[float], rel_tol: float = REL_TOL) -> float:
    """Normalized Shannon entropy of the centered-Fisher spectrum in nats.

    The normalization rank is count(lambda_i > rel_tol * lambda_max), with
    rel_tol=1e-12 by default.
    """
    vals = active_spectrum(lam, rel_tol=rel_tol)
    rank_eff = int(vals.size)
    if rank_eff == 0:
        return 0.0
    if rank_eff <= 1:
        return 0.0
    p_tilde = vals / np.sum(vals)
    H = -float(np.sum(p_tilde * np.log(p_tilde)))
    return float(np.clip(H / math.log(rank_eff), 0.0, 1.0))


def fisher_eff_rank(lam: Iterable[float], rel_tol: float = REL_TOL) -> float:
    """Thresholded Roy-Vetterli effective rank: exp(H(lambda/sum(lambda))).

    Uses the same active-set threshold as fisher_spectral_entropy (see the
    discontinuity NOTE on active_spectrum).
    """
    vals = active_spectrum(lam, rel_tol=rel_tol)
    total = float(np.sum(vals))
    if total <= 0.0:
        return 0.0
    p_tilde = vals / total
    active = p_tilde > 0.0
    H = -float(np.sum(p_tilde[active] * np.log(p_tilde[active])))
    return float(math.exp(H))


def participation_ratio(lam: Iterable[float]) -> float:
    """Participation ratio: (sum lambda)^2 / sum(lambda^2).

    NOTE: deliberately left UN-thresholded (uses the full clean spectrum, not
    active_spectrum). Unlike entropy / effective rank, (sum lambda)^2 /
    sum(lambda^2) is naturally robust to roundoff-scale modes -- they contribute
    negligibly to both numerator and denominator -- so it needs no rel_tol floor
    and stays continuous.
    """
    vals = clean_spectrum(lam)
    total = float(np.sum(vals))
    denom = float(np.sum(vals * vals))
    if total <= 0.0 or denom <= 0.0:
        return 0.0
    return float((total * total) / denom)


def shadow_logvol_post_rank(
    lam: Iterable[float],
    r: int,
    eps: float = LOGVOL_EPS,
) -> float:
    """Per-direction mean log pseudo-volume over off-top-r eigendirections."""
    vals = clean_spectrum(lam)
    r_eff = max(int(r), 0)
    tail = vals[r_eff:]
    denom = max(len(vals) - r_eff, 1)
    return float(-0.5 * np.sum(np.log(tail + eps)) / denom)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    shifted = z - np.max(z)
    e = np.exp(shifted)
    return e / np.sum(e)


def centered_fisher(p: np.ndarray, W: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    W_centered = W - np.sum(p[:, None] * W, axis=0, keepdims=True)
    F = np.einsum("v,vi,vj->ij", p, W_centered, W_centered, optimize=True)
    return 0.5 * (F + F.T)


def fisher_spectrum_from_p_W(p: np.ndarray, W: np.ndarray) -> np.ndarray:
    return clean_spectrum(np.linalg.eigvalsh(centered_fisher(p, W)))


def vocab_bracket_spectrum(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    B = np.diag(p) - np.outer(p, p)
    return clean_spectrum(np.linalg.eigvalsh(0.5 * (B + B.T)))


# ─────────────────────────────────────────────────────────────────────────────
#  Checks
# ─────────────────────────────────────────────────────────────────────────────


def test_definitional_spectrum_identities() -> None:
    k = 7
    flat = np.full(k, 3.25)
    assert np.isclose(fisher_eff_rank(flat), k, atol=1e-9)
    assert np.isclose(fisher_spectral_entropy(flat), 1.0, atol=1e-12)

    spiked = np.array([1.0] + [1e-15] * 9)
    assert fisher_eff_rank(spiked) < 1.0 + 1e-12
    assert fisher_spectral_entropy(spiked) == 0.0

    lam = np.array([9.0, 4.0, 1.0, 0.25, 0.0])
    for c in [0.125, 3.0, 1.0 / (1.7**2)]:
        assert np.isclose(fisher_eff_rank(c * lam), fisher_eff_rank(lam), atol=1e-12)
        assert np.isclose(
            fisher_spectral_entropy(c * lam),
            fisher_spectral_entropy(lam),
            atol=1e-12,
        )

    assert np.isclose(participation_ratio(flat), k, atol=1e-12)
    assert participation_ratio(spiked) < 1.0 + 1e-12
    distinct = np.array([0.70, 0.20, 0.10])
    assert not np.isclose(
        participation_ratio(distinct),
        fisher_eff_rank(distinct),
        rtol=1e-3,
        atol=1e-3,
    )

    asc = np.array([1.0, 9.0, 4.0], dtype=np.float64)
    eps = LOGVOL_EPS
    n = int(asc.size)
    expected_r0 = -0.5 * (
        math.log(9.0 + eps) + math.log(4.0 + eps) + math.log(1.0 + eps)
    ) / 3.0
    expected_r1 = -0.5 * (
        math.log(4.0 + eps) + math.log(1.0 + eps)
    ) / 2.0
    assert np.isclose(shadow_logvol_post_rank(asc, r=0, eps=eps), expected_r0)
    assert np.isclose(shadow_logvol_post_rank(asc, r=1, eps=eps), expected_r1)
    assert np.isclose(shadow_logvol_post_rank(asc, r=n, eps=eps), -0.0)
    assert np.isclose(shadow_logvol_post_rank(asc, r=n + 1, eps=eps), -0.0)

    asc_with_neg = np.array([-1e-14, 1.0, 9.0, 4.0], dtype=np.float64)
    expected_neg_r1 = -0.5 * (
        math.log(4.0 + eps) + math.log(1.0 + eps) + math.log(eps)
    ) / 3.0
    assert np.isclose(
        shadow_logvol_post_rank(asc_with_neg, r=1, eps=eps), expected_neg_r1
    )
    print(
        "(1) PASS  flat/spiked spectra, scale invariance, and "
        "effective-rank != participation-ratio in general"
    )


def _kl_from_delta(z: np.ndarray, T: float, delta: np.ndarray) -> float:
    p = softmax(z / T)
    q = softmax((z + delta) / T)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _central_hessian_kl(z: np.ndarray, T: float, step: float = 1e-4) -> np.ndarray:
    n = int(z.shape[0])
    H = np.zeros((n, n), dtype=np.float64)
    f0 = _kl_from_delta(z, T, np.zeros(n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        H[i, i] = (
            _kl_from_delta(z, T, ei)
            - 2.0 * f0
            + _kl_from_delta(z, T, -ei)
        ) / (step * step)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            H_ij = (
                _kl_from_delta(z, T, ei + ej)
                - _kl_from_delta(z, T, ei - ej)
                - _kl_from_delta(z, T, -ei + ej)
                + _kl_from_delta(z, T, -ei - ej)
            ) / (4.0 * step * step)
            H[i, j] = H_ij
            H[j, i] = H_ij
    return 0.5 * (H + H.T)


def test_temperature_fisher_identity() -> None:
    z = np.array([1.3, -0.4, 0.2, 2.1, -1.2], dtype=np.float64)
    for T in [0.7, 1.4, 2.0]:
        p_T = softmax(z / T)
        bracket = np.diag(p_T) - np.outer(p_T, p_T)
        expected = bracket / (T * T)
        fd_hessian = _central_hessian_kl(z, T)
        assert np.allclose(fd_hessian, expected, atol=1e-6), (
            f"T={T}: finite-difference Hessian mismatch"
        )
    print(
        "(2) PASS  Hessian_z KL equals (1/T^2)(diag(p_T)-p_T p_T^T) "
        "across temperatures"
    )


def test_temperature_flattening_direction() -> None:
    z = np.array([3.0, 1.25, 0.1, -0.7, -1.6, -2.4], dtype=np.float64)
    Ts = np.linspace(0.5, 2.0, 7)
    eff_ranks = []
    for T in Ts:
        p_T = softmax(z / T)
        eff_ranks.append(fisher_eff_rank(vocab_bracket_spectrum(p_T)))
    diffs = np.diff(eff_ranks)
    assert np.all(diffs >= -1e-10), f"effective rank decreased: {eff_ranks}"
    print(
        "(3) PASS  temperature flattening increases vocab-bracket "
        f"effective rank ({eff_ranks[0]:.3f} -> {eff_ranks[-1]:.3f})"
    )


def test_degeneracy_guard_rank_one_spike() -> None:
    # The "eff_rank -> 1 as p -> one-hot" claim is not universal for arbitrary
    # W when the vanishing mass is spread across many alternatives. This fixture
    # is deliberately spiked/rank-one: only one alternative carries the
    # vanishing mass, so F_c has one nonzero mode in the limit.
    v = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    W = np.zeros((5, 3), dtype=np.float64)
    W[1] = v
    p_looser = np.array([1.0 - 1e-3, 1e-3, 0.0, 0.0, 0.0])
    p_tighter = np.array([1.0 - 1e-8, 1e-8, 0.0, 0.0, 0.0])

    lam_looser = fisher_spectrum_from_p_W(p_looser, W)
    lam_tighter = fisher_spectrum_from_p_W(p_tighter, W)
    assert lam_tighter[0] < 1e-4 * lam_looser[0]

    sign, logabsdet = np.linalg.slogdet(centered_fisher(p_tighter, W))
    raw_logvol = math.inf if sign == 0.0 and logabsdet == -math.inf else -0.5 * float(logabsdet)
    assert math.isinf(raw_logvol) and raw_logvol > 0.0

    guarded_logvol = shadow_logvol_post_rank(lam_tighter, r=1, eps=LOGVOL_EPS)
    eff_rank = fisher_eff_rank(lam_tighter)
    assert np.isfinite(guarded_logvol)
    assert np.isfinite(eff_rank)
    assert eff_rank < 1.0 + 1e-8
    print(
        "(4) PASS  rank-one near-one-hot: lambda_max -> 0, raw logdet -> "
        "+inf, guarded logvol and effective rank stay finite"
    )


def test_h_space_vs_vocab_trap() -> None:
    rng = np.random.default_rng(20260607)
    V, d = 16, 6
    p = np.full(V, 1.0 / V)

    bracket_lam = vocab_bracket_spectrum(p)
    assert int(np.count_nonzero(bracket_lam > REL_TOL * bracket_lam[0])) == V - 1
    vocab_eff = fisher_eff_rank(bracket_lam)
    assert np.isclose(vocab_eff, V - 1, atol=1e-10)

    ones = np.ones((V, 1), dtype=np.float64) / math.sqrt(V)
    P = np.eye(V) - ones @ ones.T
    Q, _ = np.linalg.qr(P @ rng.standard_normal((V, d)))
    singular_values = np.geomspace(1.0, 1e-4, d)
    W = Q[:, :d] @ np.diag(singular_values)

    # The resolvability statistic is the h-space F_c spectrum; the V-1 limit
    # is a vocab-space property only.
    h_lam = fisher_spectrum_from_p_W(p, W)
    h_eff = fisher_eff_rank(h_lam)
    assert not np.isclose(h_eff, V - 1, rtol=1e-3, atol=1e-3)
    assert not np.isclose(h_eff, vocab_eff, rtol=1e-3, atol=1e-3)
    print(
        "(5) PASS  uniform vocab bracket has eff_rank V-1, while h-space "
        f"F_c is W-dependent ({vocab_eff:.3f} vs {h_eff:.3f})"
    )


class StubOutputProjection:
    """Minimal OutputProjection stand-in: serves a fixed (V, d) numpy W_u."""

    def __init__(self, W: np.ndarray):
        assert W.ndim == 2, "W must be (V, d)"
        self._W = W.astype(np.float64)
        self.vocab_size = int(W.shape[0])
        self.hidden_size = int(W.shape[1])
        self._raw_svd_cache = None

    def project(self, hidden_vec: np.ndarray) -> np.ndarray:
        return (self._W @ hidden_vec.astype(np.float64)).astype(np.float32)

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        return self._W[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def _install_fake_mlx_modules() -> None:
    mlx_mod = types.ModuleType("mlx")
    core_mod = types.ModuleType("mlx.core")

    core_mod.array = lambda x, *args, **kwargs: np.asarray(x)
    core_mod.eval = lambda *args, **kwargs: None
    core_mod.take = lambda a, indices, axis=0: np.take(np.asarray(a), indices, axis=axis)
    core_mod.dequantize = lambda w, scales, biases=None: np.asarray(w)
    core_mod.float32 = np.float32
    core_mod.clear_cache = lambda: None

    metal_mod = types.SimpleNamespace(clear_cache=lambda: None)
    core_mod.metal = metal_mod
    mlx_mod.core = core_mod

    mlx_lm_mod = types.ModuleType("mlx_lm")
    mlx_lm_mod.load = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("fake mlx_lm.load is not available in this test")
    )
    mlx_lm_mod.generate = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("fake mlx_lm.generate is not available in this test")
    )

    sys.modules["mlx"] = mlx_mod
    sys.modules["mlx.core"] = core_mod
    sys.modules["mlx_lm"] = mlx_lm_mod

    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("sklearn.metrics")

    def _fake_roc_auc_score(*args, **kwargs):
        raise RuntimeError(
            "fake sklearn.metrics.roc_auc_score is not available in this test"
        )

    metrics_mod.roc_auc_score = _fake_roc_auc_score
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod


def test_guarded_production_cross_check() -> None:
    global PRODUCTION_STATUS
    _install_fake_mlx_modules()
    try:
        pipeline = importlib.import_module("pri_runtime")
    except ModuleNotFoundError as exc:
        missing = exc.name or "<unknown>"
        PRODUCTION_STATUS = f"failed: missing dependency {missing}"
        raise AssertionError(
            f"production import failed; missing dependency {missing}"
        ) from exc
    except Exception as exc:
        PRODUCTION_STATUS = f"failed: {type(exc).__name__}: {exc}"
        raise AssertionError(f"production import failed: {type(exc).__name__}: {exc}") from exc

    rng = np.random.default_rng(1001)
    V, d = 14, 6
    W = rng.standard_normal((V, d)) * 0.2
    p = softmax(rng.standard_normal(V) * 0.7)
    dh = rng.standard_normal(d) * 0.1
    rank_values = [1, 2, 3, 6]

    pri = pipeline.PRIComputer(StubOutputProjection(W), final_norm_gamma=None)
    out = pri.kl_discharged_and_centered(dh, p, rank_values=rank_values)

    F_c = centered_fisher(p, W)
    eigvals_raw, eigvecs_raw = np.linalg.eigh(F_c)
    eigvals = np.maximum(eigvals_raw[::-1], 0.0)
    eigvecs = eigvecs_raw[:, ::-1]
    cum_eig = np.cumsum(eigvals) / (float(np.sum(eigvals)) + 1e-12)

    z = W @ dh
    mu = float(np.dot(p, z))
    kl_ref = 0.5 * float(np.dot(p, (z - mu) ** 2))
    assert np.isclose(out["kl_discharged"], kl_ref, atol=1e-9)

    proj = eigvecs.T @ dh
    kl_per_dir = 0.5 * eigvals * (proj ** 2)
    cum_kl = np.cumsum(kl_per_dir)

    for r in rank_values:
        r_eff = int(min(r, len(eigvals)))
        method_energy = out[f"fisher_energy_centered_rank{r}"]
        direct_energy = float(cum_eig[r_eff - 1]) if r_eff > 0 else 0.0
        assert np.isclose(method_energy, direct_energy, atol=1e-7), (
            f"rank={r}: energy method {method_energy:.12f} vs direct {direct_energy:.12f}"
        )
        kl_topr = float(cum_kl[r_eff - 1]) if r_eff > 0 else 0.0
        direct_null = 0.0 if kl_ref <= 1e-12 else max(kl_ref - kl_topr, 0.0) / kl_ref
        method_null = out[f"null_ratio_centered_post_rank{r}"]
        assert np.isclose(method_null, direct_null, atol=1e-7), (
            f"rank={r}: null_ratio method {method_null:.12f} vs direct {direct_null:.12f}"
        )

    PRODUCTION_STATUS = "passed"
    print("(6) PASS  production centered eigendecomp/null-ratio matches reference F_c")


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        rho = spearmanr(x, y).statistic
        return float(rho)
    except Exception:
        rx = _rankdata_average_ties(np.asarray(x, dtype=np.float64))
        ry = _rankdata_average_ties(np.asarray(y, dtype=np.float64))
        rx = rx - np.mean(rx)
        ry = ry - np.mean(ry)
        denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
        if denom == 0.0:
            return float("nan")
        return float(np.dot(rx, ry) / denom)


def demo_temperature_decorrelation() -> None:
    rng = np.random.default_rng(424242)
    V, d, n_samples = 24, 8, 32
    W = rng.standard_normal((V, d)) * 0.35
    Ts = np.linspace(0.5, 2.0, 7)
    surprises = []
    eff_ranks = []
    for _ in range(n_samples):
        h = rng.standard_normal(d)
        logits = W @ h + 0.15 * rng.standard_normal(V)
        top = int(np.argmax(logits))
        for T in Ts:
            p_T = softmax(logits / T)
            surprises.append(-math.log(max(float(p_T[top]), 1e-300)))
            eff_ranks.append(fisher_eff_rank(fisher_spectrum_from_p_W(p_T, W)))
    rho = _spearman_rho(np.asarray(surprises), np.asarray(eff_ranks))
    print(
        "(7) PASS  demo_temperature_decorrelation: "
        f"Spearman rho(surprise, eff_rank) = {rho:.3f}; "
        "|rho|>0.9 => statistic is confidence-in-disguise "
        "(candidate dies at this pre-check)"
    )


def test_demo_temperature_decorrelation() -> None:
    demo_temperature_decorrelation()


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    tests: list[Callable[[], None]] = [
        test_definitional_spectrum_identities,
        test_temperature_fisher_identity,
        test_temperature_flattening_direction,
        test_degeneracy_guard_rank_one_spike,
        test_h_space_vs_vocab_trap,
        test_guarded_production_cross_check,
        test_demo_temperature_decorrelation,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures} / {len(tests)} FAIL")
        return 1
    print(f"ALL {len(tests)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
