"""Fast unit tests for scripts/rauq_at_commit.py — the pure RAUQ math.

No model load. Covers head-select (unsupervised, position-pooled), the 1a
commit-only score, the 1b prompt-recurrence (hand-computed reference values
incl. the α=1 memoryless degenerate case), NaN/empty guards, and the
fixed-direction + sign-free AUROC pair.

Run:
    .venv/bin/pytest tests/test_rauq_at_commit.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.rauq_at_commit import (
    ALPHA,
    _auroc_pair,
    score_1a,
    score_1b,
    select_heads,
)


# ── select_heads ────────────────────────────────────────────────────────────

def test_select_heads_picks_highest_mean_subdiag():
    # 3 heads; head 1 has the largest a_{i,i-1} everywhere.
    s1 = np.array([[0.1, 0.1], [0.8, 0.9], [0.3, 0.2]], dtype=np.float32)
    s2 = np.array([[0.2, 0.0], [0.7, 0.7], [0.4, 0.1]], dtype=np.float32)
    h, mean = select_heads([s1, s2])
    assert h == 1
    # pooled over all 4 positions of head 1: (0.8+0.9+0.7+0.7)/4 = 0.775
    assert mean == pytest.approx(0.775, abs=1e-6)


def test_select_heads_pools_positions_not_per_sample_mean():
    # A long low sample + a short high sample. Position-pooling (RAUQ's
    # mean-over-tokens) must let the many low positions outweigh the few
    # high ones for head 0 — distinguishing it from a mean-of-per-sample.
    long_low = np.full((2, 100), 0.10, dtype=np.float32)
    short_high = np.array([[0.90], [0.05]], dtype=np.float32)
    h, mean = select_heads([long_low, short_high])
    # head 0 pooled: (0.10*100 + 0.90)/101 ≈ 0.1079; head 1: (0.10*100+0.05)/101 ≈ 0.0995
    assert h == 0
    assert mean == pytest.approx((0.10 * 100 + 0.90) / 101, abs=1e-6)


def test_select_heads_skips_empty_subdiags():
    empty = np.empty((3, 0), dtype=np.float32)
    good = np.array([[0.1, 0.1], [0.9, 0.9], [0.2, 0.2]], dtype=np.float32)
    h, _ = select_heads([empty, good])
    assert h == 1


def test_select_heads_raises_on_no_input():
    with pytest.raises(ValueError):
        select_heads([])


def test_select_heads_raises_when_all_empty():
    with pytest.raises(ValueError):
        select_heads([np.empty((4, 0), dtype=np.float32)])


# ── score_1a ────────────────────────────────────────────────────────────────

def test_score_1a_is_one_minus_g():
    assert score_1a(0.0) == pytest.approx(1.0)
    assert score_1a(0.25) == pytest.approx(0.75)
    assert score_1a(1.0) == pytest.approx(0.0)


def test_score_1a_nan_passthrough():
    assert math.isnan(score_1a(float("nan")))


# ── score_1b ────────────────────────────────────────────────────────────────

def test_score_1b_hand_computed_alpha_half():
    # prefix g = [0.2, 0.6], commit_g = 0.4, α = 0.5
    #   u1 = 1 - 0.2 = 0.8
    #   t1 = 0.5*(1-0.6) + 0.5*0.8 = 0.6
    #   commit = 0.5*(1-0.4) + 0.5*0.6 = 0.6
    out = score_1b(np.array([0.2, 0.6]), 0.4, alpha=0.5)
    assert out == pytest.approx(0.6, abs=1e-9)


def test_score_1b_default_alpha_is_module_constant():
    assert ALPHA == 0.5
    assert score_1b(np.array([0.2, 0.6]), 0.4) == pytest.approx(0.6, abs=1e-9)


def test_score_1b_alpha_one_is_memoryless():
    # α=1 collapses the recurrence to 1 - g_commit regardless of the prefix.
    out = score_1b(np.array([0.2, 0.6, 0.9]), 0.4, alpha=1.0)
    assert out == pytest.approx(1.0 - 0.4, abs=1e-9)


def test_score_1b_single_prompt_token():
    # prefix g = [0.0] → u1 = 1.0 ; commit_g 0.0 → 0.5*1 + 0.5*1 = 1.0
    assert score_1b(np.array([0.0]), 0.0, alpha=0.5) == pytest.approx(1.0)


def test_score_1b_guards():
    assert math.isnan(score_1b(np.array([]), 0.5))
    assert math.isnan(score_1b(np.array([0.3]), float("nan")))
    # NaN positions in the prefix are filtered before the recurrence runs.
    clean = score_1b(np.array([0.2, 0.6]), 0.4, alpha=0.5)
    with_nan = score_1b(np.array([0.2, np.nan, 0.6]), 0.4, alpha=0.5)
    assert with_nan == pytest.approx(clean, abs=1e-9)


# ── _auroc_pair ─────────────────────────────────────────────────────────────

def test_auroc_pair_separable_fixed_direction():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    ap = _auroc_pair(y, scores)
    assert ap["auroc"] == pytest.approx(1.0)
    assert ap["direction"] == "hi"
    assert ap["auroc_signfree"] == pytest.approx(1.0)
    assert ap["n_scored"] == 6


def test_auroc_pair_inverted_keeps_low_direction_signfree_recovers():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    ap = _auroc_pair(y, scores)
    assert ap["auroc"] == pytest.approx(0.0)
    assert ap["direction"] == "lo"
    assert ap["auroc_signfree"] == pytest.approx(1.0)


def test_auroc_pair_too_few_samples_is_none():
    y = np.array([0, 1, 0, 1], dtype=np.float64)
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    ap = _auroc_pair(y, scores)
    assert ap["auroc"] is None
    assert ap["auroc_signfree"] is None


def test_auroc_pair_counts_only_finite_scores():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    scores = np.array([0.1, 0.2, np.nan, 0.7, 0.8, 0.9])
    ap = _auroc_pair(y, scores)
    assert ap["n_scored"] == 5
