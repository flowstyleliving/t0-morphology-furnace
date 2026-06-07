"""Fast unit tests for scripts/sinkprobe_baseline.py — the pure SinkProbe math.

No model load. Covers _topk_sum, the 6 per-layer reductions (attention-mass +
‖V‖-weighted incl. the GQA expand), the head-layout guards, and the
fixed-direction + sign-free AUROC pair.

Run:
    .venv/bin/pytest tests/test_sinkprobe_baseline.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.sinkprobe_baseline import (
    TOP_K,
    _auroc_pair,
    _topk_sum,
    sink_metrics,
)


# ── _topk_sum ───────────────────────────────────────────────────────────────

def test_topk_sum_basic():
    assert _topk_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3) == pytest.approx(12.0)


def test_topk_sum_fewer_than_k_sums_all():
    assert _topk_sum(np.array([1.0, 2.0]), 4) == pytest.approx(3.0)


def test_topk_sum_nan_filtered_and_empty():
    assert _topk_sum(np.array([1.0, np.nan, 4.0, 5.0]), 2) == pytest.approx(9.0)
    assert math.isnan(_topk_sum(np.array([]), 3))
    assert math.isnan(_topk_sum(np.array([np.nan, np.nan]), 2))


# ── sink_metrics: attention-mass ────────────────────────────────────────────

def test_sink_metrics_mass_hand_computed():
    sink = np.array([[0.9, 0.1, 0.2], [0.7, 0.3, 0.1]])  # (H=2, T=3)
    vn = np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 3.0]])     # (n_kv=2, T=3)
    m = sink_metrics(sink, vn, n_q=2, n_kv=2, k=4)
    assert m["sink_bos"] == pytest.approx(0.8)            # mean(0.9, 0.7)
    assert m["sink_top1"] == pytest.approx(0.8)           # mean(max row)
    assert m["sink_topk_sum"] == pytest.approx(1.15)      # T<k → sum all: (1.2+1.1)/2


def test_sink_metrics_topk_respects_k():
    sink = np.array([[0.9, 0.1, 0.2], [0.7, 0.3, 0.1]])
    vn = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    m = sink_metrics(sink, vn, n_q=2, n_kv=2, k=2)
    # head0 top2 = 0.9+0.2 = 1.1 ; head1 top2 = 0.7+0.3 = 1.0 ; mean = 1.05
    assert m["sink_topk_sum"] == pytest.approx(1.05)


# ── sink_metrics: ‖V‖-weighted ──────────────────────────────────────────────

def test_sink_metrics_vweighted_no_gqa():
    sink = np.array([[0.9, 0.1, 0.2], [0.7, 0.3, 0.1]])
    vn = np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 3.0]])
    m = sink_metrics(sink, vn, n_q=2, n_kv=2, k=4)
    # sv = sink*vn = [[1.8,0.1,0.2],[0.7,0.3,0.3]]
    assert m["sink_bos_vw"] == pytest.approx(1.25)        # mean(1.8, 0.7)
    assert m["sink_top1_vw"] == pytest.approx(1.25)
    assert m["sink_topk_sum_vw"] == pytest.approx(1.7)    # (2.1+1.3)/2


def test_sink_metrics_vweighted_gqa_expand():
    # n_q=4, n_kv=2, repeats=2 → v rows map [v0,v0,v1,v1].
    sink = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])  # (4,2)
    vn = np.array([[2.0, 1.0], [5.0, 1.0]])                            # (2,2)
    m = sink_metrics(sink, vn, n_q=4, n_kv=2, k=4)
    # v_per_q = [[2,1],[2,1],[5,1],[5,1]] ; sv col0 = [2,4,15,20] ; mean = 10.25
    assert m["sink_bos_vw"] == pytest.approx(10.25)


def test_sink_metrics_gqa_mismatch_nans_only_vw():
    sink = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.2]])  # H=3
    vn = np.array([[1.0, 1.0], [1.0, 1.0]])                # n_kv=2 ; 3 % 2 != 0
    m = sink_metrics(sink, vn, n_q=3, n_kv=2, k=4)
    assert math.isfinite(m["sink_bos"]) and math.isfinite(m["sink_top1"])
    assert math.isnan(m["sink_bos_vw"])
    assert math.isnan(m["sink_top1_vw"])
    assert math.isnan(m["sink_topk_sum_vw"])


def test_sink_metrics_vnorm_shape_mismatch_nans_vw():
    sink = np.array([[0.9, 0.1], [0.7, 0.3]])
    vn = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # T mismatch (3 vs 2)
    m = sink_metrics(sink, vn, n_q=2, n_kv=2, k=4)
    assert math.isfinite(m["sink_bos"])
    assert math.isnan(m["sink_bos_vw"])


def test_sink_metrics_degenerate_all_nan():
    m = sink_metrics(np.empty((0, 0)), np.empty((0, 0)), n_q=0, n_kv=0, k=4)
    assert all(math.isnan(v) for v in m.values())


def test_default_k_is_four():
    assert TOP_K == 4


# ── _auroc_pair ─────────────────────────────────────────────────────────────

def test_auroc_pair_separable_and_inverted():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    hi = _auroc_pair(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert hi["auroc"] == pytest.approx(1.0) and hi["direction"] == "hi"
    assert hi["auroc_signfree"] == pytest.approx(1.0) and hi["n_scored"] == 6
    lo = _auroc_pair(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))
    assert lo["auroc"] == pytest.approx(0.0) and lo["direction"] == "lo"
    assert lo["auroc_signfree"] == pytest.approx(1.0)


def test_auroc_pair_too_few_and_nan_count():
    y = np.array([0, 1, 0, 1], dtype=np.float64)
    ap = _auroc_pair(y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert ap["auroc"] is None and ap["auroc_signfree"] is None
    y6 = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    ap2 = _auroc_pair(y6, np.array([0.1, 0.2, np.nan, 0.7, 0.8, 0.9]))
    assert ap2["n_scored"] == 5
