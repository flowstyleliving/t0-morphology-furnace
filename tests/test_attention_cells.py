"""Tests for the Attention cell family (t0-candidate #5, landed 2026-05-15).

Fast unit tests cover the helpers (ATTENTION_PANEL shape, label-split, column-name
and cell-label dispatching, _compute_attention_score on synthetic captures).

Slow integration test (@pytest.mark.slow) runs an end-to-end calibration on
Gemma-3-1B with --attention-only panel + the existing puzzle_calibration_jsonl
fixture and verifies that (a) the calibrator can pick an Attention winner,
(b) the detector reloads the profile and self-tests within numerical
tolerance (|reported - deployed AUROC| < 1e-3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pri_calibrator import (
    ATTENTION_FAMILY,
    ATTENTION_LAYERS,
    ATTENTION_METRICS,
    ATTENTION_METRICS_V_NORMS,
    ATTENTION_PANEL,
    ATTENTION_PANEL_MULTISTEP,
    ATTENTION_PANEL_WITH_V_NORMS,
    ATTENTION_STEPS_DEFAULT,
    ATTENTION_STEPS_MULTISTEP,
    DEFAULT_PANEL,
    _build_derivation,
    _cell_label,
    _column_name,
    _compute_attention_score,
    _is_attention_cell,
    _is_v_norm_metric,
    _requires_attention_capture,
    _requires_v_norm_capture,
    _split_attention_label,
    make_attention_panel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fast unit tests — no model load
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionPanelShape:
    """The ATTENTION_PANEL constant is exactly 12 cells, gen_step=1, all unique."""

    def test_panel_has_twelve_cells(self):
        assert len(ATTENTION_PANEL) == 12

    def test_all_cells_are_attention_family(self):
        assert all(cell[1] == ATTENTION_FAMILY for cell in ATTENTION_PANEL)

    def test_all_cells_at_gen_step_1(self):
        assert all(cell[0] == 1 for cell in ATTENTION_PANEL)

    def test_labels_are_unique(self):
        labels = [cell[2] for cell in ATTENTION_PANEL]
        assert len(set(labels)) == len(labels)

    def test_panel_covers_full_layer_metric_cross_product(self):
        expected = {f"{layer}_{metric}" for layer in ATTENTION_LAYERS for metric in ATTENTION_METRICS}
        actual = {cell[2] for cell in ATTENTION_PANEL}
        assert expected == actual


class TestAttentionPredicates:
    def test_is_attention_cell_positive(self):
        for cell in ATTENTION_PANEL:
            assert _is_attention_cell(cell)

    def test_is_attention_cell_negative(self):
        for cell in DEFAULT_PANEL:
            assert not _is_attention_cell(cell)

    def test_requires_attention_capture_true_when_present(self):
        panel = list(DEFAULT_PANEL) + [ATTENTION_PANEL[0]]
        assert _requires_attention_capture(panel)

    def test_requires_attention_capture_false_for_default_panel(self):
        assert not _requires_attention_capture(DEFAULT_PANEL)

    def test_requires_attention_capture_false_for_empty(self):
        assert not _requires_attention_capture([])


class TestSplitAttentionLabel:
    def test_all_panel_labels_split_cleanly(self):
        for cell in ATTENTION_PANEL:
            layer_metric = cell[2]
            parsed = _split_attention_label(layer_metric)
            assert parsed is not None
            layer, metric = parsed
            assert layer in ATTENTION_LAYERS
            assert metric in ATTENTION_METRICS
            assert f"{layer}_{metric}" == layer_metric

    def test_invalid_label_returns_none(self):
        assert _split_attention_label("garbage") is None
        assert _split_attention_label("final_unknown_metric") is None
        assert _split_attention_label("nonexistent_layer_js") is None
        assert _split_attention_label("") is None

    def test_metric_underscores_handled(self):
        # `js_kv_groups`, `js_no_bos`, `bos_mass` all contain underscores
        # and the splitter must anchor on layer prefixes, not naive rsplit.
        for layer in ATTENTION_LAYERS:
            for metric in ("js_kv_groups", "js_no_bos", "bos_mass"):
                parsed = _split_attention_label(f"{layer}_{metric}")
                assert parsed == (layer, metric)


class TestColumnNameAndCellLabel:
    def test_column_name_for_attention(self):
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _column_name(cell) == "attention::final_js_kv_groups"

    def test_cell_label_for_attention(self):
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _cell_label(cell) == "attention[final_js_kv_groups] @ step 1"

    def test_attention_doesnt_break_existing_dispatcher(self):
        # Existing cell types should keep their old labels/columns.
        scalar_cell = (1, "scalar", "d_F_full")
        assert _column_name(scalar_cell) == "d_F_full"
        assert _cell_label(scalar_cell) == "d_F_full @ step 1"
        composite_cell = (1, "Composite", "additive_S_fisher_r=1")
        assert _column_name(composite_cell) == "composite::additive_S_fisher_r=1"


class TestBuildDerivation:
    def test_attention_winners_have_no_derivation(self):
        # The detector reads attention scores from captures directly, not
        # via the derivation payload. _build_derivation must return None so
        # the schema doesn't carry a stale composite/residualized payload.
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _build_derivation(cell, resid_coeffs={}) is None


class TestComputeAttentionScore:
    """Synthetic-captures unit tests. No model load."""

    @staticmethod
    def _synthetic_captures(weights_by_layer):
        """Build a captures dict in the shape attention_capture would produce.

        Each value is a list of attention-weight arrays (one per forward
        call); the calibrator reads index 1 (first gen forward). We pad
        index 0 with a placeholder to mirror real-world layout (prefix
        forward at idx 0, first gen forward at idx 1).
        """
        return {
            layer: [np.zeros((1, 1), dtype=np.float64), w]
            for layer, w in weights_by_layer.items()
        }

    def test_js_metric_on_uniform_heads(self):
        # All heads identical → js_radius == 0 (no head disagreement).
        w = np.ones((4, 8), dtype=np.float64) / 8
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js")
        score = _compute_attention_score(cell, captures, {"final": 4})
        assert score is not None
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_js_metric_on_divergent_heads(self):
        # Two heads concentrate on opposite positions → js_radius > 0.
        w = np.zeros((2, 4), dtype=np.float64)
        w[0, 0] = 1.0
        w[1, 3] = 1.0
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js")
        score = _compute_attention_score(cell, captures, {"final": 2})
        assert score is not None
        assert score > 0.0

    def test_bos_mass_metric(self):
        # Half the heads on BOS, half spread; mean BOS mass = 0.5.
        w = np.zeros((4, 4), dtype=np.float64)
        w[0, 0] = 1.0  # full BOS
        w[1, 0] = 1.0  # full BOS
        w[2] = 0.25    # uniform → 0.25 BOS
        w[3] = 0.25
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_bos_mass")
        score = _compute_attention_score(cell, captures, {"final": 4})
        assert score is not None
        assert score == pytest.approx(0.625, abs=1e-6)  # (1 + 1 + 0.25 + 0.25)/4

    def test_js_no_bos_drops_bos_column(self):
        # BOS column dominates; after dropping it + renorming, heads agree.
        w = np.zeros((2, 3), dtype=np.float64)
        w[0] = [0.99, 0.005, 0.005]
        w[1] = [0.99, 0.005, 0.005]
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js_no_bos")
        score = _compute_attention_score(cell, captures, {"final": 2})
        assert score is not None
        assert score == pytest.approx(0.0, abs=1e-6)  # identical post-drop

    def test_js_kv_groups_requires_n_kv_heads(self):
        w = np.ones((4, 8), dtype=np.float64) / 8
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _compute_attention_score(cell, captures, {}) is None
        assert _compute_attention_score(cell, captures, {"final": 2}) is not None

    def test_js_kv_groups_collapses_repeats(self):
        # 4 q-heads sharing 2 kv-groups; pairs are identical pre-collapse.
        # Post-collapse: still 2 distinct distributions → js_radius matches
        # the un-collapsed value (because grouping doesn't change the
        # distinct positions). Sanity check: doesn't return NaN/None.
        w = np.zeros((4, 4), dtype=np.float64)
        w[0] = [1, 0, 0, 0]
        w[1] = [1, 0, 0, 0]
        w[2] = [0, 0, 0, 1]
        w[3] = [0, 0, 0, 1]
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        score = _compute_attention_score(cell, captures, {"final": 2})
        assert score is not None
        assert score > 0.0  # heads disagree post-grouping

    def test_missing_captures_returns_none(self):
        cell = (1, ATTENTION_FAMILY, "final_js")
        assert _compute_attention_score(cell, {}, {}) is None

    def test_too_few_capture_entries_returns_none(self):
        # captures[tag] must have ≥ 2 entries (prefix + first gen forward).
        captures = {"final": [np.zeros((1, 1))]}
        cell = (1, ATTENTION_FAMILY, "final_js")
        assert _compute_attention_score(cell, captures, {"final": 1}) is None

    def test_step_zero_returns_none(self):
        w = np.ones((2, 4), dtype=np.float64) / 4
        captures = self._synthetic_captures({"final": w})
        cell = (0, ATTENTION_FAMILY, "final_js")  # step 0 invalid
        assert _compute_attention_score(cell, captures, {"final": 2}) is None

    def test_step_past_captures_returns_none(self):
        w = np.ones((2, 4), dtype=np.float64) / 4
        # captures[final] = [prefix, step1] → step 2 is past available data
        captures = self._synthetic_captures({"final": w})
        cell = (2, ATTENTION_FAMILY, "final_js")
        assert _compute_attention_score(cell, captures, {"final": 2}) is None

    def test_non_attention_family_returns_none(self):
        captures = self._synthetic_captures({"final": np.ones((2, 4)) / 4})
        # Calling with a non-Attention cell is defensive — should refuse.
        cell = (1, "scalar", "d_F_full")
        assert _compute_attention_score(cell, captures, {"final": 2}) is None


class TestMultistepAttention:
    """Multi-step attention panel — steps 2-4 added 2026-05-15 evening."""

    def test_multistep_steps_constant(self):
        assert ATTENTION_STEPS_DEFAULT == (1,)
        assert ATTENTION_STEPS_MULTISTEP == (1, 2, 3, 4)

    def test_multistep_panel_has_48_cells(self):
        # 3 layers × 4 metrics × 4 steps = 48
        assert len(ATTENTION_PANEL_MULTISTEP) == 48

    def test_multistep_panel_supersets_default_panel(self):
        default_set = set(ATTENTION_PANEL)
        multistep_set = set(ATTENTION_PANEL_MULTISTEP)
        assert default_set.issubset(multistep_set)

    def test_multistep_steps_cover_one_through_four(self):
        steps = {cell[0] for cell in ATTENTION_PANEL_MULTISTEP}
        assert steps == {1, 2, 3, 4}

    def test_multistep_cells_unique(self):
        # No duplicate (step, family, label) tuples.
        assert len(set(ATTENTION_PANEL_MULTISTEP)) == 48

    def test_make_attention_panel_default_matches_constant(self):
        rebuilt = make_attention_panel(ATTENTION_STEPS_DEFAULT)
        assert rebuilt == ATTENTION_PANEL

    def test_make_attention_panel_custom_steps(self):
        # User-defined panel — e.g. just step 2 + step 4
        panel = make_attention_panel(steps=(2, 4))
        assert len(panel) == 24  # 2 steps × 3 layers × 4 metrics
        steps = {cell[0] for cell in panel}
        assert steps == {2, 4}

    def test_compute_attention_score_step_2(self):
        # Step 2 needs captures[layer] to have at least 3 entries
        # (prefix + 2 gen forwards). Build captures with 3 entries.
        w0 = np.zeros((1, 1), dtype=np.float64)  # prefix placeholder
        w1 = np.ones((2, 4), dtype=np.float64) / 4  # step 1
        w2 = np.zeros((2, 4), dtype=np.float64)  # step 2 — heads concentrate on different positions
        w2[0, 0] = 1.0
        w2[1, 3] = 1.0
        captures = {"final": [w0, w1, w2]}
        cell_step1 = (1, ATTENTION_FAMILY, "final_js")
        cell_step2 = (2, ATTENTION_FAMILY, "final_js")
        score1 = _compute_attention_score(cell_step1, captures, {"final": 2})
        score2 = _compute_attention_score(cell_step2, captures, {"final": 2})
        assert score1 is not None
        assert score2 is not None
        assert score1 == pytest.approx(0.0, abs=1e-6)  # uniform heads
        assert score2 > 0.0  # divergent heads


class TestVNormAttention:
    """V-norm SinkProbe-style metrics — added 2026-05-15 evening."""

    @staticmethod
    def _make_captures(weights_by_layer, v_norms_by_layer=None):
        """Build (captures, v_norm_captures) in the wrapper's per-call list
        shape. Both maps are layer → list of arrays where index 0 is the
        prefix placeholder and index 1 is the gen_step=1 forward."""
        captures = {
            layer: [np.zeros((1, 1)), w] for layer, w in weights_by_layer.items()
        }
        if v_norms_by_layer is None:
            return captures, None
        v_caps = {
            layer: [np.zeros((1, 1)), v] for layer, v in v_norms_by_layer.items()
        }
        return captures, v_caps

    def test_v_norms_panel_has_21_cells(self):
        assert len(ATTENTION_PANEL_WITH_V_NORMS) == 21  # 3 layers × (4 + 3) metrics

    def test_v_norms_panel_includes_default_cells(self):
        # The default 12 attention cells should be in the v-norms panel.
        assert set(ATTENTION_PANEL).issubset(set(ATTENTION_PANEL_WITH_V_NORMS))

    def test_v_norm_metric_names(self):
        assert ATTENTION_METRICS_V_NORMS == (
            "v_norm_bos", "v_norm_max", "v_norm_lastq_weighted",
        )

    def test_is_v_norm_metric_positive(self):
        for m in ATTENTION_METRICS_V_NORMS:
            assert _is_v_norm_metric(m)

    def test_is_v_norm_metric_negative(self):
        for m in ATTENTION_METRICS:
            assert not _is_v_norm_metric(m)

    def test_requires_v_norm_capture_true(self):
        panel = [(1, ATTENTION_FAMILY, "final_v_norm_bos")]
        assert _requires_v_norm_capture(panel)

    def test_requires_v_norm_capture_false_for_weight_only(self):
        assert not _requires_v_norm_capture(ATTENTION_PANEL)

    def test_requires_v_norm_capture_false_for_default(self):
        assert not _requires_v_norm_capture(DEFAULT_PANEL)

    def test_v_norm_label_split(self):
        for metric in ATTENTION_METRICS_V_NORMS:
            for layer in ATTENTION_LAYERS:
                parsed = _split_attention_label(f"{layer}_{metric}")
                assert parsed == (layer, metric)

    def test_v_norm_bos_correctness(self):
        # ‖V_0‖ averaged over 4 KV heads. v_norms[:, 0] = [1.0, 2.0, 3.0, 4.0],
        # mean = 2.5.
        v = np.array(
            [[1.0, 0.5, 0.2], [2.0, 0.5, 0.2], [3.0, 0.5, 0.2], [4.0, 0.5, 0.2]],
            dtype=np.float64,
        )
        w = np.ones((4, 3), dtype=np.float64) / 3
        captures, v_caps = self._make_captures({"final": w}, {"final": v})
        cell = (1, ATTENTION_FAMILY, "final_v_norm_bos")
        score = _compute_attention_score(
            cell, captures, {"final": 4}, v_norm_captures=v_caps,
        )
        assert score == pytest.approx(2.5, abs=1e-6)

    def test_v_norm_max_correctness(self):
        # max_i v_norms[h, i] for each h, then mean over h.
        # Per-head maxes: h0 max=1, h1 max=2, h2 max=3, h3 max=4 → mean = 2.5.
        v = np.array(
            [[0.1, 0.5, 1.0], [0.1, 2.0, 0.5], [0.1, 0.5, 3.0], [4.0, 0.5, 0.1]],
            dtype=np.float64,
        )
        w = np.ones((4, 3), dtype=np.float64) / 3
        captures, v_caps = self._make_captures({"final": w}, {"final": v})
        cell = (1, ATTENTION_FAMILY, "final_v_norm_max")
        score = _compute_attention_score(
            cell, captures, {"final": 4}, v_norm_captures=v_caps,
        )
        assert score == pytest.approx(2.5, abs=1e-6)

    def test_v_norm_lastq_weighted_correctness(self):
        # 2 KV heads, 2 Q heads (MHA case, n_q == n_kv == 2). Heads:
        # h0 attends fully to position 0; h1 attends fully to position 2.
        # ‖V‖ per head: h0=[10, 1, 1], h1=[1, 1, 100].
        # Per-head Σ A_i · ‖V_i‖ = h0: 10, h1: 100. Mean over heads = 55.
        v = np.array([[10.0, 1.0, 1.0], [1.0, 1.0, 100.0]], dtype=np.float64)
        w = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        captures, v_caps = self._make_captures({"final": w}, {"final": v})
        cell = (1, ATTENTION_FAMILY, "final_v_norm_lastq_weighted")
        score = _compute_attention_score(
            cell, captures, {"final": 2}, v_norm_captures=v_caps,
        )
        assert score == pytest.approx(55.0, abs=1e-6)

    def test_v_norm_metric_returns_none_without_v_captures(self):
        # If v_norm_captures is None, v-norm cells should return None.
        w = np.ones((2, 4), dtype=np.float64) / 4
        captures, _ = self._make_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_v_norm_bos")
        score = _compute_attention_score(cell, captures, {"final": 2}, v_norm_captures=None)
        assert score is None


# ─────────────────────────────────────────────────────────────────────────────
# Slow integration test — Gemma-3-1B end-to-end
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestAttentionEndToEnd:
    """End-to-end: calibrate Gemma-3-1B with the attention panel, then
    reload the profile in a Detector and confirm byte-exact self-test
    parity. Mirrors test_pri_detector.py::TestRuntime."""

    def test_calibrate_then_self_test(
        self, gemma_3_1b_slug, puzzle_calibration_jsonl, tmp_path
    ):
        from pri_calibrator import calibrate
        from pri_detector import Detector, _self_test

        # Use the attention panel only to keep the smoke focused.
        profile = calibrate(
            model_slug=gemma_3_1b_slug,
            calibration_jsonl_path=str(puzzle_calibration_jsonl),
            task_label="attention_e2e_smoke",
            panel=list(ATTENTION_PANEL),
            n_bootstrap=50,
            max_new_tokens=4,
        )

        # Profile should record the attention wrapper hash since the panel
        # contained Attention cells.
        assert profile.provenance.get("attention_wrapper_module_hash_sha256") is not None

        # Winner should be an Attention cell.
        assert profile.detector["metric"]["family"] == ATTENTION_FAMILY

        # Persist + reload + self-test.
        profile_path = tmp_path / "gemma_3_1b_attention.profile.json"
        profile.to_json(str(profile_path))
        rc = _self_test(str(profile_path), str(puzzle_calibration_jsonl))
        assert rc == 0, "detector self-test failed for attention winner profile"

    def test_calibrate_then_self_test_with_v_norms(
        self, gemma_3_1b_slug, puzzle_calibration_jsonl, tmp_path
    ):
        """V-norm panel end-to-end. Exercises the
        attention_capture_with_values context-manager wiring on both the
        calibration side (calibrate_with_state) and the deploy side
        (Detector._score_attention's v-norm branch).
        """
        from pri_calibrator import calibrate
        from pri_detector import _self_test

        profile = calibrate(
            model_slug=gemma_3_1b_slug,
            calibration_jsonl_path=str(puzzle_calibration_jsonl),
            task_label="attention_v_norms_e2e_smoke",
            panel=list(ATTENTION_PANEL_WITH_V_NORMS),
            n_bootstrap=50,
            max_new_tokens=4,
        )

        assert profile.detector["metric"]["family"] == ATTENTION_FAMILY
        assert profile.provenance.get("attention_wrapper_module_hash_sha256") is not None

        profile_path = tmp_path / "gemma_3_1b_attention_v_norms.profile.json"
        profile.to_json(str(profile_path))
        rc = _self_test(str(profile_path), str(puzzle_calibration_jsonl))
        assert rc == 0, "detector self-test failed for v-norm attention winner profile"
