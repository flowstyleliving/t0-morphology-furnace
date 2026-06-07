"""Tests for pri_calibrator.py — schema dataclass, pure helpers, scoring,
and end-to-end calibration (slow tier).

Run with:
    .venv/bin/pytest tests/test_pri_calibrator.py -m "not slow"   # fast unit tests
    .venv/bin/pytest tests/test_pri_calibrator.py -m slow         # model-load tests
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pri_calibrator import (  # noqa: E402
    CalibrationProfile,
    CalibrationState,
    DEFAULT_PANEL,
    SCHEMA_VERSION,
    _bootstrap_auroc,
    _cell_label,
    _column_name,
    _compose_score,
    _emit_warnings,
    _load_calibration_jsonl,
    _nested_bootstrap_oob_auroc,
    _residualize_in_place,
    _score_candidate,
    calibrate,
    calibrate_with_state,
    load_calibration_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationProfile dataclass — schema v1.0
# ─────────────────────────────────────────────────────────────────────────────


class TestCalibrateWithStateSignature:
    """Regression tests for the 2026-05-13 calibrate_with_state refactor that
    Codex's adversarial review flagged. Catches the seed-NameError bug at
    parse time without needing a model load."""

    def test_calibrate_with_state_does_not_accept_seed_kwarg(self):
        """The split moved `seed` from a kwarg into state.seed. Anyone passing
        `seed=` to calibrate_with_state() should get TypeError, not silent
        misuse."""
        import inspect
        sig = inspect.signature(calibrate_with_state)
        assert "seed" not in sig.parameters, (
            "calibrate_with_state must not accept a `seed` kwarg — the seed "
            "lives in CalibrationState.seed now."
        )

    def test_calibrate_with_state_function_body_references_state_seed(self):
        """Catches the regression Codex found: the body referencing bare
        `seed` after the refactor. Source-level grep is a low-tech but
        effective unit-level guard."""
        import inspect
        src = inspect.getsource(calibrate_with_state)
        # Bootstrap calls must use state.seed (not a bare `seed` variable).
        assert "n_bootstrap, state.seed" in src or "state.seed," in src, (
            "calibrate_with_state appears not to thread state.seed into "
            "bootstrap calls — bare `seed` reference would NameError at runtime"
        )

    def test_calibration_state_dataclass_has_seed_field(self):
        """CalibrationState must carry seed since calibrate_with_state needs it."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(CalibrationState)}
        assert "seed" in fields
        assert "model_slug" in fields
        assert "model" in fields
        assert "pri_computer" in fields
        assert "prompt_strategy" in fields


class TestProfile:
    def test_to_json_roundtrip(self, tmp_path, synthetic_profile_dict):
        p = CalibrationProfile(**synthetic_profile_dict)
        out = tmp_path / "p.json"
        p.to_json(str(out))
        p2 = CalibrationProfile.from_json(str(out))
        assert p.detector == p2.detector
        assert p.model == p2.model
        assert p.task == p2.task
        assert p.calibration_stats == p2.calibration_stats
        assert p.provenance == p2.provenance
        assert p.warnings == p2.warnings

    def test_schema_version_mismatch_rejected(self, tmp_path, synthetic_profile_dict):
        synthetic_profile_dict["schema_version"] = "99.0"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(synthetic_profile_dict))
        with pytest.raises(ValueError, match=r"schema 99\.0 != supported 1\.2"):
            CalibrationProfile.from_json(str(path))

    def test_warnings_default_empty_list(self, synthetic_profile_dict):
        d = dict(synthetic_profile_dict)
        d.pop("warnings", None)
        p = CalibrationProfile(**d)
        assert p.warnings == []
        # Mutating one instance's warnings must not leak to another's default.
        p.warnings.append("x")
        d2 = dict(synthetic_profile_dict)
        d2.pop("warnings", None)
        p2 = CalibrationProfile(**d2)
        assert p2.warnings == []


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — no model, no PRIComputer
# ─────────────────────────────────────────────────────────────────────────────


class TestPureHelpers:
    # _column_name table
    @pytest.mark.parametrize("cell,expected", [
        ((1, "scalar", "d_F_full"), "d_F_full"),
        ((1, "scalar", "kl_discharged"), "kl_discharged"),
        ((1, "Fisher", "r=1"), "null_ratio_post_rank1"),
        ((3, "Fisher", "r=2"), "null_ratio_post_rank2"),
        ((1, "Centered", "r=2"), "null_ratio_centered_post_rank2"),
        ((1, "Centered", "r=4"), "null_ratio_centered_post_rank4"),
        ((4, "Raw", "r=2"), "null_ratio_raw_post_rank2"),
        ((3, "Raw", "r=21"), "null_ratio_raw_post_rank21"),
    ])
    def test_column_name_known_families(self, cell, expected):
        assert _column_name(cell) == expected

    def test_column_name_unknown_family_raises(self):
        with pytest.raises(ValueError, match="unknown family"):
            _column_name((1, "Banana", "r=1"))

    @pytest.mark.parametrize("cell,expected", [
        ((1, "scalar", "d_F_full"), "d_F_full @ step 1"),
        ((1, "scalar", "kl_discharged"), "kl_discharged @ step 1"),
        ((3, "Raw", "r=21"), "Raw r=21 @ step 3"),
        ((4, "Centered", "r=4"), "Centered r=4 @ step 4"),
    ])
    def test_cell_label(self, cell, expected):
        assert _cell_label(cell) == expected

    def test_default_panel_shape(self):
        # 2026-05-13/14 panel expansion (post-Codex fix):
        #   8 direct cells + 3 residualized (step-1 only, where d_F_full
        #   lives in the panel) + 4 composites (additive + gated) = 15.
        assert len(DEFAULT_PANEL) == 15
        for cell in DEFAULT_PANEL:
            assert len(cell) == 3
            step, fam, label = cell
            assert isinstance(step, int) and step >= 1
            assert fam in {
                "scalar", "Fisher", "Raw", "Centered",
                "Fisher_resid", "Raw_resid", "Centered_resid",
                "Composite",
            }
            # _column_name should accept every panel cell
            _column_name(cell)


class TestLoadCalibrationJsonl:
    def test_minimal(self, tmp_calibration_jsonl):
        path = tmp_calibration_jsonl([("hello world", 0), ("goodbye", 1)])
        prompts, labels, h = _load_calibration_jsonl(str(path))
        assert prompts == ["hello world", "goodbye"]
        assert list(labels) == [0, 1]
        assert isinstance(h, str) and len(h) == 64  # sha256 hex

    def test_deterministic_hash_same_content(self, tmp_calibration_jsonl):
        rows = [("a", 0), ("b", 1), ("c", 0)]
        p1 = tmp_calibration_jsonl(rows, filename="cal1.jsonl")
        p2 = tmp_calibration_jsonl(rows, filename="cal2.jsonl")
        _, _, h1 = _load_calibration_jsonl(str(p1))
        _, _, h2 = _load_calibration_jsonl(str(p2))
        assert h1 == h2

    def test_hash_changes_on_reorder(self, tmp_calibration_jsonl):
        p1 = tmp_calibration_jsonl([("a", 0), ("b", 1)], filename="cal1.jsonl")
        p2 = tmp_calibration_jsonl([("b", 1), ("a", 0)], filename="cal2.jsonl")
        _, _, h1 = _load_calibration_jsonl(str(p1))
        _, _, h2 = _load_calibration_jsonl(str(p2))
        assert h1 != h2

    def test_invalid_label_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"prompt": "x", "label": 2}) + "\n")
        with pytest.raises(ValueError, match=r"label must be 0 or 1"):
            _load_calibration_jsonl(str(path))

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        # Was SystemExit until 2026-05-13 — library helpers now raise
        # RuntimeError so callers (sweep runner) can catch failures.
        with pytest.raises(RuntimeError, match="no calibration samples"):
            _load_calibration_jsonl(str(path))

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "with_blanks.jsonl"
        path.write_text(
            json.dumps({"prompt": "a", "label": 0}) + "\n"
            "\n"
            + json.dumps({"prompt": "b", "label": 1}) + "\n"
        )
        prompts, labels, _ = _load_calibration_jsonl(str(path))
        assert prompts == ["a", "b"]


# ─────────────────────────────────────────────────────────────────────────────
# Warning logic
# ─────────────────────────────────────────────────────────────────────────────


class TestEmitWarnings:
    HEALTHY_KWARGS = dict(
        n_calibration=30, n_pos=15, n_neg=15,
        best_auroc=0.91, ci_lo=0.78, ci_hi=1.00,
        panel_eval_counts={(1, "scalar", "d_F_full"): 30},
    )

    def test_clean_run_no_warnings(self):
        w = _emit_warnings(**self.HEALTHY_KWARGS)
        assert w == []

    def test_small_n_fires(self):
        kw = dict(self.HEALTHY_KWARGS, n_calibration=10, n_pos=5, n_neg=5,
                  panel_eval_counts={(1, "scalar", "d_F_full"): 10})
        w = _emit_warnings(**kw)
        assert any("small_calibration_n" in x for x in w)

    def test_low_auroc_fires(self):
        kw = dict(self.HEALTHY_KWARGS, best_auroc=0.55)
        w = _emit_warnings(**kw)
        assert any("low_auroc" in x for x in w)

    def test_wide_ci_fires(self):
        kw = dict(self.HEALTHY_KWARGS, ci_lo=0.55, ci_hi=0.99)  # width 0.44
        w = _emit_warnings(**kw)
        assert any("wide_ci" in x for x in w)

    def test_class_imbalance_fires(self):
        kw = dict(self.HEALTHY_KWARGS, n_pos=24, n_neg=6,
                  panel_eval_counts={(1, "scalar", "d_F_full"): 30})
        w = _emit_warnings(**kw)
        assert any("class_imbalance" in x for x in w)

    def test_compose_multiple(self):
        kw = dict(self.HEALTHY_KWARGS, n_calibration=8, n_pos=4, n_neg=4,
                  best_auroc=0.55,
                  panel_eval_counts={(1, "scalar", "d_F_full"): 8})
        w = _emit_warnings(**kw)
        assert any("small_calibration_n" in x for x in w)
        assert any("low_auroc" in x for x in w)

    def test_insufficient_coverage_fires(self):
        kw = dict(self.HEALTHY_KWARGS,
                  panel_eval_counts={
                      (1, "scalar", "d_F_full"): 30,   # full
                      (4, "Raw", "r=2"): 10,           # 10/30 < 80% — fires
                  })
        w = _emit_warnings(**kw)
        assert any("insufficient_coverage_at_" in x for x in w)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + bootstrap (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────


class TestScoring:
    def test_perfect_separation_positive_sign(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["perfect_pos"]
        auc, sign, n = _score_candidate(scores, labels)
        assert auc == pytest.approx(1.0)
        assert sign == 1
        assert n == 10

    def test_perfect_separation_negative_sign(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["perfect_neg"]
        auc, sign, n = _score_candidate(scores, labels)
        # Sign-agnostic AUROC still 1.0; sign is negative.
        assert auc == pytest.approx(1.0)
        assert sign == -1

    def test_too_few_samples(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["tiny"]
        auc, sign, n = _score_candidate(scores, labels)
        assert np.isnan(auc)
        assert sign == 0

    def test_single_class(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["single_class"]
        auc, sign, n = _score_candidate(scores, labels)
        assert np.isnan(auc)
        assert sign == 0

    def test_nan_scores_dropped(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["with_nans"]
        # 7 finite scores remain: positions 0,2,4,5,7,8,9 → labels [0,0,0,1,1,1,1]
        auc, sign, n = _score_candidate(scores, labels)
        assert n == 7
        assert auc == pytest.approx(1.0)  # finite half is perfectly separable
        assert sign == 1

    def test_random_data_near_chance(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["random"]
        auc, sign, n = _score_candidate(scores, labels)
        # Should be near 0.5 but sign-agnostic AUROC is in [0.5, 1]
        assert 0.5 <= auc <= 1.0
        assert sign in {-1, 1}

    def test_bootstrap_ci_brackets_point_perfect(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["perfect_pos"]
        ci_lo, ci_hi = _bootstrap_auroc(scores, labels, sign=1, n_bootstrap=500, seed=42)
        # Perfect separation + locked sign → every bootstrap resample also perfect
        assert ci_lo == pytest.approx(1.0)
        assert ci_hi == pytest.approx(1.0)

    def test_bootstrap_ci_seed_deterministic(self, synthetic_scores_and_labels):
        scores, labels = synthetic_scores_and_labels["random"]
        a1 = _bootstrap_auroc(scores, labels, sign=1, n_bootstrap=200, seed=42)
        a2 = _bootstrap_auroc(scores, labels, sign=1, n_bootstrap=200, seed=42)
        assert a1 == a2

    def test_bootstrap_ci_locked_sign_inverts_for_wrong_direction(
        self, synthetic_scores_and_labels
    ):
        """If we lock the WRONG sign on perfectly-separable data, the bootstrap
        CI should collapse near 0.0 (anti-discrimination), not 1.0. This is
        the property that makes direction-preserving scoring honest."""
        scores, labels = synthetic_scores_and_labels["perfect_pos"]
        ci_lo, ci_hi = _bootstrap_auroc(scores, labels, sign=-1, n_bootstrap=500, seed=42)
        assert ci_lo == pytest.approx(0.0)
        assert ci_hi == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Nested OOB bootstrap (Codex review fix for post-selection bias)
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedBootstrap:
    def test_oob_with_separable_data_recovers_high_auroc(self):
        """Two cells in the panel: one is perfect, one is pure noise.
        Nested bootstrap should pick the perfect cell in most rounds,
        winner_stability should be high, and OOB AUROC should be near 1.0.
        """
        n = 30
        rng = np.random.RandomState(0)
        labels = np.array([0]*15 + [1]*15, dtype=np.int32)
        # Column 0: perfectly separable (will be picked); column 1: noise.
        score_matrix = np.column_stack([
            np.array([float(y) + rng.normal(0, 0.05) for y in labels]),
            rng.randn(n),
        ])
        panel = [(1, "scalar", "d_F_full"), (1, "scalar", "kl_discharged")]
        stats = _nested_bootstrap_oob_auroc(
            score_matrix, labels, panel, n_bootstrap=500, seed=42,
        )
        assert stats["oob_n_bootstrap_used"] > 400
        assert stats["oob_auroc_median"] > 0.9
        # Winner should be the separable cell on almost every round.
        assert stats["winner_stability"] > 0.9
        # The dominant winner label should match cell 0 (d_F_full).
        assert "d_F_full @ step 1" in stats["winner_counts"]

    def test_oob_unstable_when_two_cells_tie(self):
        """Two cells with identical noisy signal: winner choice is roughly
        50/50 across resamples, winner_stability should be ~0.5."""
        n = 30
        rng = np.random.RandomState(1)
        labels = np.array([0]*15 + [1]*15, dtype=np.int32)
        # Both columns: same weak signal + independent noise → tie.
        base = np.array([float(y) for y in labels])
        score_matrix = np.column_stack([
            base + rng.normal(0, 1.0, size=n),
            base + rng.normal(0, 1.0, size=n),
        ])
        panel = [(1, "scalar", "d_F_full"), (1, "scalar", "kl_discharged")]
        stats = _nested_bootstrap_oob_auroc(
            score_matrix, labels, panel, n_bootstrap=500, seed=42,
        )
        # Neither cell should dominate — stability well below 0.9 (the value
        # we'd get if one cell were truly better).
        assert stats["winner_stability"] < 0.9

    def test_oob_seed_deterministic(self):
        n = 20
        rng = np.random.RandomState(2)
        labels = np.array([0]*10 + [1]*10, dtype=np.int32)
        score_matrix = rng.randn(n, 3)
        panel = [(1, "scalar", "d_F_full"), (1, "scalar", "kl_discharged"), (1, "Fisher", "r=1")]
        s1 = _nested_bootstrap_oob_auroc(score_matrix, labels, panel, n_bootstrap=200, seed=7)
        s2 = _nested_bootstrap_oob_auroc(score_matrix, labels, panel, n_bootstrap=200, seed=7)
        # Float comparison is exact because we use np.random with the same seed.
        assert s1["oob_auroc_median"] == s2["oob_auroc_median"]
        assert s1["oob_auroc_ci_lo"] == s2["oob_auroc_ci_lo"]
        assert s1["oob_auroc_ci_hi"] == s2["oob_auroc_ci_hi"]
        assert s1["winner_stability"] == s2["winner_stability"]


class TestDerivedCells:
    """Coverage for the 2026-05-13 panel expansion — composites + residuals.
    Validates that the new cells compute correctly from primitive columns
    and that residualization actually removes the d_F_full magnitude
    confound (E18 sealed acceptance form)."""

    def test_compose_additive_S_fisher(self):
        result = {"surprise": 0.5, "null_ratio_post_rank1": 0.95}
        v = _compose_score(result, "additive_S_fisher_r=1")
        assert v == pytest.approx(1.45)

    def test_compose_gated_dF_fisher(self):
        result = {"d_F_full": 2.0, "null_ratio_post_rank1": 0.95}
        v = _compose_score(result, "gated_dF_fisher_r=1")
        assert v == pytest.approx(1.90)

    def test_compose_additive_S_centered(self):
        result = {"surprise": 0.2, "null_ratio_centered_post_rank2": 0.88}
        v = _compose_score(result, "additive_S_centered_r=2")
        assert v == pytest.approx(1.08)

    def test_compose_returns_none_on_missing_primitive(self):
        # No d_F_full in result → can't compute gated; must return None.
        result = {"null_ratio_post_rank1": 0.95}
        assert _compose_score(result, "gated_dF_fisher_r=1") is None

    def test_compose_returns_none_on_nan(self):
        result = {"surprise": float("nan"), "null_ratio_post_rank1": 0.95}
        assert _compose_score(result, "additive_S_fisher_r=1") is None

    def test_compose_unknown_op_returns_none(self):
        result = {"surprise": 0.5, "null_ratio_post_rank1": 0.95}
        assert _compose_score(result, "subtractive_S_fisher_r=1") is None

    def test_residualize_strips_d_F_confound(self):
        """Construct null_ratio = 0.9 + 0.05 * d_F + small_noise. After
        regressing out d_F, the residual should be ~the noise — small and
        un-correlated with d_F. AUROC of the residual should be near chance
        for any d_F-aligned label set."""
        rng = np.random.RandomState(0)
        n = 30
        d_F = rng.uniform(1.0, 5.0, size=n)
        noise = rng.normal(0, 0.01, size=n)
        null_ratio = 0.9 + 0.05 * d_F + noise
        # Build a panel with just d_F_full + one Fisher_resid cell.
        panel = [
            (1, "scalar", "d_F_full"),
            (1, "Fisher_resid", "r=1"),
        ]
        score_matrix = np.column_stack([d_F, null_ratio])
        _residualize_in_place(score_matrix, panel, prompts_n=n)
        # After residualization, the resid column should be near zero-mean
        # (and equal to the noise term within numerical tolerance).
        residual = score_matrix[:, 1]
        assert np.allclose(np.mean(residual), 0.0, atol=1e-9)
        # Residual should match the original noise within bootstrap noise
        # (the regression recovers the linear part exactly when no noise
        # in the model, so here the residual ≈ noise centered to mean 0).
        assert np.std(residual) < 0.05  # small (just the noise term)
        # And the residual should not be strongly correlated with d_F any more.
        corr = np.corrcoef(d_F, residual)[0, 1]
        assert abs(corr) < 1e-6  # OLS residuals are orthogonal to predictor by construction

    def test_residualize_handles_nan(self):
        """If some samples have NaN in either column, those rows' residuals
        must remain NaN (not be silently fabricated by the regression)."""
        n = 20
        rng = np.random.RandomState(1)
        d_F = rng.uniform(1, 5, size=n)
        null_ratio = 0.9 + 0.05 * d_F + rng.normal(0, 0.01, n)
        # NaN out a couple samples
        null_ratio[5] = np.nan
        d_F[10] = np.nan
        panel = [
            (1, "scalar", "d_F_full"),
            (1, "Fisher_resid", "r=1"),
        ]
        score_matrix = np.column_stack([d_F, null_ratio])
        _residualize_in_place(score_matrix, panel, prompts_n=n)
        assert np.isnan(score_matrix[5, 1])
        assert np.isnan(score_matrix[10, 1])
        # Other rows: residual is finite
        finite_idx = [i for i in range(n) if i not in (5, 10)]
        for i in finite_idx:
            assert np.isfinite(score_matrix[i, 1])

    def test_residualize_returns_coefficients(self):
        """v1.2 requires coefficients persisted so Detector can reproduce
        the residual at score time. _residualize_in_place returns a dict
        mapping each successfully-residualized cell → (b0, b1, base_column,
        regress_against, regress_against_step)."""
        rng = np.random.RandomState(0)
        n = 20
        d_F = rng.uniform(1, 5, n)
        null_ratio = 0.9 + 0.05 * d_F + rng.normal(0, 0.01, n)
        panel = [
            (1, "scalar", "d_F_full"),
            (1, "Fisher_resid", "r=1"),
        ]
        score_matrix = np.column_stack([d_F, null_ratio])
        coeffs = _residualize_in_place(score_matrix, panel, prompts_n=n)
        assert (1, "Fisher_resid", "r=1") in coeffs
        entry = coeffs[(1, "Fisher_resid", "r=1")]
        # Coefficients recovered within tolerance from data we generated
        assert abs(entry["b0"] - 0.9) < 0.05
        assert abs(entry["b1"] - 0.05) < 0.02
        assert entry["base_column"] == "null_ratio_post_rank1"
        assert entry["regress_against"] == "d_F_full"
        assert entry["regress_against_step"] == 1

    def test_residualize_no_d_F_at_step_marks_nan(self):
        """Codex review fix (2026-05-14): residualizing a step-N cell when
        the panel only has d_F_full at step 1 must NOT silently regress
        against the wrong step. Instead the cell column is filled with NaN."""
        rng = np.random.RandomState(1)
        n = 20
        d_F_step1 = rng.uniform(1, 5, n)
        null_ratio_step3 = rng.uniform(0.85, 0.95, n)
        panel = [
            (1, "scalar", "d_F_full"),
            (3, "Fisher_resid", "r=2"),   # No d_F_full @ step 3 in panel!
        ]
        score_matrix = np.column_stack([d_F_step1, null_ratio_step3])
        coeffs = _residualize_in_place(score_matrix, panel, prompts_n=n)
        # Cross-step residualization must NOT have been performed
        assert (3, "Fisher_resid", "r=2") not in coeffs
        # Cell column is NaN — caller can detect "no usable resid here"
        assert np.all(np.isnan(score_matrix[:, 1]))

    def test_residualize_too_few_samples_marks_nan(self):
        """If <3 finite pairs exist (regression needs at least 3), the
        whole resid column should be NaN — don't ship a profile claiming
        a residualized cell when there isn't enough data to fit."""
        n = 4
        score_matrix = np.array([
            [1.0, 0.91],
            [np.nan, 0.92],
            [3.0, np.nan],
            [np.nan, np.nan],
        ])
        panel = [
            (1, "scalar", "d_F_full"),
            (1, "Fisher_resid", "r=1"),
        ]
        _residualize_in_place(score_matrix, panel, prompts_n=n)
        assert np.all(np.isnan(score_matrix[:, 1]))


class TestWarningsOob:
    HEALTHY_KWARGS = dict(
        n_calibration=30, n_pos=15, n_neg=15,
        best_auroc=0.91, ci_lo=0.78, ci_hi=1.00,
        panel_eval_counts={(1, "scalar", "d_F_full"): 30},
    )

    def test_oob_low_auroc_fires(self):
        kw = dict(self.HEALTHY_KWARGS,
                  oob_auroc_median=0.50, winner_stability=0.9)
        w = _emit_warnings(**kw)
        assert any("oob_low_auroc" in x for x in w)

    def test_large_oob_in_sample_gap_fires(self):
        # in-sample 0.91, OOB 0.65 → gap 0.26 > 0.15
        kw = dict(self.HEALTHY_KWARGS,
                  oob_auroc_median=0.65, winner_stability=0.9)
        w = _emit_warnings(**kw)
        assert any("large_oob_in_sample_gap" in x for x in w)

    def test_winner_unstable_fires(self):
        kw = dict(self.HEALTHY_KWARGS,
                  oob_auroc_median=0.85, winner_stability=0.4)
        w = _emit_warnings(**kw)
        assert any("winner_unstable" in x for x in w)

    def test_oob_warnings_silent_when_healthy(self):
        kw = dict(self.HEALTHY_KWARGS,
                  oob_auroc_median=0.85, winner_stability=0.9)
        w = _emit_warnings(**kw)
        assert not any("oob_low_auroc" in x for x in w)
        assert not any("large_oob_in_sample_gap" in x for x in w)
        assert not any("winner_unstable" in x for x in w)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end calibration (slow — loads a real model)
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    @pytest.mark.slow
    def test_calibrate_e2e_gemma_3_1b(
        self, puzzle_calibration_jsonl, gemma_3_1b_slug
    ):
        """Full calibrate() on 20 synthetic puzzles + smallest cached model.
        Asserts profile structure + plausible bounds, not exact AUROC."""
        profile = calibrate(
            model_slug=gemma_3_1b_slug,
            calibration_jsonl_path=str(puzzle_calibration_jsonl),
            task_label="unit_test_puzzles",
            n_bootstrap=200,
            max_new_tokens=8,
            seed=42,
        )
        assert profile.schema_version == SCHEMA_VERSION
        assert profile.model["slug"] == gemma_3_1b_slug
        assert profile.model["output_projection_kind"] in {"lm_head", "tied_embed"}
        assert profile.task["n_calibration"] == 20
        assert profile.task["n_pos"] + profile.task["n_neg"] == 20
        assert len(profile.task["data_hash_sha256"]) == 64

        # Detector contract
        assert profile.detector["gen_step"] >= 1
        assert profile.detector["layer"] == "final"
        assert profile.detector["alpha"] == 1.0
        assert profile.detector["sign"] in {-1, 1}
        assert profile.detector["metric"]["family"] in {"scalar", "Fisher", "Raw", "Centered"}

        # In-sample stats (post-selection, legacy semantics)
        auc = profile.calibration_stats["auroc"]
        assert 0.0 <= auc <= 1.0
        ci_lo = profile.calibration_stats["auroc_bootstrap_ci_lo"]
        ci_hi = profile.calibration_stats["auroc_bootstrap_ci_hi"]
        assert 0.0 <= ci_lo <= ci_hi <= 1.0
        assert len(profile.calibration_stats["candidate_panel"]) == len(DEFAULT_PANEL)

        # OOB stats — the honest deployment estimate (v1.1)
        assert "oob_auroc_median" in profile.calibration_stats
        assert "oob_auroc_ci_lo" in profile.calibration_stats
        assert "oob_auroc_ci_hi" in profile.calibration_stats
        assert profile.calibration_stats["oob_n_bootstrap_used"] >= 0
        # winner_stability in [0, 1] when bootstrap produced any usable rounds
        ws = profile.calibration_stats["winner_stability"]
        if profile.calibration_stats["oob_n_bootstrap_used"] > 0:
            assert 0.0 <= ws <= 1.0
        # winner_counts: dict, non-negative integer counts
        assert isinstance(profile.calibration_stats["winner_counts"], dict)
        for label, count in profile.calibration_stats["winner_counts"].items():
            assert isinstance(label, str)
            assert isinstance(count, int) and count >= 0

        # Provenance must be populated for ALL score-critical artifacts (v1.1)
        assert profile.provenance["calibration_seed"] == 42
        assert len(profile.provenance["pipeline_module_hash_sha256"]) == 64
        assert len(profile.provenance["io_plugins_module_hash_sha256"]) == 64
        assert len(profile.provenance["model_adapters_module_hash_sha256"]) == 64
        assert len(profile.provenance["calibrator_module_hash_sha256"]) == 64
        # model_snapshot_sha may be None (uncached) but if present must be 40-char hex
        snap = profile.provenance["model_snapshot_sha"]
        if snap is not None:
            assert len(snap) == 40
            assert all(c in "0123456789abcdef" for c in snap)

    @pytest.mark.slow
    def test_calibrate_with_state_direct_two_calls_reuse_model(
        self, puzzle_calibration_jsonl, gemma_3_1b_slug
    ):
        """Regression test for the 2026-05-13 refactor — exercises the
        two-step path (load_calibration_state + calibrate_with_state) that
        scripts/anli_full_sweep.py depends on. Catches:
          * the seed-NameError bug Codex found (would fire on first bootstrap)
          * any state-leak between back-to-back calibrate_with_state calls
        """
        state = load_calibration_state(gemma_3_1b_slug, seed=42)
        # Two back-to-back calls on the same dataset must produce identical
        # detector + calibration_stats (the slug + data identity hold).
        p1 = calibrate_with_state(
            state, str(puzzle_calibration_jsonl),
            task_label="reg_test_round_a",
            n_bootstrap=200, max_new_tokens=8,
        )
        p2 = calibrate_with_state(
            state, str(puzzle_calibration_jsonl),
            task_label="reg_test_round_b",
            n_bootstrap=200, max_new_tokens=8,
        )
        # Detector identity must match — same model + same data + same seed
        # (in state) → identical winner cell, sign, and locked-sign AUROC.
        assert p1.detector == p2.detector
        assert p1.calibration_stats["auroc"] == p2.calibration_stats["auroc"]
        # Task label differs (the kwarg) but the rest of `task` should match.
        assert p1.task["data_hash_sha256"] == p2.task["data_hash_sha256"]
        assert p1.task["n_calibration"] == p2.task["n_calibration"]
        # Provenance: pipeline + module hashes must be identical (same code);
        # only calibrated_at_iso may differ.
        for k in [
            "pipeline_module_hash_sha256",
            "io_plugins_module_hash_sha256",
            "model_adapters_module_hash_sha256",
            "calibrator_module_hash_sha256",
            "model_snapshot_sha",
            "calibration_seed",
        ]:
            assert p1.provenance[k] == p2.provenance[k]

    @pytest.mark.slow
    def test_calibrate_determinism_same_seed_same_profile(
        self, puzzle_calibration_jsonl, gemma_3_1b_slug
    ):
        """Two calibrate() runs with identical input must produce identical
        metric/sign/AUROC/CI (provenance.calibrated_at_iso allowed to differ)."""
        p1 = calibrate(
            model_slug=gemma_3_1b_slug,
            calibration_jsonl_path=str(puzzle_calibration_jsonl),
            n_bootstrap=200, max_new_tokens=8, seed=42,
        )
        p2 = calibrate(
            model_slug=gemma_3_1b_slug,
            calibration_jsonl_path=str(puzzle_calibration_jsonl),
            n_bootstrap=200, max_new_tokens=8, seed=42,
        )
        assert p1.detector == p2.detector
        assert p1.calibration_stats == p2.calibration_stats
        assert p1.task == p2.task
        assert p1.model == p2.model
        # Provenance: hashes match; only the timestamp can differ
        assert (
            p1.provenance["pipeline_module_hash_sha256"]
            == p2.provenance["pipeline_module_hash_sha256"]
        )
        assert (
            p1.provenance["calibrator_module_hash_sha256"]
            == p2.provenance["calibrator_module_hash_sha256"]
        )
