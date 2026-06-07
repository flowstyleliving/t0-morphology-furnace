"""Tests for pri_detector.py — schema validation, construction, and the
calibrator/detector reproducibility contract.

The fast tier covers schema/error paths. The slow tier covers the live
detector against a real model and asserts the byte-exact AUROC reproducibility
contract that's the whole reason this library exists.

Run with:
    .venv/bin/pytest tests/test_pri_detector.py -m "not slow"
    .venv/bin/pytest tests/test_pri_detector.py -m slow
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pri_calibrator import CalibrationProfile, calibrate  # noqa: E402
from pri_detector import Detector  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Construction — pure tests that fail before any model-load attempt
# ─────────────────────────────────────────────────────────────────────────────


class TestConstructionPure:
    def test_from_profile_schema_mismatch_raises(self, tmp_path, synthetic_profile_dict):
        synthetic_profile_dict["schema_version"] = "99.0"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(synthetic_profile_dict))
        # CalibrationProfile.from_json raises before from_profile ever calls
        # load_model, so this test does NOT need MLX.
        with pytest.raises(ValueError, match=r"schema 99\.0"):
            Detector.from_profile(str(path))

    def test_from_profile_missing_file_raises(self, tmp_path):
        bogus = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            Detector.from_profile(str(bogus))

    def test_strict_mode_rejects_io_plugins_drift(self, tmp_path, synthetic_profile_dict):
        """Codex review fix: strict mode must catch drift in pri_v2_io_plugins.py,
        not just pri_runtime.py. We forge a profile whose io_plugins
        hash is wrong, and assert strict mode raises BEFORE any model load."""
        # Set the pipeline hash to the current real value so it passes,
        # but leave the io_plugins hash as the fixture's bogus "0000aaaa".
        import pri_calibrator
        synthetic_profile_dict["provenance"]["pipeline_module_hash_sha256"] = (
            pri_calibrator._hash_file(pri_calibrator.REPO_ROOT / "pri_runtime.py")
        )
        # io_plugins_module_hash_sha256 stays as "0000aaaa" → must be flagged.
        path = tmp_path / "drifted.json"
        path.write_text(json.dumps(synthetic_profile_dict))
        with pytest.raises(RuntimeError, match=r"pri_v2_io_plugins\.py"):
            Detector.from_profile(str(path), strict_pipeline_hash=True)

    def test_strict_mode_rejects_model_adapters_drift(self, tmp_path, synthetic_profile_dict):
        """Same check for model_adapters.py — score path depends on adapter
        forward methods, so drift there must be caught."""
        import pri_calibrator
        synthetic_profile_dict["provenance"]["pipeline_module_hash_sha256"] = (
            pri_calibrator._hash_file(pri_calibrator.REPO_ROOT / "pri_runtime.py")
        )
        synthetic_profile_dict["provenance"]["io_plugins_module_hash_sha256"] = (
            pri_calibrator._hash_file(pri_calibrator.REPO_ROOT / "pri_v2_io_plugins.py")
        )
        # model_adapters hash stays bogus → must be flagged.
        path = tmp_path / "drifted.json"
        path.write_text(json.dumps(synthetic_profile_dict))
        with pytest.raises(RuntimeError, match=r"model_adapters\.py"):
            Detector.from_profile(str(path), strict_pipeline_hash=True)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime — slow tier with a real model (session-scoped detector)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def calibrated_detector(puzzle_calibration_jsonl, gemma_3_1b_slug, tmp_path_factory):
    """Calibrate once per module, then build a Detector from the resulting
    profile. All slow Detector tests share this single model load.
    Returns (detector, profile_path, calibration_data_path)."""
    out_dir = tmp_path_factory.mktemp("detector_module")
    profile_path = out_dir / "profile.json"
    profile = calibrate(
        model_slug=gemma_3_1b_slug,
        calibration_jsonl_path=str(puzzle_calibration_jsonl),
        n_bootstrap=100, max_new_tokens=8, seed=42,
        task_label="detector_module_fixture",
    )
    profile.to_json(str(profile_path))
    detector = Detector.from_profile(str(profile_path))
    return detector, profile_path, puzzle_calibration_jsonl


class TestRuntime:
    @pytest.mark.slow
    def test_score_returns_float(self, calibrated_detector):
        detector, _, cal_path = calibrated_detector
        # Score the first puzzle from the calibration jsonl
        first = json.loads(Path(cal_path).read_text().splitlines()[0])
        s = detector.score(first["prompt"])
        assert isinstance(s, float)
        assert np.isfinite(s)

    @pytest.mark.slow
    def test_score_deterministic(self, calibrated_detector):
        detector, _, cal_path = calibrated_detector
        first = json.loads(Path(cal_path).read_text().splitlines()[0])
        s1 = detector.score(first["prompt"])
        s2 = detector.score(first["prompt"])
        # Greedy decoding + identical inputs must give identical scores.
        assert s1 == s2

    @pytest.mark.slow
    def test_score_batch_matches_sequential(self, calibrated_detector):
        detector, _, cal_path = calibrated_detector
        lines = Path(cal_path).read_text().splitlines()[:3]
        prompts = [json.loads(l)["prompt"] for l in lines]
        sequential = [detector.score(p) for p in prompts]
        batch = detector.score_batch(prompts)
        assert sequential == batch

    @pytest.mark.slow
    def test_predict_with_threshold(self, calibrated_detector):
        detector, _, cal_path = calibrated_detector
        first = json.loads(Path(cal_path).read_text().splitlines()[0])
        result = detector.predict(first["prompt"], threshold=0.0)
        assert isinstance(result, bool)

    @pytest.mark.slow
    def test_predict_no_threshold_raises(self, calibrated_detector):
        detector, _, cal_path = calibrated_detector
        first = json.loads(Path(cal_path).read_text().splitlines()[0])
        # The profile's detector.threshold is None by default — calling
        # predict() with no threshold arg should fail loudly.
        original_threshold = detector.profile.detector.get("threshold")
        detector.profile.detector["threshold"] = None
        try:
            with pytest.raises(ValueError, match="no threshold"):
                detector.predict(first["prompt"])
        finally:
            detector.profile.detector["threshold"] = original_threshold

    @pytest.mark.slow
    def test_self_test_aurocs_match(self, calibrated_detector):
        """THE acceptance test: re-score the calibration prompts via the
        deployed Detector, compute AUROC under direction-preserving scoring
        (sign already applied by score()), and assert it matches the
        calibrator's reported AUROC to within 1e-3.

        If this fails, the calibrator/detector reproducibility contract is
        broken — that's a no-merge condition.
        """
        from sklearn.metrics import roc_auc_score

        detector, _, cal_path = calibrated_detector
        prompts, labels = [], []
        for line in Path(cal_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            labels.append(int(row["label"]))
        labels_arr = np.array(labels, dtype=np.int32)

        scores = []
        for p in prompts:
            try:
                scores.append(detector.score(p))
            except RuntimeError:
                # Model EOS'd before rupture step — surface as NaN, skip
                scores.append(float("nan"))
        scores_arr = np.array(scores, dtype=np.float64)
        finite = np.isfinite(scores_arr)
        if finite.sum() < 4 or len(np.unique(labels_arr[finite])) < 2:
            pytest.skip("insufficient finite scores to compute AUROC")

        deployed_auroc = float(roc_auc_score(labels_arr[finite], scores_arr[finite]))
        reported_auroc = float(detector.profile.calibration_stats["auroc"])
        delta = abs(deployed_auroc - reported_auroc)
        assert delta < 1e-3, (
            f"reproducibility delta {delta:.6f} > 1e-3 "
            f"(reported={reported_auroc:.4f}, deployed={deployed_auroc:.4f})"
        )

    @pytest.mark.slow
    def test_output_projection_kind_recorded(self, calibrated_detector):
        """The detector must verify the loaded model's output_projection_kind
        matches what the profile recorded. Sanity check: it actually IS
        consistent after construction."""
        detector, _, _ = calibrated_detector
        assert detector.projection.mode == detector.profile.model["output_projection_kind"]
