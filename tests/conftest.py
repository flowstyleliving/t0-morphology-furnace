"""Pytest fixtures for the PRI calibrator + detector test suite.

Fast-tier fixtures (no model load): tmp_calibration_jsonl, synthetic_profile_dict,
synthetic_scores_and_labels.

Slow-tier fixtures (need a real model): puzzle_calibration_jsonl,
gemma_3_1b_slug. Used only by tests marked `@pytest.mark.slow`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fast-tier fixtures (no model load)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_calibration_jsonl(tmp_path) -> Callable[[List[Tuple[str, int]]], Path]:
    """Factory: build a calibration.jsonl from a list of (prompt, label) pairs.
    Returns the path. Caller decides how many lines."""
    def _make(rows: List[Tuple[str, int]], filename: str = "calibration.jsonl") -> Path:
        path = tmp_path / filename
        with path.open("w") as f:
            for prompt, label in rows:
                f.write(json.dumps({"prompt": prompt, "label": int(label)}) + "\n")
        return path
    return _make


@pytest.fixture
def synthetic_profile_dict() -> Dict[str, Any]:
    """Minimal valid CalibrationProfile-shaped dict (schema v1.2)."""
    return {
        "schema_version": "1.2",
        "model": {"slug": "test/fake-model", "output_projection_kind": "lm_head"},
        "task": {
            "label": "unit_test",
            "n_calibration": 10,
            "n_pos": 5,
            "n_neg": 5,
            "data_hash_sha256": "abc",
        },
        "detector": {
            "gen_step": 1,
            "layer": "final",
            "alpha": 1.0,
            "metric": {
                "family": "scalar",
                "label": "d_F_full",
                "column_name": "d_F_full",
                "derivation": None,
            },
            "sign": -1,
            "threshold": None,
        },
        "calibration_stats": {
            "auroc": 0.85,
            "auroc_bootstrap_ci_lo": 0.72,
            "auroc_bootstrap_ci_hi": 0.95,
            "oob_auroc_median": 0.78,
            "oob_auroc_ci_lo": 0.62,
            "oob_auroc_ci_hi": 0.90,
            "oob_n_bootstrap_used": 100,
            "winner_stability": 0.85,
            "winner_counts": {"d_F_full @ step 1": 85, "kl_discharged @ step 1": 15},
            "candidate_panel": [],
        },
        "provenance": {
            "calibration_seed": 42,
            "n_bootstrap": 100,
            "pipeline_module_hash_sha256": "deadbeef",
            "io_plugins_module_hash_sha256": "0000aaaa",
            "model_adapters_module_hash_sha256": "0000bbbb",
            "calibrator_module_hash_sha256": "cafef00d",
            "model_snapshot_sha": None,
            "calibrated_at_iso": "2026-05-13T00:00:00Z",
            "max_new_tokens": 8,
        },
        "warnings": [],
    }


@pytest.fixture
def synthetic_scores_and_labels() -> Dict[str, np.ndarray]:
    """Known-AUROC labeled data for _score_candidate / _bootstrap_auroc tests."""
    rng = np.random.RandomState(42)
    return {
        # Perfect separation, +sign: higher score → label 1
        "perfect_pos": (
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
        ),
        # Perfect separation, -sign: higher score → label 0
        "perfect_neg": (
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64),
            np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32),
        ),
        # Random
        "random": (
            rng.randn(20).astype(np.float64),
            rng.randint(0, 2, size=20).astype(np.int32),
        ),
        # Too few samples
        "tiny": (
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 1, 1], dtype=np.int32),
        ),
        # Single class
        "single_class": (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([1, 1, 1, 1, 1], dtype=np.int32),
        ),
        # NaN in some scores
        "with_nans": (
            np.array([1, np.nan, 3, np.nan, 5, 6, np.nan, 8, 9, 10], dtype=np.float64),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slow-tier fixtures (need MLX + a model)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gemma_3_1b_slug() -> str:
    """Smallest cached model with passing smoke; used for slow-tier integration.
    Module-scoped so module-scoped fixtures (calibrated_detector) can depend on it."""
    return "mlx-community/gemma-3-1b-it-4bit"


@pytest.fixture(scope="module")
def puzzle_calibration_jsonl(tmp_path_factory) -> Path:
    """Generate 20 synthetic logic puzzles via the pipeline's PuzzleGenerator
    and write as a calibration jsonl. Module-scoped (uses tmp_path_factory,
    which is session-scoped) so module-scoped consumers like
    `calibrated_detector` can depend on it without a ScopeMismatch."""
    import pri_v2_mlx_pipeline as pipeline

    gen = pipeline.PuzzleGenerator(seed=42)
    df = gen.generate_dataset(n_per_cell=5, chain_lengths=[2, 5])  # 5 × 2 cells × 2 contradictions = 20
    out_dir = tmp_path_factory.mktemp("puzzle_calibration")
    path = out_dir / "puzzles.jsonl"
    with path.open("w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "prompt": row["prompt"],
                "label": int(bool(row["contradiction"])),
            }) + "\n")
    return path
