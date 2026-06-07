"""PRI calibrator — produce a deployable (model, task) detector profile from
a small labeled set.

Why this exists
---------------
The 2026-05-12 Codex adversarial review + the ANLI cross-task pilot showed
PRI rupture detection cannot be deployed label-free: both the *which metric*
and the *which sign* questions are heterogeneous across models (synthetic
N=11: 8 +/3 −) and across tasks for the same model (Mistral-Nemo: synthetic
+, ANLI R2 −). The rupture is real — every (model, task) we measured had
AUROC ≈ 0.70–1.0 on some cell at gen_step=1 — but each (model, task) pair
needs a small labeled calibration set to *pick the cell* and *lock the sign*.

This module is the calibration harness. Usage:

    python pri_calibrator.py \\
        --model mlx-community/Mistral-Nemo-Instruct-2407-4bit \\
        --data calibration.jsonl \\
        --out my_model_my_task.profile.json

It loads the model once, runs the model on each labeled sample, computes a
short, curated panel of candidate rupture metrics at fixed gen_steps,
selects the best (metric, sign) by direction-preserving AUROC on the
calibration set, bootstraps a confidence interval, and persists a versioned
`CalibrationProfile` JSON that downstream consumers (a future `pri_detector.py`)
can load to score new prompts.

Schema is v1.0 and frozen. Migrations land in `pri_profile_migrations/` once
we have a v2.0.

Input format (jsonl):
    {"prompt": "Premise: ... Hypothesis: ... Answer:", "label": 0}
    {"prompt": "Premise: ... Hypothesis: ... Answer:", "label": 1}

`label` ∈ {0, 1}: 1 = the target class (contradiction/positive), 0 = consistent/
non-contradiction. Prompts are PRE-built by the researcher; the calibrator
does NOT inject a prompt template — it only applies the model's chat-template
strategy via `pri_v2_io_plugins.get_prompt_strategy`.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pri_v2_io_plugins as io_plugins
from analyze_adaptive_step import auroc_signed


class _LazyRuntimeProxy:
    """Defer the MLX runtime import until a model/tracing path actually needs it."""

    def __init__(self) -> None:
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module("pri_v2_mlx_pipeline")
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


# Reuse existing runtime primitives without forcing MLX import on pure-helper paths.
pipeline = _LazyRuntimeProxy()


SCHEMA_VERSION = "1.2"
# v1.0 → v1.1 (2026-05-13): Codex adversarial review fixes.
#   * calibration_stats gains oob_auroc_median + oob_auroc_ci_{lo,hi} +
#     oob_n_bootstrap_used + winner_stability + winner_counts. The in-sample
#     auroc/CI are still recorded (legacy semantics, useful for inspection),
#     but the OOB stats are the honest "deployment-ready" estimate.
#   * provenance gains io_plugins_module_hash_sha256 +
#     model_adapters_module_hash_sha256 + model_snapshot_sha. Detector's
#     strict mode validates ALL hashes, not just the pipeline file.
# v1.1 → v1.2 (2026-05-14): support derived winners (composites + residualized).
#   * profile.detector.metric optionally carries a `derivation` payload that
#     tells the detector how to compute the metric at score time:
#       - composite: {"kind": "composite", "formula": "<label>"}
#       - residualized: {"kind": "residualized", "base_column": "...",
#                        "regress_against": "d_F_full", "b0": float, "b1": float}
#     Direct cells leave `derivation: None`.
#   * Detector v1.2 reads `derivation` and reproduces both forms.
#   * Earlier (v1.1) profiles are rejected — re-calibrate. Existing v1.1
#     calibrator wrote `column_name = "composite::..."` for composite winners
#     which detector couldn't resolve; v1.2 makes that contract explicit.


# ─────────────────────────────────────────────────────────────────────────────
# Metric panel — fixed-step per candidate.
# ─────────────────────────────────────────────────────────────────────────────

# `(parquet_gen_step, family, rank_label)` — these match the column conventions
# the existing pipeline + step-sweep analyses use. The calibrator only
# evaluates these 8 cells (multiple-testing burden at n=10-50 is tractable).
PanelCell = Tuple[int, str, str]


# ─── Attention cell family (added 2026-05-15; t0-candidate #5) ───────────────
# First non-`compute_step` cell family. Cells map to metrics computed from
# captured attention weights across three target decoder blocks (final, mid,
# last_minus_1) at one or more gen_steps. Capture is opt-in: the relevant
# ATTENTION_PANEL is appended to DEFAULT_PANEL only when --attention is
# passed (or callers pass an extended panel). When the panel contains any
# Attention cell, the per-sample trace is wrapped in
# scripts.diagnose_inter_head_disagreement's observational attention_capture
# context manager; the wrapper handles each model family's native attention
# layout (Phi qkv_proj fusion, Qwen3/Gemma q_norm/k_norm). Wrapper module
# hash is recorded in provenance so detector strict mode can verify capture
# invariance.
ATTENTION_FAMILY = "Attention"
ATTENTION_LAYERS: Tuple[str, ...] = ("final", "mid", "last_minus_1")
# Default 4 metrics — derived from attention weights only.
ATTENTION_METRICS: Tuple[str, ...] = ("js", "js_kv_groups", "js_no_bos", "bos_mass")
# 3 SinkProbe-style metrics — require value-vector norms in addition to
# weights. Enabled in panels built with `with_v_norms=True` or the
# `--attention-with-v-norms` CLI flag.
ATTENTION_METRICS_V_NORMS: Tuple[str, ...] = (
    "v_norm_bos",            # ‖V_0‖ averaged over KV heads
    "v_norm_max",            # max_i ‖V_i‖ averaged over KV heads
    "v_norm_lastq_weighted", # Σ_i A^h_{q=-1,i} · ‖V_i^h‖ averaged over Q heads
)
ATTENTION_STEPS_DEFAULT: Tuple[int, ...] = (1,)              # commit step only (v3 sealed plane)
ATTENTION_STEPS_MULTISTEP: Tuple[int, ...] = (1, 2, 3, 4)    # multistep: commit + 3 post-commit
# t=0 instrument: prefill last-position attention (captures[layer][0]).
# Semantically consistent across all models — always the last prompt token's
# attention over the full prefix, computed in the same forward pass as
# prefix_probs[-1]. Gen_step=1 captures differ per model (query is the first
# generated token: \n for Mistral, YES/NO for Qwen, etc.).
ATTENTION_STEPS_T0: Tuple[int, ...] = (0,)

def make_attention_panel(
    steps: Tuple[int, ...] = ATTENTION_STEPS_DEFAULT,
    layers: Tuple[str, ...] = ATTENTION_LAYERS,
    metrics: Tuple[str, ...] = ATTENTION_METRICS,
    *,
    with_v_norms: bool = False,
) -> List[PanelCell]:
    """Build an Attention panel as `len(steps) × len(layers) × len(metrics)`
    cells. Multi-step expansion (`steps=(1,2,3,4)`) is the multistep variant.
    When `with_v_norms=True`, also appends `len(steps) × len(layers) ×
    len(ATTENTION_METRICS_V_NORMS)` cells covering the SinkProbe-style
    value-norm metrics; calibration will switch to attention_capture_with_values.
    """
    all_metrics: Tuple[str, ...] = metrics
    if with_v_norms:
        all_metrics = tuple(metrics) + tuple(ATTENTION_METRICS_V_NORMS)
    return [
        (step, ATTENTION_FAMILY, f"{layer}_{metric}")
        for step in steps
        for layer in layers
        for metric in all_metrics
    ]

DEFAULT_PANEL: List[PanelCell] = [
    # ─── Direct (raw) cells from compute_step ──────────────────────────
    (1, "scalar", "d_F_full"),         # most task-stable across N=11
    (1, "scalar", "kl_discharged"),    # closed-form KL-grounded scalar
    (1, "Fisher", "r=1"),              # `pri_v3_null_bare` — decomposition control
    (3, "Fisher", "r=2"),              # Qwen 2.5 oracle (synthetic)
    (1, "Centered", "r=2"),            # Mistral-Nemo oracle (synthetic)
    (1, "Centered", "r=4"),            # alternate centered low-rank
    (4, "Raw", "r=2"),                 # cross-Llama universal (3B + 8B)
    (3, "Raw", "r=21"),                # cross-Qwen universal
    # ─── Residualized (E18 sealed primary form) ────────────────────────
    # `null_ratio_resid = null_ratio − predicted(null_ratio | d_F_full)`
    # via linear regression on d_F_full alone. Per pri-v3-plan.md §
    # Magnitude-independence test (sealed at step 1): this is the E18
    # acceptance variable — strips the magnitude confound the raw
    # null_ratio carries.
    #
    # CONFINED TO STEP 1 (Codex review 2026-05-14): residualization
    # requires d_F_full at the SAME step. Only step 1 has d_F_full in this
    # panel, so step-3/4 resid cells would be cross-step-leaked. The non-
    # step-1 raw cells stay in the panel for their own merit; we don't
    # provide a residualized version for them unless / until d_F_full @
    # those steps is added explicitly.
    (1, "Fisher_resid", "r=1"),
    (1, "Centered_resid", "r=2"),
    (1, "Raw_resid", "r=1"),
    # ─── Composites (E18 additive + E19 multiplicative) ────────────────
    # `pri_v3_null_ratio = S_t + α · null_ratio_final` (additive, E18)
    # `pri_v3_null_gated = d_F · null_ratio` (multiplicative, E19)
    (1, "Composite", "additive_S_fisher_r=1"),     # surprise + null_ratio_post_rank1
    (1, "Composite", "gated_dF_fisher_r=1"),       # d_F_full * null_ratio_post_rank1
    (1, "Composite", "additive_S_centered_r=2"),
    (1, "Composite", "gated_dF_centered_r=2"),
]


# 12-cell opt-in attention extension (gen_step=1 only). Appended to
# DEFAULT_PANEL via the `--attention` CLI flag or by passing
# `panel=DEFAULT_PANEL + ATTENTION_PANEL`.
ATTENTION_PANEL: List[PanelCell] = make_attention_panel(ATTENTION_STEPS_DEFAULT)

# 48-cell multistep variant (gen_step ∈ {1,2,3,4}). Opt-in via
# `--attention-multistep`. Multi-testing burden is 4× larger so the OOB
# warnings will fire more aggressively at small n; that's correct
# behavior, not noise.
ATTENTION_PANEL_MULTISTEP: List[PanelCell] = make_attention_panel(ATTENTION_STEPS_MULTISTEP)

# 21-cell variant adding 3 SinkProbe-style value-norm metrics at the
# commit step (gen_step=1). Opt-in via `--attention-with-v-norms`.
# Requires the value-norm capture path (attention_capture_with_values);
# slightly more wall per sample but typically <5% overhead.
ATTENTION_PANEL_WITH_V_NORMS: List[PanelCell] = make_attention_panel(
    ATTENTION_STEPS_DEFAULT, with_v_norms=True,
)

# t=0 variants: same 12-cell and 21-cell panels but at step=0 (prefill
# last-position). Use with --t0-commit to measure at the honest commit locus.
ATTENTION_PANEL_T0: List[PanelCell] = make_attention_panel(ATTENTION_STEPS_T0)
ATTENTION_PANEL_T0_WITH_V_NORMS: List[PanelCell] = make_attention_panel(
    ATTENTION_STEPS_T0, with_v_norms=True,
)

# 5-cell residual-stream panel at the t=0 locus (prefix-last-position hidden state).
# Analogous to DEFAULT_PANEL scalar/Fisher/Raw cells but measured before any
# generation — same locus as ATTENTION_PANEL_T0. Requires the step=0 branch in
# _compute_panel_scores_for_sample. Use with --t0-residual.
T0_RESIDUAL_PANEL: List[PanelCell] = [
    (0, "scalar", "d_F_full"),
    (0, "scalar", "kl_discharged"),
    (0, "Fisher", "r=1"),
    (0, "Fisher", "r=2"),
    (0, "Raw", "r=21"),
]


# Families that are DERIVED (not present as a direct compute_step column).
# Residualized cells use the base family's column + a regression against
# d_F_full. Composite cells combine compute_step columns (S_t / d_F_full
# with null_ratio). Both are computed in `_compute_panel_scores_for_sample`
# and (for residuals) post-processed across the full sample set.
DERIVED_RESID_FAMILIES = {"Fisher_resid", "Raw_resid", "Centered_resid"}
DERIVED_COMPOSITE_FAMILY = "Composite"


def _resid_base_family(family: str) -> str:
    """Map a `*_resid` family to the underlying compute_step family."""
    return family.removesuffix("_resid")


def _column_name(cell: PanelCell) -> str:
    """Map a PanelCell to the `compute_step` output dict key.
    For derived cells (residualized / composite), this returns the column
    of the *base* signal — the residualization or composition logic is
    applied by `_compute_panel_scores_for_sample` and the post-loop
    residualization pass in `calibrate_with_state`.

    Attention cells return a synthetic `attention::<layer>_<metric>`
    namespace; they do not appear in `compute_step` output and are scored
    from captured attention weights instead.
    """
    step, fam, label = cell
    if fam == "scalar":
        return label  # "d_F_full" / "kl_discharged"
    if fam == DERIVED_COMPOSITE_FAMILY:
        # Composite labels encode their own base column reference. Return
        # the label itself; the dispatcher in _compute_panel_scores_for_sample
        # parses it and composes from primitive columns.
        return f"composite::{label}"
    if fam == ATTENTION_FAMILY:
        return f"attention::{label}"
    base_fam = _resid_base_family(fam) if fam in DERIVED_RESID_FAMILIES else fam
    rank = int(label.split("=")[1])
    if base_fam == "Fisher":
        return f"null_ratio_post_rank{rank}"
    if base_fam == "Raw":
        return f"null_ratio_raw_post_rank{rank}"
    if base_fam == "Centered":
        return f"null_ratio_centered_post_rank{rank}"
    raise ValueError(f"unknown family: {fam}")


def _cell_label(cell: PanelCell) -> str:
    """Human-readable cell name for reports + provenance."""
    step, fam, label = cell
    if fam == "scalar":
        return f"{label} @ step {step}"
    if fam == DERIVED_COMPOSITE_FAMILY:
        return f"composite[{label}] @ step {step}"
    if fam == ATTENTION_FAMILY:
        return f"attention[{label}] @ step {step}"
    return f"{fam} {label} @ step {step}"


def _is_attention_cell(cell: PanelCell) -> bool:
    return cell[1] == ATTENTION_FAMILY


def _requires_attention_capture(panel: List[PanelCell]) -> bool:
    return any(_is_attention_cell(c) for c in panel)


def _split_attention_label(label: str) -> Optional[Tuple[str, str]]:
    """Split an Attention-cell label `<layer>_<metric>` into (layer, metric).

    Metrics with underscores (`js_kv_groups`, `js_no_bos`, `bos_mass`,
    `v_norm_bos`, `v_norm_max`, `v_norm_lastq_weighted`) require careful
    parsing — we anchor on the ATTENTION_LAYERS prefix list, not on
    the rightmost-underscore.
    """
    known_metrics = set(ATTENTION_METRICS) | set(ATTENTION_METRICS_V_NORMS)
    for layer in ATTENTION_LAYERS:
        prefix = f"{layer}_"
        if label.startswith(prefix):
            metric = label[len(prefix):]
            if metric in known_metrics:
                return layer, metric
    return None


def _is_v_norm_metric(metric: str) -> bool:
    return metric in ATTENTION_METRICS_V_NORMS


def _requires_v_norm_capture(panel: List[PanelCell]) -> bool:
    """True iff any panel cell uses a V-norm metric (and therefore the
    calibrator needs to use `attention_capture_with_values` instead of the
    weights-only `attention_capture`).
    """
    for cell in panel:
        if cell[1] != ATTENTION_FAMILY:
            continue
        parsed = _split_attention_label(cell[2])
        if parsed is None:
            continue
        _layer, metric = parsed
        if _is_v_norm_metric(metric):
            return True
    return False


def _compute_attention_score(
    cell: PanelCell,
    captures: Dict[str, List[Any]],
    n_kv_heads_by_layer: Dict[str, int],
    *,
    v_norm_captures: Optional[Dict[str, List[Any]]] = None,
) -> Optional[float]:
    """Compute one Attention-family cell's score from captured weights.

    `captures[tag]` is a list of attention-weight arrays (one per forward
    call); index 0 is the prefix forward, index k≥1 is gen_step=k's
    last-query row. The diagnostic's wrapper already slices to (H, T_kv).

    For a cell `(step, "Attention", "<layer>_<metric>")` we read
    `captures[layer][step]`. If the model EOS'd before reaching gen_step=k
    the captures list is shorter and we return None — that's how short
    generations are tolerated (the calibration sample is then NaN for
    this cell and contributes nothing to that cell's AUROC).

    V-norm metrics additionally require `v_norm_captures[tag][step]`
    (shape (n_kv_heads, T)); if `v_norm_captures` is None, those cells
    return None.

    Returns None if the cell can't be evaluated (wrong family, step < 0,
    label unparseable, captures missing/too-short, or n_kv_heads unknown
    for js_kv_groups, or v_norm_captures missing for a v-norm metric).
    Deferred imports break the import cycle with the diagnostic module.
    """
    from diagnose_inter_head_disagreement import (
        _js_radius, _js_radius_kv_groups, _js_radius_no_bos, _mean_bos_mass,
        _mean_v_norm_bos, _mean_v_norm_max, _lastq_weighted_v_norm,
    )

    step, fam, label = cell
    if fam != ATTENTION_FAMILY or step < 0:
        return None
    parsed = _split_attention_label(label)
    if parsed is None:
        return None
    layer, metric = parsed
    caps = captures.get(layer)
    if not caps or len(caps) <= step:
        return None
    w = caps[step]
    try:
        if metric == "js":
            v = _js_radius(w)
        elif metric == "js_no_bos":
            v = _js_radius_no_bos(w)
        elif metric == "bos_mass":
            v = _mean_bos_mass(w)
        elif metric == "js_kv_groups":
            n_kv = n_kv_heads_by_layer.get(layer)
            if n_kv is None:
                return None
            v = _js_radius_kv_groups(w, n_kv)
        elif metric in ATTENTION_METRICS_V_NORMS:
            if v_norm_captures is None:
                return None
            v_caps = v_norm_captures.get(layer)
            if not v_caps or len(v_caps) <= step:
                return None
            v_norms = v_caps[step]
            if metric == "v_norm_bos":
                v = _mean_v_norm_bos(v_norms)
            elif metric == "v_norm_max":
                v = _mean_v_norm_max(v_norms)
            elif metric == "v_norm_lastq_weighted":
                v = _lastq_weighted_v_norm(w, v_norms)
            else:
                return None
        else:
            return None
    except Exception as _exc:
        import warnings
        warnings.warn(
            f"_compute_attention_score: metric '{metric}' raised "
            f"{type(_exc).__name__}: {_exc}",
            stacklevel=2,
        )
        return None
    if v is None or not np.isfinite(v):
        return None
    return float(v)


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationProfile (frozen schema v1.0)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationProfile:
    """Versioned profile that fully specifies a deployable PRI detector.

    Fields are deliberately flat-friendly so the JSON form is easy to inspect
    in a text editor. `schema_version` MUST be `"1.0"` until a breaking
    change lands (then bump + add a migration under pri_profile_migrations/).
    """

    schema_version: str
    model: Dict[str, Any]                 # {slug, output_projection_kind}
    task: Dict[str, Any]                  # {label, n_calibration, n_pos, n_neg, data_hash}
    detector: Dict[str, Any]              # {gen_step, layer, alpha, metric, sign, threshold}
    calibration_stats: Dict[str, Any]     # {auroc, ci_lo, ci_hi, candidate_panel, n_evaluated_per_cell}
    provenance: Dict[str, Any]            # {calibration_seed, pipeline_module_hash, calibrated_at_iso, n_bootstrap}
    warnings: List[str] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str) -> "CalibrationProfile":
        d = json.loads(Path(path).read_text())
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"profile schema {d.get('schema_version')} != supported {SCHEMA_VERSION}; "
                f"see pri_profile_migrations/ once v2.0 lands"
            )
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion
# ─────────────────────────────────────────────────────────────────────────────


def _load_calibration_jsonl(path: str) -> Tuple[List[str], np.ndarray, str]:
    """Read calibration.jsonl → (prompts, labels, sha256_hash).
    `data_hash` is the sha256 of `<label>\\t<prompt>\\n` rows in input order so
    re-running on the same input produces an identical hash.
    """
    prompts: List[str] = []
    labels: List[int] = []
    h = hashlib.sha256()
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = str(row["prompt"])
            y = int(row["label"])
            if y not in (0, 1):
                raise ValueError(f"label must be 0 or 1, got {y!r}")
            prompts.append(p)
            labels.append(y)
            h.update(f"{y}\t{p}\n".encode("utf-8"))
    if not prompts:
        raise RuntimeError(f"no calibration samples loaded from {path}")
    return prompts, np.array(labels, dtype=np.int32), h.hexdigest()


def _hash_file(path: Path) -> str:
    """sha256 of a file's contents, used for code-version provenance."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Trace + metric computation for one calibration sample
# ─────────────────────────────────────────────────────────────────────────────


def _compute_panel_scores_for_sample(
    pri_computer: pipeline.PRIComputer,
    trace: Dict[str, Any],
    layer_name: str,
    panel: List[PanelCell],
    alpha: float = 1.0,
    v3_rank_values: Tuple[int, ...] = (1, 2, 4, 21),
    *,
    attention_captures: Optional[Dict[str, List[Any]]] = None,
    attention_n_kv_heads: Optional[Dict[str, int]] = None,
    attention_v_norm_captures: Optional[Dict[str, List[Any]]] = None,
) -> Dict[PanelCell, Optional[float]]:
    """For one calibration sample's trace, compute every panel cell's value.
    Returns dict mapping panel cell → score (or None if the model EOS'd
    before the panel step). The same `compute_step` invocation at a given
    step emits ALL column families, so we cache by step rather than
    re-computing.

    For Attention-family cells, `attention_captures` is the dict produced by
    the diagnostic module's `attention_capture` context manager wrapping the
    trace (keyed by layer tag), and `attention_n_kv_heads` maps each layer
    tag to its decoder block's n_kv_heads (needed for the GQA-aware metric).
    Both default to None; if any panel cell is an Attention cell and these
    aren't provided, that cell returns None.
    """
    gen_hidden = trace["gen_hidden"][layer_name]
    n_gen = len(gen_hidden)
    gen_probs = trace["gen_probs"]
    gen_surprises = trace["gen_surprises"]
    last_prefix = trace["last_prefix_hidden"][layer_name]

    # Cache per-step compute_step output. parquet gen_step 1 → idx 0.
    # Skip steps only used by Attention cells (those don't need compute_step).
    steps_needed = sorted({
        step for step, fam, _ in panel if fam != ATTENTION_FAMILY
    })
    step_to_result: Dict[int, Optional[Dict[str, float]]] = {}
    for step in steps_needed:
        if step == 0:
            # t=0: last prefix-token hidden state — same locus as ATTENTION_PANEL_T0.
            prefix_seq = trace["prefix_hidden"][layer_name]
            h_t0 = last_prefix
            h_prev0 = prefix_seq[-2] if len(prefix_seq) >= 2 else last_prefix
            p_t0 = trace["prefix_probs"][-1]
            S_t0 = (float(gen_surprises[0])
                    if len(gen_surprises) > 0 and np.isfinite(gen_surprises[0])
                    else 0.0)
            step_to_result[0] = pri_computer.compute_step(
                h_t=h_t0, h_prev=h_prev0, p_t=p_t0, S_t=S_t0,
                alpha=alpha, topk_values=[32], lowrank_values=[32],
                v3_rank_values=list(v3_rank_values),
                v3_capture_raw=True, v3_capture_centered=True,
            )
            continue
        idx = step - 1
        if idx >= n_gen:
            step_to_result[step] = None
            continue
        h_t = gen_hidden[idx]
        h_prev = gen_hidden[idx - 1] if idx >= 1 else last_prefix
        p_t = gen_probs[idx]
        S_t = float(gen_surprises[idx]) if np.isfinite(gen_surprises[idx]) else 0.0
        result = pri_computer.compute_step(
            h_t=h_t,
            h_prev=h_prev,
            p_t=p_t,
            S_t=S_t,
            alpha=alpha,
            topk_values=[32],
            lowrank_values=[32],
            v3_rank_values=list(v3_rank_values),
            v3_capture_raw=True,
            v3_capture_centered=True,
        )
        step_to_result[step] = result

    out: Dict[PanelCell, Optional[float]] = {}
    for cell in panel:
        step, fam, label = cell

        # ── Attention: read from captures rather than compute_step ─────
        if fam == ATTENTION_FAMILY:
            if attention_captures is None:
                out[cell] = None
                continue
            out[cell] = _compute_attention_score(
                cell, attention_captures, attention_n_kv_heads or {},
                v_norm_captures=attention_v_norm_captures,
            )
            continue

        res = step_to_result.get(step)
        if res is None:
            out[cell] = None
            continue

        # ── Composite: assemble from primitive columns at THIS sample ──
        if fam == DERIVED_COMPOSITE_FAMILY:
            v = _compose_score(res, label)
            out[cell] = v if (v is not None and np.isfinite(v)) else None
            continue

        # ── Residualized: emit the BASE column value here; the residual
        # ── pass after all samples land will overwrite this with
        # ── (raw − predicted_from_d_F).
        if fam in DERIVED_RESID_FAMILIES:
            base_fam = _resid_base_family(fam)
            base_col = _column_name((step, base_fam, label))
            v = res.get(base_col)
            out[cell] = float(v) if (v is not None and np.isfinite(v)) else None
            continue

        # ── Direct cells (scalar / Fisher / Raw / Centered) ────────────
        col = _column_name(cell)
        v = res.get(col)
        if v is None or not np.isfinite(v):
            out[cell] = None
        else:
            out[cell] = float(v)
    return out


def _compose_score(result: Dict[str, float], label: str) -> Optional[float]:
    """Build a composite score from primitive compute_step columns.

    Label conventions (extend by appending to this dispatcher):
      additive_S_fisher_r=N    → result["surprise"] + result["null_ratio_post_rankN"]
      gated_dF_fisher_r=N      → result["d_F_full"] * result["null_ratio_post_rankN"]
      additive_S_centered_r=N  → result["surprise"] + result["null_ratio_centered_post_rankN"]
      gated_dF_centered_r=N    → result["d_F_full"] * result["null_ratio_centered_post_rankN"]
      additive_S_raw_r=N       → result["surprise"] + result["null_ratio_raw_post_rankN"]
      gated_dF_raw_r=N         → result["d_F_full"] * result["null_ratio_raw_post_rankN"]
    """
    parts = label.split("_")
    if len(parts) < 4:
        return None
    op = parts[0]                # "additive" or "gated"
    scalar_key = parts[1]        # "S" or "dF"
    base_family = parts[2]       # "fisher" / "centered" / "raw"
    rank_part = parts[3]         # "r=N"
    if not rank_part.startswith("r="):
        return None
    try:
        rank = int(rank_part[2:])
    except ValueError:
        return None

    if base_family == "fisher":
        base_col = f"null_ratio_post_rank{rank}"
    elif base_family == "centered":
        base_col = f"null_ratio_centered_post_rank{rank}"
    elif base_family == "raw":
        base_col = f"null_ratio_raw_post_rank{rank}"
    else:
        return None

    base = result.get(base_col)
    if base is None or not np.isfinite(base):
        return None

    if scalar_key == "S":
        scalar = result.get("surprise")
    elif scalar_key == "dF":
        scalar = result.get("d_F_full")
    else:
        return None
    if scalar is None or not np.isfinite(scalar):
        return None

    if op == "additive":
        return float(scalar + base)
    if op == "gated":
        return float(scalar * base)
    return None


def _trace_one_prompt(
    model: Any,
    tokenizer: Any,
    projection: pipeline.OutputProjection,
    layer_indices: Dict[str, int],
    prompt: str,
    prompt_strategy,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Apply the model's chat-template strategy then call trace_sample."""
    wrapped = prompt_strategy(prompt, tokenizer)
    return pipeline.trace_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=wrapped,
        layer_indices=layer_indices,
        output_projection=projection,
        max_new_tokens=max_new_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def _score_candidate(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, int, int]:
    """Return (auroc, sign, n_evaluated). Sign is locked from THIS data —
    that's the whole point of calibration. Drops NaN scores; if fewer than
    4 samples with both labels survive, returns (nan, 0, n_finite)."""
    finite = np.isfinite(scores)
    n_eval = int(finite.sum())
    s = scores[finite]
    y = labels[finite]
    if n_eval < 4 or len(np.unique(y)) < 2:
        return float("nan"), 0, n_eval
    auc, sign = auroc_signed(y, s)
    return float(auc), int(sign), n_eval


def _nested_bootstrap_oob_auroc(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    panel: List[PanelCell],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Nested bootstrap: at each round, resample the calibration set with
    replacement to form an in-bag set; re-run the WHOLE cell selection
    (best cell + sign-lock) on the in-bag samples; then evaluate the
    selected (cell, sign) on the out-of-bag samples. The OOB AUROC
    distribution is the honest deployment estimate — it accounts for the
    selection bias that contaminates in-sample stats.

    Also tracks how often each cell is selected across rounds; if one cell
    wins ≪ 100% of resamples, the selection is noisy at this n and the
    profile gets a `winner_unstable` warning.

    Returns a dict with the OOB summary stats. If no round produced a
    valid OOB AUROC (degenerate small-n case), returns a dict full of
    NaNs.
    """
    from sklearn.metrics import roc_auc_score

    n, n_cells = score_matrix.shape
    rng = np.random.RandomState(seed + 1)  # +1 so it doesn't clash with _bootstrap_auroc's stream
    oob_aurocs: List[float] = []
    winner_counts: Dict[int, int] = {j: 0 for j in range(n_cells)}

    for _ in range(n_bootstrap):
        in_bag = rng.randint(0, n, size=n)
        in_bag_set = set(in_bag.tolist())
        oob = np.array([i for i in range(n) if i not in in_bag_set], dtype=np.int64)
        if len(oob) < 4 or len(np.unique(labels[oob])) < 2:
            continue

        # Re-run cell selection inside this resample.
        best_j = -1
        best_distance = -1.0
        best_sign = 0
        for j in range(n_cells):
            s_in = score_matrix[in_bag, j]
            y_in = labels[in_bag]
            auc, sign, _ = _score_candidate(s_in, y_in)
            if np.isfinite(auc):
                d = abs(auc - 0.5)
                if d > best_distance:
                    best_distance = d
                    best_j = j
                    best_sign = sign
        if best_j < 0:
            continue
        winner_counts[best_j] += 1

        # Evaluate the in-bag-selected cell on OOB with the in-bag-locked sign.
        s_oob = score_matrix[oob, best_j] * best_sign
        y_oob = labels[oob]
        finite = np.isfinite(s_oob)
        if finite.sum() < 4 or len(np.unique(y_oob[finite])) < 2:
            continue
        oob_aurocs.append(float(roc_auc_score(y_oob[finite], s_oob[finite])))

    total_winners = sum(winner_counts.values())
    if total_winners > 0:
        max_count = max(winner_counts.values())
        winner_stability = max_count / total_winners
    else:
        winner_stability = float("nan")
    winner_counts_labeled = {
        _cell_label(panel[j]): c for j, c in winner_counts.items() if c > 0
    }
    if not oob_aurocs:
        return {
            "oob_auroc_median": float("nan"),
            "oob_auroc_ci_lo": float("nan"),
            "oob_auroc_ci_hi": float("nan"),
            "oob_n_bootstrap_used": 0,
            "winner_stability": float(winner_stability) if np.isfinite(winner_stability) else float("nan"),
            "winner_counts": winner_counts_labeled,
        }
    arr = np.array(oob_aurocs)
    return {
        "oob_auroc_median": float(np.median(arr)),
        "oob_auroc_ci_lo": float(np.percentile(arr, 2.5)),
        "oob_auroc_ci_hi": float(np.percentile(arr, 97.5)),
        "oob_n_bootstrap_used": int(len(oob_aurocs)),
        "winner_stability": float(winner_stability),
        "winner_counts": winner_counts_labeled,
    }


def _resolve_model_snapshot_sha(model_slug: str) -> Optional[str]:
    """Resolve the HuggingFace cache snapshot SHA for the model the calibrator
    will load. This pins the exact model artifact (not just the slug). Returns
    None if the model isn't cached locally or the path can't be parsed.

    The HF cache layout puts each downloaded snapshot under
        ~/.cache/huggingface/hub/models--{owner}--{repo}/snapshots/{commit_sha}/
    so the parent directory of any file we resolve gives us the SHA directly.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    # config.json is the most reliable sentinel — present in every HF repo.
    path = try_to_load_from_cache(repo_id=model_slug, filename="config.json")
    if not path:
        return None
    try:
        sha = Path(path).parent.name
    except Exception:
        return None
    # Sanity: HF commit SHAs are 40-char hex.
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
        return None
    return sha


def _bootstrap_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    sign: int,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    """Resample with replacement n_bootstrap times, scoring with the LOCKED
    sign (no re-flipping per round). Returns (2.5%, 97.5%) percentiles.
    With locked sign, AUROC < 0.5 is meaningful — it means the sign was
    wrong for this resample (chance noise on small calibration sets).
    """
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(scores)
    s = scores[finite] * sign
    y = labels[finite]
    n = len(s)
    if n < 4 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        ys, ss = y[idx], s[idx]
        if len(np.unique(ys)) < 2:
            continue
        aucs.append(roc_auc_score(ys, ss))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────


def _emit_warnings(
    n_calibration: int,
    n_pos: int,
    n_neg: int,
    best_auroc: float,
    ci_lo: float,
    ci_hi: float,
    panel_eval_counts: Dict[PanelCell, int],
    *,
    oob_auroc_median: Optional[float] = None,
    winner_stability: Optional[float] = None,
) -> List[str]:
    """Deployability warnings, baked into the profile so downstream consumers
    see them at load time. Don't raise — these are advisory; the researcher
    decides whether to deploy."""
    w: List[str] = []
    if n_calibration < 20:
        w.append(f"small_calibration_n (n={n_calibration}; rule of thumb: >= 20)")
    if n_pos + n_neg > 0:
        pos_rate = n_pos / (n_pos + n_neg)
        if pos_rate < 0.3 or pos_rate > 0.7:
            w.append(f"class_imbalance (pos_rate={pos_rate:.2f}; aim for [0.3, 0.7])")
    if np.isfinite(best_auroc) and best_auroc < 0.65:
        w.append(f"low_auroc (best={best_auroc:.3f}; <0.65 likely not deployable)")
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        width = ci_hi - ci_lo
        if width > 0.30:
            w.append(f"wide_ci (95%% CI width={width:.2f}; >0.30 implies n too small)")
    for cell, n_eval in panel_eval_counts.items():
        if n_eval < n_calibration * 0.8:
            w.append(
                f"insufficient_coverage_at_{_cell_label(cell)} "
                f"(n_evaluated={n_eval}/{n_calibration}; model EOS'd before this step too often)"
            )
    # OOB-flavored warnings — these reflect the honest deployment estimate,
    # not the optimistically-biased in-sample stats. Added 2026-05-13 to
    # address the Codex review's selection-bias finding.
    if oob_auroc_median is not None and np.isfinite(oob_auroc_median):
        if oob_auroc_median < 0.60:
            w.append(
                f"oob_low_auroc (oob_median={oob_auroc_median:.3f}; <0.60 means the cell "
                f"selection didn't generalize beyond the calibration sample)"
            )
        if np.isfinite(best_auroc):
            gap = best_auroc - oob_auroc_median
            if gap > 0.15:
                w.append(
                    f"large_oob_in_sample_gap (in_sample={best_auroc:.3f}, "
                    f"oob_median={oob_auroc_median:.3f}, gap={gap:.3f}; "
                    f"in-sample AUROC is materially over-stated by selection bias)"
                )
    if winner_stability is not None and np.isfinite(winner_stability):
        if winner_stability < 0.70:
            w.append(
                f"winner_unstable (winner_stability={winner_stability:.2f}; "
                f"a different panel cell wins on >30% of bootstrap resamples — "
                f"the chosen cell is noise-driven at this n)"
            )
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration entry point
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationState:
    """Loaded model + everything needed to run calibrate_with_state(). Build
    once via load_calibration_state(model_slug, ...); reuse across multiple
    calibrations (e.g. ANLI R1/R2/R3 on the same model) to skip the model
    load cost. Not part of the persisted profile."""
    model_slug: str
    model: Any
    tokenizer: Any
    projection: Any
    layer_indices: Dict[str, int]
    pri_computer: Any
    prompt_strategy: Any
    layer_name: str
    seed: int


def _build_derivation(
    cell: PanelCell,
    resid_coeffs: Dict[PanelCell, Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    """Construct the `derivation` payload for `profile.detector.metric` so
    the detector can reproduce the score at deploy time. Returns None for
    direct cells (the detector falls through to the legacy
    `result[column_name]` path).

    Composite cells: `{"kind": "composite", "formula": "<label>"}` — the
    detector parses the label via the same _compose_score dispatcher.

    Residualized cells: `{"kind": "residualized", "base_column": "...",
    "regress_against": "d_F_full", "regress_against_step": int, "b0": ...,
    "b1": ...}` — the detector pulls base_column + the regress-against
    column from compute_step output, computes `raw − (b0 + b1 · regressor)`.
    """
    step, fam, label = cell
    if fam == DERIVED_COMPOSITE_FAMILY:
        return {"kind": "composite", "formula": label}
    if fam in DERIVED_RESID_FAMILIES:
        coeffs = resid_coeffs.get(cell)
        if not coeffs:
            # Residualization wasn't successfully fit for this cell — the
            # cell column is NaN, so this shouldn't be a winner. Defensive.
            return None
        return {
            "kind": "residualized",
            "base_column": coeffs["base_column"],
            "regress_against": coeffs["regress_against"],
            "regress_against_step": coeffs["regress_against_step"],
            "b0": coeffs["b0"],
            "b1": coeffs["b1"],
        }
    return None


def _residualize_in_place(
    score_matrix: np.ndarray,
    panel: List[PanelCell],
    *,
    prompts_n: int,
) -> Dict[PanelCell, Dict[str, float]]:
    """For each `*_resid` cell in the panel, replace its column in the
    score matrix with `raw − predicted(raw | d_F_full @ same_step)`. The
    regression is OLS `y = b0 + b1 * d_F_full` fitted across n samples.
    NaNs in either column drop the sample from the fit AND from the
    residual output (residual stays NaN for that row).

    STEP-MATCHED (Codex review 2026-05-14): `d_F_full` is looked up by
    step. If the panel doesn't contain `(step, "scalar", "d_F_full")` for
    a resid cell's step, the cell's column is filled with NaN — we never
    silently regress against a different step's magnitude.

    Returns a dict mapping each successfully-residualized PanelCell to
    `{"b0": float, "b1": float, "base_column": str, "regress_against": str}`
    so the calibrator can persist these coefficients in the profile and
    the detector can reproduce the residual at score time. Cells that
    couldn't be residualized (insufficient samples or missing d_F_full)
    are absent from the returned dict.
    """
    col_index: Dict[PanelCell, int] = {cell: j for j, cell in enumerate(panel)}

    # Index d_F_full by gen_step.
    d_F_col_by_step: Dict[int, int] = {}
    for cell in panel:
        if cell[1] == "scalar" and cell[2] == "d_F_full":
            d_F_col_by_step[cell[0]] = col_index[cell]
    if not d_F_col_by_step:
        return {}

    coeffs: Dict[PanelCell, Dict[str, float]] = {}
    for j, cell in enumerate(panel):
        if cell[1] not in DERIVED_RESID_FAMILIES:
            continue
        step = cell[0]
        if step not in d_F_col_by_step:
            # No d_F_full at this step → can't fit a same-step residual.
            score_matrix[:, j] = np.nan
            continue
        d_F_values = score_matrix[:, d_F_col_by_step[step]]
        raw_values = score_matrix[:, j].copy()
        finite_mask = np.isfinite(raw_values) & np.isfinite(d_F_values)
        if finite_mask.sum() < 3:
            score_matrix[:, j] = np.nan
            continue
        x = d_F_values[finite_mask]
        y = raw_values[finite_mask]
        b1, b0 = np.polyfit(x, y, 1)  # OLS deg=1 → (slope, intercept)
        predicted = b0 + b1 * d_F_values
        residuals = raw_values - predicted
        residuals = np.where(finite_mask, residuals, np.nan)
        score_matrix[:, j] = residuals

        # Record coefficients for downstream persistence.
        base_fam = _resid_base_family(cell[1])
        base_column = _column_name((step, base_fam, cell[2]))
        coeffs[cell] = {
            "b0": float(b0),
            "b1": float(b1),
            "base_column": str(base_column),
            "regress_against": "d_F_full",
            "regress_against_step": int(step),
        }
    return coeffs


def load_calibration_state(
    model_slug: str,
    *,
    layer_name: str = "final",
    seed: int = 20260512,
) -> CalibrationState:
    """Load the model + tokenizer + PRIComputer once. The returned state can
    be passed to `calibrate_with_state(state, ...)` repeatedly with different
    calibration jsonls without paying the model-load cost each time.
    """
    cfg = pipeline.Config()
    cfg.layers_to_probe = [layer_name]
    cfg.seed = seed
    model, tokenizer, projection, layer_indices = pipeline.load_model(model_slug, cfg)
    gamma = pipeline._extract_final_rmsnorm_gamma(model)
    if gamma is None:
        raise RuntimeError(
            f"could not extract final-RMSNorm gamma for {model_slug}; "
            f"check pri_v2_mlx_pipeline._extract_final_rmsnorm_gamma logs."
        )
    pri_computer = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(model_slug)
    return CalibrationState(
        model_slug=model_slug,
        model=model,
        tokenizer=tokenizer,
        projection=projection,
        layer_indices=layer_indices,
        pri_computer=pri_computer,
        prompt_strategy=prompt_strategy,
        layer_name=layer_name,
        seed=seed,
    )


def calibrate_with_state(
    state: CalibrationState,
    calibration_jsonl_path: str,
    *,
    task_label: str = "",
    panel: Optional[List[PanelCell]] = None,
    n_bootstrap: int = 1000,
    max_new_tokens: int = 8,
    alpha: float = 1.0,
) -> CalibrationProfile:
    """Run the calibration pass using a pre-loaded model state. This is the
    inner work — see `calibrate()` for the single-shot wrapper that loads +
    runs in one call.

    Use this when you want to calibrate the same model on multiple datasets
    (e.g. ANLI R1/R2/R3) without reloading the model each time.
    """
    panel = list(panel or DEFAULT_PANEL)
    prompts, labels, data_hash = _load_calibration_jsonl(calibration_jsonl_path)
    n_calibration = len(prompts)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print(f"[calibrate] model={state.model_slug}  task={task_label or '(unset)'}")
    print(f"[calibrate] n_calibration={n_calibration} (pos={n_pos}, neg={n_neg})")
    print(f"[calibrate] panel={[_cell_label(c) for c in panel]}")

    # ── Attention setup (only if panel contains Attention cells) ───────
    # Wraps the per-sample trace_sample in the diagnostic module's
    # observational attention_capture context manager. Doubles per-sample
    # wall (~2× — manual SDPA at 3 target blocks vs fused kernel) so this
    # path is gated on actually needing attention captures.
    capture_attention = _requires_attention_capture(panel)
    capture_v_norms = _requires_v_norm_capture(panel) if capture_attention else False
    attention_target_map: Dict[str, int] = {}
    attention_n_kv_heads: Dict[str, int] = {}
    if capture_attention:
        # Deferred import — breaks circular import with the diagnostic.
        from diagnose_inter_head_disagreement import (
            _find_layers, _target_layer_map, attention_capture,
            attention_capture_with_values,
        )
        decoder_layers = _find_layers(state.model)
        attention_target_map = _target_layer_map(len(decoder_layers))
        for tag, idx in attention_target_map.items():
            n_kv = getattr(decoder_layers[idx].self_attn, "n_kv_heads", None)
            if n_kv is None:
                # MHA models: fall back to n_heads (q heads). Means
                # js_kv_groups == js for these; that's the right answer
                # since there are no KV groups to collapse over.
                n_kv = getattr(decoder_layers[idx].self_attn, "n_heads", None)
            if n_kv is not None:
                attention_n_kv_heads[tag] = int(n_kv)
        print(f"[calibrate] attention capture active: target_map={attention_target_map} "
              f"n_kv_heads={attention_n_kv_heads}  v_norms={capture_v_norms}")

    # Per-sample × per-cell score matrix.
    n_cells = len(panel)
    score_matrix = np.full((n_calibration, n_cells), np.nan, dtype=np.float64)
    print(f"[calibrate] tracing {n_calibration} samples...")
    for i, prompt in enumerate(prompts):
        sample_v_norm_captures: Optional[Dict[str, List[Any]]] = None
        if capture_attention and capture_v_norms:
            with attention_capture_with_values(
                decoder_layers, attention_target_map,
            ) as (caps, v_caps):
                trace = _trace_one_prompt(
                    state.model, state.tokenizer, state.projection, state.layer_indices,
                    prompt, state.prompt_strategy, max_new_tokens,
                )
                sample_captures = {tag: list(caps[tag]) for tag in caps}
                sample_v_norm_captures = {tag: list(v_caps[tag]) for tag in v_caps}
        elif capture_attention:
            with attention_capture(decoder_layers, attention_target_map) as caps:
                trace = _trace_one_prompt(
                    state.model, state.tokenizer, state.projection, state.layer_indices,
                    prompt, state.prompt_strategy, max_new_tokens,
                )
                # Snapshot captures while still inside the context — the
                # wrapper restores native attention on exit, so we read
                # the per-call list refs before that point.
                sample_captures = {tag: list(caps[tag]) for tag in caps}
        else:
            trace = _trace_one_prompt(
                state.model, state.tokenizer, state.projection, state.layer_indices,
                prompt, state.prompt_strategy, max_new_tokens,
            )
            sample_captures = None

        per_cell = _compute_panel_scores_for_sample(
            state.pri_computer, trace, state.layer_name, panel, alpha=alpha,
            attention_captures=sample_captures,
            attention_n_kv_heads=attention_n_kv_heads if capture_attention else None,
            attention_v_norm_captures=sample_v_norm_captures,
        )
        for j, cell in enumerate(panel):
            v = per_cell.get(cell)
            if v is not None:
                score_matrix[i, j] = v
        if (i + 1) % 10 == 0 or i + 1 == n_calibration:
            print(f"[calibrate]   {i+1}/{n_calibration}")

    # Post-loop residualization pass (E18 sealed primary form). For each
    # `*_resid` cell, regress the cell's CURRENT score_matrix column
    # (which holds the raw null_ratio values at this point) against the
    # d_F_full column AT THE SAME STEP across all calibration samples, and
    # replace the column with residuals. Cells whose corresponding d_F_full
    # is missing at that step (or with <3 finite pairs) become NaN.
    # The returned coefficients let us persist (b0, b1) per residualized
    # cell so the detector can reproduce the residual at score time.
    resid_coeffs = _residualize_in_place(score_matrix, panel, prompts_n=n_calibration)

    # Score every candidate; pick best by |AUROC - 0.5|.
    candidate_results = []
    panel_eval_counts: Dict[PanelCell, int] = {}
    best_idx = -1
    best_distance = -1.0
    for j, cell in enumerate(panel):
        scores_j = score_matrix[:, j]
        auc, sign, n_eval = _score_candidate(scores_j, labels)
        panel_eval_counts[cell] = n_eval
        candidate_results.append({
            "cell": _cell_label(cell),
            "step": cell[0],
            "family": cell[1],
            "rank_label": cell[2],
            "column_name": _column_name(cell),
            "auroc": auc,
            "sign": sign,
            "n_evaluated": n_eval,
        })
        if np.isfinite(auc):
            d = abs(auc - 0.5)
            if d > best_distance:
                best_distance = d
                best_idx = j

    if best_idx < 0:
        raise RuntimeError(
            "no candidate cell produced a finite AUROC — calibration data "
            "may be too small or all samples EOS'd before reaching panel steps."
        )

    best_cell = panel[best_idx]
    best_auroc = float(candidate_results[best_idx]["auroc"])
    best_sign = int(candidate_results[best_idx]["sign"])

    print(f"[calibrate] best cell: {_cell_label(best_cell)}  "
          f"AUROC={best_auroc:.3f}  sign={best_sign:+d}")

    # In-sample bootstrap CI on the locked sign (legacy semantics).
    ci_lo, ci_hi = _bootstrap_auroc(
        score_matrix[:, best_idx], labels, best_sign, n_bootstrap, state.seed,
    )
    print(f"[calibrate] in-sample 95%% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"(n_bootstrap={n_bootstrap})")

    # Nested OOB bootstrap — the honest deployment estimate. Re-runs cell
    # selection inside each resample, evaluates on the out-of-bag samples.
    # 2026-05-13: added in response to the Codex adversarial review's
    # post-selection-bias finding.
    print(f"[calibrate] nested OOB bootstrap...")
    oob_stats = _nested_bootstrap_oob_auroc(
        score_matrix, labels, panel, n_bootstrap, state.seed,
    )
    if oob_stats["oob_n_bootstrap_used"] > 0:
        print(
            f"[calibrate] OOB median AUROC: {oob_stats['oob_auroc_median']:.3f}  "
            f"CI [{oob_stats['oob_auroc_ci_lo']:.3f}, {oob_stats['oob_auroc_ci_hi']:.3f}]  "
            f"(used {oob_stats['oob_n_bootstrap_used']}/{n_bootstrap} rounds)"
        )
        print(
            f"[calibrate] winner stability: {oob_stats['winner_stability']:.2f}  "
            f"counts: {oob_stats['winner_counts']}"
        )
    else:
        print("[calibrate] OOB bootstrap: 0/N usable rounds (calibration set "
              "too small or degenerate); deployment estimate unavailable")

    warnings_list = _emit_warnings(
        n_calibration, n_pos, n_neg, best_auroc, ci_lo, ci_hi, panel_eval_counts,
        oob_auroc_median=oob_stats["oob_auroc_median"],
        winner_stability=oob_stats["winner_stability"],
    )
    for w in warnings_list:
        print(f"[calibrate]   WARNING: {w}")

    pipeline_path = REPO_ROOT / "pri_v2_mlx_pipeline.py"
    io_plugins_path = REPO_ROOT / "pri_v2_io_plugins.py"
    model_adapters_path = REPO_ROOT / "model_adapters.py"
    attention_wrapper_path = REPO_ROOT / "scripts" / "diagnose_inter_head_disagreement.py"
    model_snapshot_sha = _resolve_model_snapshot_sha(state.model_slug)
    profile = CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        model={
            "slug": state.model_slug,
            "output_projection_kind": state.projection.mode,
        },
        task={
            "label": task_label,
            "n_calibration": n_calibration,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "data_hash_sha256": data_hash,
        },
        detector={
            "gen_step": int(best_cell[0]),
            "layer": state.layer_name,
            "alpha": float(alpha),
            "metric": {
                "family": best_cell[1],
                "label": best_cell[2],
                "column_name": _column_name(best_cell),
                # `derivation` tells Detector.score() how to compute this
                # metric at deploy time. None for direct cells (the detector
                # just looks up column_name in compute_step output). Set for
                # composite + residualized winners — schema v1.2.
                "derivation": _build_derivation(best_cell, resid_coeffs),
            },
            "sign": best_sign,
            "threshold": None,  # researcher chooses at deploy time (Youden's J etc.)
        },
        calibration_stats={
            # In-sample post-selection — kept for inspection but NOT deployable.
            "auroc": best_auroc,
            "auroc_bootstrap_ci_lo": ci_lo,
            "auroc_bootstrap_ci_hi": ci_hi,
            # OOB stats — the honest deployment estimate.
            "oob_auroc_median": oob_stats["oob_auroc_median"],
            "oob_auroc_ci_lo": oob_stats["oob_auroc_ci_lo"],
            "oob_auroc_ci_hi": oob_stats["oob_auroc_ci_hi"],
            "oob_n_bootstrap_used": oob_stats["oob_n_bootstrap_used"],
            "winner_stability": oob_stats["winner_stability"],
            "winner_counts": oob_stats["winner_counts"],
            "candidate_panel": candidate_results,
        },
        provenance={
            "calibration_seed": int(state.seed),
            "n_bootstrap": int(n_bootstrap),
            "pipeline_module_hash_sha256": _hash_file(pipeline_path),
            "io_plugins_module_hash_sha256": _hash_file(io_plugins_path),
            "model_adapters_module_hash_sha256": _hash_file(model_adapters_path),
            "calibrator_module_hash_sha256": _hash_file(REPO_ROOT / "pri_calibrator.py"),
            # None when no Attention cell was in the panel; detector strict
            # mode only checks this when the loaded profile's winner is an
            # Attention cell (the wrapper module isn't a runtime dep otherwise).
            "attention_wrapper_module_hash_sha256": (
                _hash_file(attention_wrapper_path) if capture_attention else None
            ),
            "model_snapshot_sha": model_snapshot_sha,  # may be None if uncached
            "calibrated_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_new_tokens": int(max_new_tokens),
        },
        warnings=warnings_list,
    )
    return profile


def calibrate(
    model_slug: str,
    calibration_jsonl_path: str,
    *,
    task_label: str = "",
    panel: Optional[List[PanelCell]] = None,
    seed: int = 20260512,
    n_bootstrap: int = 1000,
    max_new_tokens: int = 8,
    layer_name: str = "final",
    alpha: float = 1.0,
) -> CalibrationProfile:
    """Single-shot calibration: load the model then run the calibration pass.

    Thin wrapper around `load_calibration_state` + `calibrate_with_state`.
    Use the two-step form directly when you want to calibrate the same model
    on multiple datasets without reloading.

    `max_new_tokens` defaults to 8 — enough to cover gen_step ∈ {1..5} for the
    default panel (max panel step is 4; one extra for safety).
    """
    state = load_calibration_state(model_slug, layer_name=layer_name, seed=seed)
    return calibrate_with_state(
        state,
        calibration_jsonl_path,
        task_label=task_label,
        panel=panel,
        n_bootstrap=n_bootstrap,
        max_new_tokens=max_new_tokens,
        alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="PRI calibrator (v1.2 schema)")
    p.add_argument("--model", required=True, help="model slug, e.g. mlx-community/Mistral-Nemo-Instruct-2407-4bit")
    p.add_argument("--data", required=True, help="calibration jsonl path")
    p.add_argument("--out", required=True, help="output profile json path")
    p.add_argument("--task-label", default="", help="task identifier for provenance (e.g. 'anli_r2_dev')")
    p.add_argument("--seed", type=int, default=20260512)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="generation budget per sample (default 8, or 1 when --t0-commit)")
    p.add_argument("--layer", default="final", help="capture layer (default: final)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--attention", action="store_true",
        help="extend the panel with the 12-cell ATTENTION_PANEL (3 layers × 4 metrics × gen_step=1). "
             "Wraps trace_sample in the diagnostic module's observational attention_capture context "
             "manager; ~2× per-sample wall.",
    )
    p.add_argument(
        "--attention-multistep", action="store_true",
        help="extend the panel with the 48-cell ATTENTION_PANEL_MULTISTEP (3 layers × 4 metrics × "
             "gen_step ∈ {1,2,3,4}). Probes post-commit attention dynamics in addition to the "
             "commit step. Multi-testing burden is 4× larger so OOB safety warnings fire more "
             "aggressively at small n — that's correct behavior, not noise. Implies --attention.",
    )
    p.add_argument(
        "--attention-with-v-norms", action="store_true",
        help="extend the panel with the 21-cell ATTENTION_PANEL_WITH_V_NORMS — adds 3 SinkProbe-"
             "style value-norm metrics (v_norm_bos, v_norm_max, v_norm_lastq_weighted) at the "
             "commit step on top of the 12 default attention cells. Switches to the "
             "attention_capture_with_values capture path; <5%% extra wall vs --attention.",
    )
    p.add_argument(
        "--attention-only", action="store_true",
        help="use the attention panel alone (no DEFAULT_PANEL cells). Combines with "
             "--attention / --attention-multistep / --attention-with-v-norms to pick "
             "which attention panel.",
    )
    p.add_argument(
        "--t0-commit", action="store_true",
        help="use the t=0 prefill-last-position attention locus instead of gen_step=1. "
             "Captures captures[layer][0] (the prefill forward's last-query-row) rather "
             "than captures[layer][1] (the first generated token's attention). "
             "Semantically consistent across all models: always the last prompt token's "
             "attention over the full prefix. Implies --attention-only (use with "
             "--attention-with-v-norms for the 21-cell panel). "
             "Sets max_new_tokens=1 unless overridden (prefill is sufficient).",
    )
    p.add_argument(
        "--t0-residual", action="store_true",
        help="use the 5-cell T0_RESIDUAL_PANEL — residual-stream cells (d_F_full, "
             "kl_discharged, Fisher r=1/r=2, Raw r=21) measured at the t=0 prefix-last-"
             "position locus. Structurally consistent across all model families. "
             "Sets max_new_tokens=1 (prefill + one token for S_t).",
    )
    args = p.parse_args()

    # t=0-commit overrides the step index. Panel constants use step=0 instead of step=1.
    if args.t0_commit:
        if args.attention_with_v_norms:
            attn_panel = list(ATTENTION_PANEL_T0_WITH_V_NORMS)
        else:
            attn_panel = list(ATTENTION_PANEL_T0)
        # Prefill is sufficient; one generation step to complete the trace cleanly.
        if args.max_new_tokens is None:
            args.max_new_tokens = 1
    elif args.attention_multistep:
        attn_panel = list(ATTENTION_PANEL_MULTISTEP)
    elif args.attention_with_v_norms:
        attn_panel = list(ATTENTION_PANEL_WITH_V_NORMS)
    elif args.attention or args.attention_only:
        attn_panel = list(ATTENTION_PANEL)
    else:
        attn_panel = []

    if args.t0_residual and args.max_new_tokens is None:
        args.max_new_tokens = 1
    if args.max_new_tokens is None:
        args.max_new_tokens = 8

    if args.t0_residual:
        panel = list(T0_RESIDUAL_PANEL)
    elif args.attention_only or args.t0_commit:
        panel = list(attn_panel) if attn_panel else list(ATTENTION_PANEL)
    elif attn_panel:
        panel = list(DEFAULT_PANEL) + attn_panel
    else:
        panel = None  # use DEFAULT_PANEL

    profile = calibrate(
        model_slug=args.model,
        calibration_jsonl_path=args.data,
        task_label=args.task_label,
        panel=panel,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        max_new_tokens=args.max_new_tokens,
        layer_name=args.layer,
        alpha=args.alpha,
    )
    profile.to_json(args.out)
    print(f"[calibrate] wrote profile: {args.out}")
    if profile.warnings:
        print(f"[calibrate] {len(profile.warnings)} warning(s) — see profile['warnings']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
