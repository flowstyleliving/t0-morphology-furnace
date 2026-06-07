#!/usr/bin/env python3
"""Literal t=0 belief readout for the step-0 ANLI panel.

This path intentionally stays narrow:
  * literal YES/NO token buckets only
  * no continuation rescue path
  * no synonym widening
  * no threshold learned from preamble behavior

The prompt contract fixes the score direction up front:
  YES = entailment / consistency
  NO  = contradiction
So contradiction label B is scored on -lean where
  lean = log((p_yes + eps) / (p_no + eps)).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _load_calibration_jsonl


DEFAULT_DATA = REPO_ROOT / "experiments" / "anli-sweep" / "2026-05-15" / "run-02" / "anli_R1_seed20260513_n100.jsonl"
# --prereg is required (no default): the prereg doc lives in the vault, and
# repo files must not carry vault-path refs (repo<->wiki separation).
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_SEED = 20260423
EPS = 1e-12
HIGH_COVERAGE_BAR = 0.80
ANCHOR_MIN_AGREEMENT = 0.95
CONTROL_FLOOR_MULTIPLIER = 5.0
SCORING_MODE = "token_partition_v1"
MISTRAL_NEMO_SLUG = "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
LOCKED_MODEL_PANEL = (
    "mlx-community/Qwen3-1.7B-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
)
PROMPT_IDENTITY_PROBE = (
    "Premise: A robin is a bird.\n"
    "Hypothesis: A robin is an animal.\n"
    "Answer YES or NO only."
)
OUT_OF_SCOPE_LINE = (
    "Downstream self-retraction / trajectory self-contradiction is a separate "
    "later-gen-step experiment and is forbidden from entering this metric."
)
ANCHOR_RATIONALE = (
    "Mistral-Nemo is the immediate-commit validity anchor. Agreement stays at "
    "0.95 because this check is guarding the measurement premise itself, not a "
    "descriptive downstream claim."
)
SEMANTIC_AFFIRMATIVE = frozenset({"yes", "yeah", "yep", "correct", "true", "right"})
SEMANTIC_NEGATIVE = frozenset({"no", "nope", "incorrect", "false", "wrong"})
_SPECIAL_TOKEN_RE = re.compile(r"^<\|[^|>]+?\|>$")


def short_model_name(slug: str) -> str:
    return slug.split("/")[-1]


def canary_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_belief_canary.json"


def spec_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_belief_spec.json"


def readout_csv_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_belief_readout.csv"


def readout_json_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_belief_readout.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp_path, path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def _normalize_literal_token(text: str) -> str:
    s = str(text).casefold()
    while s.startswith(("▁", "Ġ", "ġ")):
        s = s[1:]
    return s.strip(" \t\r\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _callable_source_sha256(fn: Any) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = repr(fn)
    return _sha256_text(source)


def prompt_identity_bundle(model_slug: str, tokenizer: Any) -> Dict[str, str]:
    strategy = io_plugins.get_prompt_strategy(model_slug)
    rendered = strategy(PROMPT_IDENTITY_PROBE, tokenizer)
    if not isinstance(rendered, str) or rendered == "":
        raise RuntimeError(
            f"prompt identity probe rendered invalid prompt for {model_slug}: {type(rendered).__name__}"
        )
    return {
        "prompt_strategy_name": strategy.__name__,
        "prompt_strategy_source_sha256": _callable_source_sha256(strategy),
        "prompt_probe_input_sha256": _sha256_text(PROMPT_IDENTITY_PROBE),
        "prompt_probe_output_sha256": _sha256_text(rendered),
    }


def validate_locked_model_panel(model_slugs: Sequence[str]) -> None:
    actual = tuple(str(slug) for slug in model_slugs)
    if actual != LOCKED_MODEL_PANEL:
        raise RuntimeError(
            "locked 10-model panel mismatch: "
            f"expected {list(LOCKED_MODEL_PANEL)}, got {list(actual)}"
        )


def _require_finite_array(name: str, values: np.ndarray, *, model_slug: str, sample_idx: int) -> None:
    arr = np.asarray(values)
    if np.all(np.isfinite(arr)):
        return
    bad = np.flatnonzero(~np.isfinite(arr))
    preview = ",".join(str(int(x)) for x in bad[:5])
    raise RuntimeError(
        f"[{model_slug}] sample {sample_idx}: non-finite values in {name} at index/indices {preview}"
    )


def _require_finite_scalars(
    values: Dict[str, float],
    *,
    model_slug: str,
    sample_idx: int,
) -> None:
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise RuntimeError(
                f"[{model_slug}] sample {sample_idx}: non-finite {name}={value!r}"
            )


def _validate_live_literal_token_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    expected_normalized: str,
    bucket_name: str,
    model_slug: str,
) -> List[str]:
    decoded_tokens: List[str] = []
    for token_id in token_ids:
        decoded = pipeline.decode_ids(tokenizer, [int(token_id)])
        normalized = _normalize_literal_token(decoded)
        if normalized != expected_normalized:
            raise RuntimeError(
                f"[{model_slug}] frozen {bucket_name} token id {token_id} re-decodes to "
                f"{decoded!r} (normalized={normalized!r}), not literal {expected_normalized!r}"
            )
        decoded_tokens.append(decoded)
    return decoded_tokens


def literal_yes_no_token_buckets(tokenizer: Any, vocab_size: int) -> Dict[str, Any]:
    yes_ids: List[int] = []
    no_ids: List[int] = []
    yes_tokens: List[str] = []
    no_tokens: List[str] = []
    for token_id in range(vocab_size):
        decoded = pipeline.decode_ids(tokenizer, [token_id])
        normalized = _normalize_literal_token(decoded)
        if normalized == "yes":
            yes_ids.append(token_id)
            yes_tokens.append(decoded)
        elif normalized == "no":
            no_ids.append(token_id)
            no_tokens.append(decoded)
    if not yes_ids or not no_ids:
        raise RuntimeError(
            f"literal token rule found empty bucket(s): yes={len(yes_ids)} no={len(no_ids)}"
        )
    return {
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "yes_tokens": yes_tokens,
        "no_tokens": no_tokens,
    }


def semantic_shortlist_token_buckets(tokenizer: Any, vocab_size: int) -> Dict[str, Any]:
    yes_ids: List[int] = []
    no_ids: List[int] = []
    yes_tokens: List[str] = []
    no_tokens: List[str] = []
    for token_id in range(vocab_size):
        decoded = pipeline.decode_ids(tokenizer, [token_id])
        normalized = _normalize_literal_token(decoded)
        if normalized in SEMANTIC_AFFIRMATIVE:
            yes_ids.append(token_id)
            yes_tokens.append(decoded)
        elif normalized in SEMANTIC_NEGATIVE:
            no_ids.append(token_id)
            no_tokens.append(decoded)
    return {
        "semantic_yes_token_ids": yes_ids,
        "semantic_no_token_ids": no_ids,
        "semantic_yes_tokens": yes_tokens,
        "semantic_no_tokens": no_tokens,
    }


def control_marker_token_bundle(tokenizer: Any, vocab_size: int) -> Dict[str, Any]:
    control_ids: List[int] = []
    control_tokens: List[str] = []
    for token_id in range(vocab_size):
        decoded = pipeline.decode_ids(tokenizer, [token_id])
        stripped = _normalize_literal_token(decoded)
        raw_stripped = str(decoded).strip()
        if stripped == "" or _SPECIAL_TOKEN_RE.fullmatch(raw_stripped):
            control_ids.append(token_id)
            control_tokens.append(decoded)
    if not control_ids:
        raise RuntimeError("control-marker rule found no control tokens")
    return {
        "control_token_ids": control_ids,
        "control_tokens": control_tokens,
    }


def _gold_yes_no(label: int) -> str:
    return "NO" if int(label) == 1 else "YES"


def _score_row_from_probs(
    probs: np.ndarray,
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
    semantic_yes_token_ids: Sequence[int],
    semantic_no_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
) -> Dict[str, float]:
    p_yes = float(np.sum(probs[list(yes_token_ids)], dtype=np.float64))
    p_no = float(np.sum(probs[list(no_token_ids)], dtype=np.float64))
    decidedness = p_yes + p_no
    semantic_yes = float(np.sum(probs[list(semantic_yes_token_ids)], dtype=np.float64))
    semantic_no = float(np.sum(probs[list(semantic_no_token_ids)], dtype=np.float64))
    semantic_decidedness = semantic_yes + semantic_no
    control_mass = float(np.sum(probs[list(control_token_ids)], dtype=np.float64))
    decidedness_floor = float(CONTROL_FLOOR_MULTIPLIER * control_mass)
    lean = float(math.log((p_yes + EPS) / (p_no + EPS)))
    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "decidedness": decidedness,
        "semantic_yes_mass": semantic_yes,
        "semantic_no_mass": semantic_no,
        "semantic_decidedness": semantic_decidedness,
        "semantic_offliteral_mass": max(0.0, semantic_decidedness - decidedness),
        "control_mass": control_mass,
        "decidedness_floor": decidedness_floor,
        "above_floor": bool(decidedness > decidedness_floor),
        "lean": lean,
    }


def _top_token_records(tokenizer: Any, probs: np.ndarray, *, k: int = 10) -> List[Dict[str, Any]]:
    top_ids = np.argsort(-probs)[:k]
    out: List[Dict[str, Any]] = []
    for rank, token_id in enumerate(top_ids, start=1):
        decoded = pipeline.decode_ids(tokenizer, [int(token_id)])
        out.append({
            "rank": rank,
            "token_id": int(token_id),
            "decoded": decoded,
            "normalized": _normalize_literal_token(decoded),
            "prob": float(probs[int(token_id)]),
        })
    return out


def _prefix_summary_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _make_spec(
    *,
    model_slug: str,
    data_hash_sha256: str,
    prompt_identity: Dict[str, str],
    tokenizer_fix_flags: Dict[str, Any],
    token_buckets: Dict[str, Any],
    canary_path: Path,
    prereg_path: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": "belief_spec_v1",
        "model_slug": model_slug,
        "data_hash_sha256": data_hash_sha256,
        "prompt_strategy_name": prompt_identity["prompt_strategy_name"],
        "prompt_strategy_source_sha256": prompt_identity["prompt_strategy_source_sha256"],
        "prompt_probe_input_sha256": prompt_identity["prompt_probe_input_sha256"],
        "prompt_probe_output_sha256": prompt_identity["prompt_probe_output_sha256"],
        "scoring_mode": SCORING_MODE,
        "tokenizer_fix_flags": dict(tokenizer_fix_flags),
        "yes_token_ids": list(token_buckets["yes_token_ids"]),
        "no_token_ids": list(token_buckets["no_token_ids"]),
        "yes_tokens": list(token_buckets["yes_tokens"]),
        "no_tokens": list(token_buckets["no_tokens"]),
        "semantic_affirmative_forms": sorted(SEMANTIC_AFFIRMATIVE),
        "semantic_negative_forms": sorted(SEMANTIC_NEGATIVE),
        "semantic_yes_token_ids": list(token_buckets["semantic_yes_token_ids"]),
        "semantic_no_token_ids": list(token_buckets["semantic_no_token_ids"]),
        "semantic_yes_tokens": list(token_buckets["semantic_yes_tokens"]),
        "semantic_no_tokens": list(token_buckets["semantic_no_tokens"]),
        "control_token_ids": list(token_buckets["control_token_ids"]),
        "control_tokens": list(token_buckets["control_tokens"]),
        "control_floor_multiplier": CONTROL_FLOOR_MULTIPLIER,
        "canary_path": _prefix_summary_path(canary_path),
        "prereg_path": _prefix_summary_path(prereg_path),
        "anchor_rationale": ANCHOR_RATIONALE,
        "out_of_scope": OUT_OF_SCOPE_LINE,
    }


def _write_frozen_spec(path: Path, spec: Dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if existing != spec:
            raise RuntimeError(
                f"refusing to overwrite frozen spec with different content: {path}"
            )
        return
    _write_json(path, spec)


def _validate_spec(
    spec: Dict[str, Any],
    *,
    model_slug: str,
    data_hash_sha256: str,
    prompt_identity: Dict[str, str],
    tokenizer_fix_flags: Dict[str, Any],
    tokenizer: Any,
) -> Dict[str, Any]:
    if spec.get("schema_version") != "belief_spec_v1":
        raise RuntimeError(f"unsupported spec schema: {spec.get('schema_version')}")
    if spec.get("scoring_mode") != SCORING_MODE:
        raise RuntimeError(f"unsupported scoring mode: {spec.get('scoring_mode')}")
    if spec.get("model_slug") != model_slug:
        raise RuntimeError(
            f"spec model mismatch: expected {model_slug}, got {spec.get('model_slug')}"
        )
    if spec.get("data_hash_sha256") != data_hash_sha256:
        raise RuntimeError(
            f"spec data hash mismatch: expected {data_hash_sha256}, got {spec.get('data_hash_sha256')}"
        )
    if spec.get("prompt_strategy_name") != prompt_identity["prompt_strategy_name"]:
        raise RuntimeError(
            "spec prompt strategy mismatch: "
            f"expected {prompt_identity['prompt_strategy_name']}, got {spec.get('prompt_strategy_name')}"
        )
    for field in (
        "prompt_strategy_source_sha256",
        "prompt_probe_input_sha256",
        "prompt_probe_output_sha256",
    ):
        if spec.get(field) != prompt_identity[field]:
            raise RuntimeError(
                f"spec prompt identity mismatch for {field}: "
                f"expected {prompt_identity[field]}, got {spec.get(field)}"
            )
    if dict(spec.get("tokenizer_fix_flags", {})) != dict(tokenizer_fix_flags):
        raise RuntimeError(
            "spec tokenizer fix mismatch: "
            f"expected {tokenizer_fix_flags}, got {spec.get('tokenizer_fix_flags')}"
        )
    if float(spec.get("control_floor_multiplier", float("nan"))) != CONTROL_FLOOR_MULTIPLIER:
        raise RuntimeError(
            "spec control-floor multiplier mismatch: "
            f"expected {CONTROL_FLOOR_MULTIPLIER}, got {spec.get('control_floor_multiplier')}"
        )
    yes_token_ids = [int(x) for x in spec.get("yes_token_ids", [])]
    no_token_ids = [int(x) for x in spec.get("no_token_ids", [])]
    semantic_yes_token_ids = [int(x) for x in spec.get("semantic_yes_token_ids", [])]
    semantic_no_token_ids = [int(x) for x in spec.get("semantic_no_token_ids", [])]
    control_token_ids = [int(x) for x in spec.get("control_token_ids", [])]
    if not yes_token_ids or not no_token_ids:
        raise RuntimeError("spec has empty literal YES/NO bucket(s)")
    yes_tokens_live = _validate_live_literal_token_ids(
        tokenizer,
        yes_token_ids,
        expected_normalized="yes",
        bucket_name="YES",
        model_slug=model_slug,
    )
    no_tokens_live = _validate_live_literal_token_ids(
        tokenizer,
        no_token_ids,
        expected_normalized="no",
        bucket_name="NO",
        model_slug=model_slug,
    )
    return {
        "yes_token_ids": yes_token_ids,
        "no_token_ids": no_token_ids,
        "yes_tokens_live": yes_tokens_live,
        "no_tokens_live": no_tokens_live,
        "semantic_yes_token_ids": semantic_yes_token_ids,
        "semantic_no_token_ids": semantic_no_token_ids,
        "control_token_ids": control_token_ids,
    }


def _bootstrap_ci_fixed_direction(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(scores)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.min() == y.max():
            continue
        aucs.append(float(roc_auc_score(y, scores[idx])))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(aucs, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def build_coverage_curve(
    *,
    labels: np.ndarray,
    lean_scores: np.ndarray,
    decidedness: np.ndarray,
    eligible_mask: np.ndarray,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> List[Dict[str, Any]]:
    total = len(labels)
    if total == 0:
        return []
    _require_finite_array("lean_scores", lean_scores, model_slug="panel", sample_idx=-1)
    _require_finite_array("decidedness", decidedness, model_slug="panel", sample_idx=-1)
    eligible_idx = np.flatnonzero(eligible_mask)
    if len(eligible_idx) == 0:
        return []
    order = eligible_idx[np.argsort(-decidedness[eligible_idx], kind="stable")]
    y = labels[order].astype(int)
    s = (-lean_scores[order]).astype(np.float64)
    decided = decidedness[order].astype(np.float64)
    curve: List[Dict[str, Any]] = []
    for prefix_n in range(2, len(order) + 1):
        yp = y[:prefix_n]
        if len(np.unique(yp)) < 2:
            continue
        sp = s[:prefix_n]
        auc = float(roc_auc_score(yp, sp))
        ci_lo, ci_hi = _bootstrap_ci_fixed_direction(
            sp,
            yp,
            n_resamples=n_bootstrap,
            seed=seed,
        )
        curve.append({
            "n_prefix": prefix_n,
            "coverage": float(prefix_n / total),
            "auroc_b_signed": auc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "min_decidedness": float(decided[prefix_n - 1]),
        })
    return curve


def classify_verdict(
    curve: Sequence[Dict[str, Any]],
    *,
    eligible_coverage: float,
    undetermined_coverage: float,
) -> str:
    significant = [
        point for point in curve
        if point["ci_lo"] is not None and point["ci_lo"] > 0.50
    ]
    if any(point["coverage"] >= HIGH_COVERAGE_BAR for point in significant):
        return "Recoverable-for-M"
    if eligible_coverage < HIGH_COVERAGE_BAR and undetermined_coverage >= HIGH_COVERAGE_BAR:
        return "Undetermined-for-M"
    if eligible_coverage < HIGH_COVERAGE_BAR:
        return "Low-decidedness-for-M"
    return "Decided-but-non-B-for-M"


def _c_prediction_from_lean(lean: float) -> Optional[str]:
    if lean > 0:
        return "YES"
    if lean < 0:
        return "NO"
    return None


def _summarize_auxiliary_c(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    defined = [
        row for row in rows
        if row["above_floor"] and row["c_pred_yes_no"] is not None
    ]
    if not defined:
        return {
            "accuracy": None,
            "n_defined": 0,
            "coverage_defined": 0.0,
            "label_source": "gold_yes_no_from_B_prompt_contract",
        }
    correct = sum(1 for row in defined if row["c_pred_yes_no"] == row["gold_yes_no"])
    return {
        "accuracy": float(correct / len(defined)),
        "n_defined": int(len(defined)),
        "coverage_defined": float(len(defined) / len(rows)),
        "label_source": "gold_yes_no_from_B_prompt_contract",
    }


def _anchor_summary(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    anchor_rows = [
        row for row in rows
        if "anchor_match" in row and row["above_floor"] and row["c_pred_yes_no"] is not None
    ]
    if not anchor_rows:
        return None
    matches = sum(1 for row in anchor_rows if row["anchor_match"] is True)
    agreement = float(matches / len(anchor_rows)) if anchor_rows else float("nan")
    return {
        "n_total": int(len(anchor_rows)),
        "n_matches": int(matches),
        "agreement": agreement,
        "passed": bool(agreement >= ANCHOR_MIN_AGREEMENT),
    }


def _collect_sample_rows(
    *,
    model: Any,
    tokenizer: Any,
    output_projection: Any,
    layer_indices: Dict[str, int],
    prompts: Sequence[str],
    labels: Sequence[int],
    model_slug: str,
    prompt_strategy_name: str,
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
    semantic_yes_token_ids: Sequence[int],
    semantic_no_token_ids: Sequence[int],
    control_token_ids: Sequence[int],
    limit: int,
    include_anchor: bool,
    anchor_max_new_tokens: int,
) -> List[Dict[str, Any]]:
    strategy = io_plugins.get_prompt_strategy(model_slug)
    rows: List[Dict[str, Any]] = []
    n_rows = len(prompts) if limit <= 0 else min(limit, len(prompts))
    for i in range(n_rows):
        label = int(labels[i])
        wrapped = strategy(prompts[i], tokenizer)
        prefix = pipeline.prefix_readout(model, tokenizer, wrapped)
        _require_finite_array("last_probs", prefix["last_probs"], model_slug=model_slug, sample_idx=i)
        probs = prefix["last_probs"]
        score_row = _score_row_from_probs(
            probs,
            yes_token_ids,
            no_token_ids,
            semantic_yes_token_ids,
            semantic_no_token_ids,
            control_token_ids,
        )
        _require_finite_scalars(
            {
                "p_yes": score_row["p_yes"],
                "p_no": score_row["p_no"],
                "decidedness": score_row["decidedness"],
                "control_mass": score_row["control_mass"],
                "decidedness_floor": score_row["decidedness_floor"],
                "lean": score_row["lean"],
            },
            model_slug=model_slug,
            sample_idx=i,
        )
        top1_id = int(np.argmax(probs))
        top1_decoded = pipeline.decode_ids(tokenizer, [top1_id])
        top1_prob = float(probs[top1_id])
        gold_yes_no = _gold_yes_no(label)
        c_pred_yes_no = _c_prediction_from_lean(score_row["lean"])
        row: Dict[str, Any] = {
            "sample_idx": i,
            "label_B": label,
            "gold_yes_no": gold_yes_no,
            "p_yes": score_row["p_yes"],
            "p_no": score_row["p_no"],
            "decidedness": score_row["decidedness"],
            "semantic_yes_mass": score_row["semantic_yes_mass"],
            "semantic_no_mass": score_row["semantic_no_mass"],
            "semantic_decidedness": score_row["semantic_decidedness"],
            "semantic_offliteral_mass": score_row["semantic_offliteral_mass"],
            "control_mass": score_row["control_mass"],
            "decidedness_floor": score_row["decidedness_floor"],
            "above_floor": score_row["above_floor"],
            "lean": score_row["lean"],
            "c_pred_yes_no": c_pred_yes_no,
            "top1_token_id": top1_id,
            "top1_decoded": top1_decoded,
            "top1_normalized": _normalize_literal_token(top1_decoded),
            "top1_prob": top1_prob,
            "prompt_strategy_name": prompt_strategy_name,
        }
        if include_anchor:
            trace = pipeline.trace_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=wrapped,
                layer_indices=layer_indices,
                output_projection=output_projection,
                max_new_tokens=anchor_max_new_tokens,
                v3_capture=False,
            )
            generated = trace.get("generated_text") or ""
            parsed = io_plugins.parse_yes_no(generated)
            row["anchor_freegen_pred"] = parsed
            row["anchor_match"] = (
                bool(parsed is not None and parsed == c_pred_yes_no)
                if row["above_floor"] and c_pred_yes_no is not None
                else None
            )
        rows.append(row)
    return rows


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_idx",
        "label_B",
        "gold_yes_no",
        "p_yes",
        "p_no",
        "decidedness",
        "semantic_yes_mass",
        "semantic_no_mass",
        "semantic_decidedness",
        "semantic_offliteral_mass",
        "control_mass",
        "decidedness_floor",
        "above_floor",
        "lean",
        "c_pred_yes_no",
        "top1_token_id",
        "top1_decoded",
        "top1_normalized",
        "top1_prob",
        "prompt_strategy_name",
        "anchor_freegen_pred",
        "anchor_match",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_summary(
    *,
    model_slug: str,
    data_hash_sha256: str,
    prompt_strategy_name: str,
    spec_path: Path,
    prereg_path: Path,
    rows: Sequence[Dict[str, Any]],
    curve: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    auxiliary_c = _summarize_auxiliary_c(rows)
    anchor = _anchor_summary(rows)
    eligible_coverage = float(sum(1 for row in rows if row["above_floor"]) / len(rows)) if rows else 0.0
    undetermined_coverage = (
        float(
            sum(
                1
                for row in rows
                if row["semantic_offliteral_mass"] > row["decidedness"]
                and row["semantic_decidedness"] > row["decidedness_floor"]
            ) / len(rows)
        )
        if rows else 0.0
    )
    significant_coverages = [
        float(point["coverage"])
        for point in curve
        if point["ci_lo"] is not None and point["ci_lo"] > 0.50
    ]
    verdict = classify_verdict(
        curve,
        eligible_coverage=eligible_coverage,
        undetermined_coverage=undetermined_coverage,
    )
    return {
        "schema_version": "belief_readout_v1",
        "model": model_slug,
        "data_hash_sha256": data_hash_sha256,
        "prompt_strategy_name": prompt_strategy_name,
        "scoring_mode": SCORING_MODE,
        "n_total": int(len(rows)),
        "eps": EPS,
        "high_coverage_bar": HIGH_COVERAGE_BAR,
        "control_floor_multiplier": CONTROL_FLOOR_MULTIPLIER,
        "prereg_path": _prefix_summary_path(prereg_path),
        "spec_path": _prefix_summary_path(spec_path),
        "auxiliary_C": auxiliary_c,
        "curve": list(curve),
        "eligible_coverage": eligible_coverage,
        "undetermined_coverage": undetermined_coverage,
        "max_significant_coverage": max(significant_coverages) if significant_coverages else 0.0,
        "verdict": verdict,
        "anchor": anchor,
        "anchor_rationale": ANCHOR_RATIONALE,
        "out_of_scope": OUT_OF_SCOPE_LINE,
    }


def validate_panel_specs(
    *,
    spec_paths: Sequence[Path],
    expected_data_hash_sha256: str,
    expected_models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not spec_paths:
        raise RuntimeError("no spec paths supplied")
    model_slugs: List[str] = []
    hashes: set[str] = set()
    for spec_path in spec_paths:
        spec = _load_json(spec_path)
        model_slug = spec.get("model_slug")
        if not model_slug:
            raise RuntimeError(f"spec missing model slug: {spec_path}")
        model_slugs.append(str(model_slug))
        hashes.add(str(spec.get("data_hash_sha256")))
    if hashes != {expected_data_hash_sha256}:
        raise RuntimeError(
            f"spec data hashes do not match expected {expected_data_hash_sha256}: {sorted(hashes)}"
        )
    if expected_models is not None:
        validate_locked_model_panel(expected_models)
        if tuple(model_slugs) != tuple(expected_models):
            raise RuntimeError(
                f"spec model list mismatch: expected {list(expected_models)}, got {model_slugs}"
            )
    return {
        "n_specs": len(spec_paths),
        "data_hash_sha256": expected_data_hash_sha256,
        "model_slugs": model_slugs,
    }


def write_panel_summary_from_status(
    *,
    out_dir: Path,
    status_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    failed_models: List[str] = []
    data_hashes: set[str] = set()
    prereg_paths: set[str] = set()
    for status in status_rows:
        model_slug = status["model"]
        if status["status"] != "ok":
            failed_models.append(model_slug)
            summary_rows.append({
                "model": model_slug,
                "status": "failed",
                "verdict": "run-failed",
                "data_hash_sha256": "",
                "eligible_coverage": None,
                "undetermined_coverage": None,
                "max_significant_coverage": None,
                "c_accuracy": None,
                "c_defined_coverage": None,
                "anchor_agreement": None,
                "anchor_passed": None,
                "prompt_strategy_name": "",
                "summary_json_path": "",
            })
            continue

        summary_path = Path(status["summary_json_path"])
        payload = _load_json(summary_path)
        data_hashes.add(str(payload.get("data_hash_sha256")))
        prereg_paths.add(str(payload.get("prereg_path")))
        anchor = payload.get("anchor") or {}
        aux_c = payload.get("auxiliary_C") or {}
        summary_rows.append({
            "model": model_slug,
            "status": "ok",
            "verdict": payload["verdict"],
            "data_hash_sha256": payload["data_hash_sha256"],
            "eligible_coverage": payload.get("eligible_coverage"),
            "undetermined_coverage": payload.get("undetermined_coverage"),
            "max_significant_coverage": payload["max_significant_coverage"],
            "c_accuracy": aux_c.get("accuracy"),
            "c_defined_coverage": aux_c.get("coverage_defined"),
            "anchor_agreement": anchor.get("agreement"),
            "anchor_passed": anchor.get("passed"),
            "prompt_strategy_name": payload.get("prompt_strategy_name", ""),
            "summary_json_path": str(summary_path),
        })

    if len(data_hashes) > 1:
        raise RuntimeError(f"panel summary saw multiple data hashes: {sorted(data_hashes)}")
    if len(prereg_paths) > 1:
        raise RuntimeError(f"panel summary saw multiple prereg paths: {sorted(prereg_paths)}")

    csv_path = out_dir / "panel_summary.csv"
    json_path = out_dir / "panel_summary.json"
    fields = [
        "model",
        "status",
        "verdict",
        "data_hash_sha256",
        "eligible_coverage",
        "undetermined_coverage",
        "max_significant_coverage",
        "c_accuracy",
        "c_defined_coverage",
        "anchor_agreement",
        "anchor_passed",
        "prompt_strategy_name",
        "summary_json_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    meta = {
        "complete": len(failed_models) == 0,
        "n_models": len(summary_rows),
        "failed_models": failed_models,
        "data_hash_sha256": next(iter(data_hashes), ""),
        "prereg_path": next(iter(prereg_paths), ""),
        "rows": summary_rows,
    }
    _write_json(json_path, _sanitize_for_json(meta))
    return meta


def run_canary(args: argparse.Namespace) -> int:
    prereg_path = Path(args.prereg).expanduser().resolve()
    if not prereg_path.exists():
        raise SystemExit(f"prereg doc not found: {prereg_path}")
    data_path = Path(args.data).expanduser().resolve()
    prompts, labels, data_hash_sha256 = _load_calibration_jsonl(str(data_path))
    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, output_projection, _layer_indices = pipeline.load_model(args.model, cfg)
    prompt_identity = prompt_identity_bundle(args.model, tokenizer)
    prompt_strategy = io_plugins.get_prompt_strategy(args.model)
    prompt_strategy_name = prompt_identity["prompt_strategy_name"]
    tokenizer_fix_flags = pipeline.tokenizer_config_for_model(args.model)
    token_buckets = literal_yes_no_token_buckets(tokenizer, int(output_projection.vocab_size))
    token_buckets.update(semantic_shortlist_token_buckets(tokenizer, int(output_projection.vocab_size)))
    token_buckets.update(control_marker_token_bundle(tokenizer, int(output_projection.vocab_size)))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    canary_path = canary_path_for(out_dir, args.model)
    spec_path = spec_path_for(out_dir, args.model)

    sample_payloads: List[Dict[str, Any]] = []
    n_rows = min(int(args.limit), len(prompts))
    for i in range(n_rows):
        wrapped = prompt_strategy(prompts[i], tokenizer)
        prefix = pipeline.prefix_readout(model, tokenizer, wrapped)
        scores = _score_row_from_probs(
            prefix["last_probs"],
            token_buckets["yes_token_ids"],
            token_buckets["no_token_ids"],
            token_buckets["semantic_yes_token_ids"],
            token_buckets["semantic_no_token_ids"],
            token_buckets["control_token_ids"],
        )
        sample_payloads.append({
            "sample_idx": i,
            "label_B": int(labels[i]),
            "gold_yes_no": _gold_yes_no(int(labels[i])),
            "p_yes": scores["p_yes"],
            "p_no": scores["p_no"],
            "decidedness": scores["decidedness"],
            "semantic_decidedness": scores["semantic_decidedness"],
            "semantic_offliteral_mass": scores["semantic_offliteral_mass"],
            "control_mass": scores["control_mass"],
            "decidedness_floor": scores["decidedness_floor"],
            "above_floor": scores["above_floor"],
            "lean": scores["lean"],
            "top10": _top_token_records(tokenizer, prefix["last_probs"], k=10),
        })

    payload = {
        "schema_version": "belief_canary_v1",
        "model": args.model,
        "data_hash_sha256": data_hash_sha256,
        "prompt_strategy_name": prompt_strategy_name,
        "prompt_strategy_source_sha256": prompt_identity["prompt_strategy_source_sha256"],
        "prompt_probe_input_sha256": prompt_identity["prompt_probe_input_sha256"],
        "prompt_probe_output_sha256": prompt_identity["prompt_probe_output_sha256"],
        "tokenizer_fix_flags": tokenizer_fix_flags,
        "prereg_path": _prefix_summary_path(prereg_path),
        "literal_yes_tokens": token_buckets["yes_tokens"],
        "literal_no_tokens": token_buckets["no_tokens"],
        "literal_yes_token_ids": token_buckets["yes_token_ids"],
        "literal_no_token_ids": token_buckets["no_token_ids"],
        "semantic_affirmative_forms": sorted(SEMANTIC_AFFIRMATIVE),
        "semantic_negative_forms": sorted(SEMANTIC_NEGATIVE),
        "semantic_yes_tokens": token_buckets["semantic_yes_tokens"],
        "semantic_no_tokens": token_buckets["semantic_no_tokens"],
        "control_tokens": token_buckets["control_tokens"],
        "control_floor_multiplier": CONTROL_FLOOR_MULTIPLIER,
        "samples": sample_payloads,
        "out_of_scope": OUT_OF_SCOPE_LINE,
    }
    _write_json(canary_path, _sanitize_for_json(payload))
    spec = _make_spec(
        model_slug=args.model,
        data_hash_sha256=data_hash_sha256,
        prompt_identity=prompt_identity,
        tokenizer_fix_flags=tokenizer_fix_flags,
        token_buckets=token_buckets,
        canary_path=canary_path,
        prereg_path=prereg_path,
    )
    _write_frozen_spec(spec_path, spec)
    print(f"[belief-canary] wrote {canary_path}")
    print(f"[belief-canary] wrote {spec_path}")
    for sample in sample_payloads:
        print(
            f"[belief-canary] sample={sample['sample_idx']} label_B={sample['label_B']} "
            f"p_yes={sample['p_yes']:.4f} p_no={sample['p_no']:.4f} "
            f"decidedness={sample['decidedness']:.4f} "
            f"semantic={sample['semantic_decidedness']:.4f} "
            f"control={sample['control_mass']:.4f}"
        )
    return 0


def run_score(args: argparse.Namespace) -> int:
    prereg_path = Path(args.prereg).expanduser().resolve()
    if not prereg_path.exists():
        raise SystemExit(f"prereg doc not found: {prereg_path}")
    data_path = Path(args.data).expanduser().resolve()
    prompts, labels, data_hash_sha256 = _load_calibration_jsonl(str(data_path))
    out_dir = Path(args.out_dir).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve() if args.spec else spec_path_for(out_dir, args.model)
    if not spec_path.exists():
        raise SystemExit(f"frozen spec not found: {spec_path}")

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, output_projection, layer_indices = pipeline.load_model(args.model, cfg)
    prompt_identity = prompt_identity_bundle(args.model, tokenizer)
    prompt_strategy_name = prompt_identity["prompt_strategy_name"]
    tokenizer_fix_flags = pipeline.tokenizer_config_for_model(args.model)
    spec = _load_json(spec_path)
    validated = _validate_spec(
        spec,
        model_slug=args.model,
        data_hash_sha256=data_hash_sha256,
        prompt_identity=prompt_identity,
        tokenizer_fix_flags=tokenizer_fix_flags,
        tokenizer=tokenizer,
    )

    rows = _collect_sample_rows(
        model=model,
        tokenizer=tokenizer,
        output_projection=output_projection,
        layer_indices=layer_indices,
        prompts=prompts,
        labels=labels,
        model_slug=args.model,
        prompt_strategy_name=prompt_strategy_name,
        yes_token_ids=validated["yes_token_ids"],
        no_token_ids=validated["no_token_ids"],
        semantic_yes_token_ids=validated["semantic_yes_token_ids"],
        semantic_no_token_ids=validated["semantic_no_token_ids"],
        control_token_ids=validated["control_token_ids"],
        limit=int(args.limit),
        include_anchor=(args.model == MISTRAL_NEMO_SLUG),
        anchor_max_new_tokens=int(args.anchor_max_new_tokens),
    )
    labels_arr = np.asarray([int(row["label_B"]) for row in rows], dtype=np.int32)
    lean_arr = np.asarray([float(row["lean"]) for row in rows], dtype=np.float64)
    decidedness_arr = np.asarray([float(row["decidedness"]) for row in rows], dtype=np.float64)
    eligible_mask = np.asarray([bool(row["above_floor"]) for row in rows], dtype=bool)
    curve = build_coverage_curve(
        labels=labels_arr,
        lean_scores=lean_arr,
        decidedness=decidedness_arr,
        eligible_mask=eligible_mask,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.bootstrap_seed),
    )
    summary = _score_summary(
        model_slug=args.model,
        data_hash_sha256=data_hash_sha256,
        prompt_strategy_name=prompt_strategy_name,
        spec_path=spec_path,
        prereg_path=prereg_path,
        rows=rows,
        curve=curve,
    )

    csv_path = readout_csv_path_for(out_dir, args.model)
    json_path = readout_json_path_for(out_dir, args.model)
    _write_rows_csv(csv_path, rows)
    _write_json(json_path, _sanitize_for_json(summary))
    print(f"[belief-score] wrote {csv_path}")
    print(f"[belief-score] wrote {json_path}")
    print(
        f"[belief-score] verdict={summary['verdict']} "
        f"eligible_coverage={summary['eligible_coverage']:.3f} "
        f"undetermined_coverage={summary['undetermined_coverage']:.3f} "
        f"max_significant_coverage={summary['max_significant_coverage']:.3f}"
    )
    anchor = summary.get("anchor")
    if anchor is not None:
        print(
            f"[belief-score] anchor agreement={anchor['agreement']:.4f} "
            f"passed={anchor['passed']}"
        )
        if not anchor["passed"]:
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="literal t=0 belief readout for ANLI step-0")
    sub = p.add_subparsers(dest="cmd", required=True)

    canary = sub.add_parser("canary", help="write a 3-sample canary + frozen literal token spec")
    canary.add_argument("--model", required=True)
    canary.add_argument("--data", default=str(DEFAULT_DATA))
    canary.add_argument("--out-dir", required=True)
    canary.add_argument("--prereg", required=True)
    canary.add_argument("--limit", type=int, default=3)
    canary.set_defaults(func=run_canary)

    score = sub.add_parser("score", help="score the labeled panel with a frozen literal token spec")
    score.add_argument("--model", required=True)
    score.add_argument("--data", default=str(DEFAULT_DATA))
    score.add_argument("--out-dir", required=True)
    score.add_argument("--spec", default="")
    score.add_argument("--prereg", required=True)
    score.add_argument("--limit", type=int, default=0)
    score.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_N)
    score.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    score.add_argument("--anchor-max-new-tokens", type=int, default=6)
    score.set_defaults(func=run_score)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
