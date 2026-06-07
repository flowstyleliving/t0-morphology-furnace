#!/usr/bin/env python3
"""RAUQ at the commit step — honest prior-art baseline for the t0 attention panel.

RAUQ (Recurrent Attention-based Uncertainty Quantification) selects, per
decoder layer, the head that most strongly attends to the immediately
preceding token (argmax_h mean_i a_{i,i-1}), propagates a recurrent
uncertainty signal along the sequence, and takes the max over layers.

This scores RAUQ at *our* setting (commit step = gen_step 1, ANLI R1, the
same 9-model panel) in TWO variants so the methodological fork is a reported
result, not a hidden choice:

  1a  commit-only       score = 1 - g_commit         (recurrence stripped:
                                                       RAUQ has only one
                                                       generated token at the
                                                       commit step)
  1b  prompt-recurrence  u_t = α(1-g_t) + (1-α)u_{t-1} run over the PROMPT
                          tokens, read out at the commit token (keeps RAUQ's
                          defining recurrent mechanism alive over the prefix)

g_t is the selected head's attention from position t to position t-1
(sub-diagonal a_{t,t-1}); g_commit is the commit token's attention to the
last prompt token.

Pinned decisions (see wiki/results/rauq-sinkprobe-vs-ours-2026-05-16.md):
  · head-select: full-n UNSUPERVISED argmax_h mean_i a_{i,i-1} (no label
    leak; mild in-sample; matches RAUQ's own small-unlabeled-set selection)
  · scoring:     RAUQ-native FIXED direction is primary; the sign-free
    max(a, 1-a) AUROC is reported only as a sensitivity footnote
  · aggregate:   max over the 3 PANEL layers (final / mid / last_minus_1) —
    NOT full-depth RAUQ; the 3-layer restriction is a noted limitation
  · α = 0.5, pinned, untuned (tuning on labels would leak; unsupervised
    tuning has no signal)

Observational guarantee: the capture wrapper returns attn(x, mask, cache)
UNMODIFIED. Its q/k/rope/scale/mask/softmax math is byte-identical to
diagnose_inter_head_disagreement._capture_last_query_weights (shared helpers
imported + a copied weights core kept in sync — see the NOTE in
_capture_subdiag_and_lastq). Run --invariance-check before trusting any
AUROC: requires N/N byte-identical gen_token_ids vs the unwrapped path.

Usage:
  # parity gate (hard prerequisite — mirrors Step 1 Phase 1)
  .venv/bin/python scripts/rauq_at_commit.py \\
      --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
      --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
      --invariance-check 10

  # full RAUQ scoring for one model
  .venv/bin/python scripts/rauq_at_commit.py \\
      --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
      --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
      --out experiments/t0-baselines/2026-05-16/run-01/rauq/Mistral-7B-Instruct-v0.3-4bit.rauq.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _hash_file, _load_calibration_jsonl
from scripts.diagnose_inter_head_disagreement import (
    _apply_attention_mask,
    _find_layers,
    _project_queries_keys,
    _raw_auroc,
    _target_layer_map,
)

ALPHA = 0.5  # RAUQ recurrence weight; pinned + untuned (see module docstring).
PANEL_LAYERS = ("final", "mid", "last_minus_1")
DEFAULT_MIN_USABLE_FRACTION = 0.95
SCHEMA = "rauq_at_commit/v1"
DIAGNOSE_PATH = REPO_ROOT / "scripts" / "diagnose_inter_head_disagreement.py"
PIPELINE_PATH = REPO_ROOT / "pri_v2_mlx_pipeline.py"


# ─────────────────────────────────────────────────────────────────────────────
# Capture wrapper — same attention math as the inter-head wrapper, but keeps
# the sub-diagonal a_{i,i-1} (RAUQ head-select + 1b recurrence need it; the
# last-query slice the inter-head wrapper keeps cannot provide it).
# ─────────────────────────────────────────────────────────────────────────────


def _capture_subdiag_and_lastq(
    attn: Any, x: mx.array, mask: Optional[Any], cache: Optional[Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (last_query_row (H, T_k), sub_diagonal (H, T_q-1)).

    NOTE: the weights computation below MUST stay byte-identical to
    diagnose_inter_head_disagreement._capture_last_query_weights (projection
    via the shared _project_queries_keys helper, rope, GQA repeat, fp32 cast
    BEFORE the matmul, attn.scale, shared _apply_attention_mask, precise
    fp32 softmax). If that function changes, mirror the change here and
    re-run --invariance-check.
    """
    if cache is not None:
        raise RuntimeError(
            "attention capture expects cache=None; trace path changed unexpectedly"
        )
    batch, _seqlen, _ = x.shape
    if batch != 1:
        raise RuntimeError(f"expected batch size 1 during trace, got {batch}")
    queries, keys = _project_queries_keys(attn, x)
    queries = attn.rope(queries)
    keys = attn.rope(keys)
    n_repeats = int(attn.n_heads) // int(attn.n_kv_heads)
    if n_repeats > 1:
        keys = mx.repeat(keys, n_repeats, axis=1)
    queries = queries.astype(mx.float32)
    keys = keys.astype(mx.float32)
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * attn.scale
    scores = _apply_attention_mask(scores, mask)
    weights = mx.softmax(scores, axis=-1, precise=True).astype(mx.float32)
    mx.eval(weights)
    w = np.array(weights)[0]  # (H, T_q, T_k)
    last_q = w[:, -1, :]  # (H, T_k)
    t_q = w.shape[1]
    if t_q >= 2:
        # a_{i, i-1} for i = 1 .. T_q-1  → (H, T_q-1). Square attention in the
        # no-KV-cache trace path (T_k == T_q), so key index i-1 is in range.
        subdiag = w[:, np.arange(1, t_q), np.arange(0, t_q - 1)]
    else:
        subdiag = np.empty((w.shape[0], 0), dtype=w.dtype)
    return np.ascontiguousarray(last_q), np.ascontiguousarray(subdiag)


class _RauqWrap:
    """Module-shaped proxy: captures sub-diag + last-query, returns the
    ORIGINAL attention forward unmodified (observational)."""

    def __init__(
        self,
        orig: Any,
        lastq_list: List[np.ndarray],
        subdiag_list: List[np.ndarray],
    ) -> None:
        self._orig = orig
        self._lastq = lastq_list
        self._subdiag = subdiag_list

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)

    def __call__(
        self, x: mx.array, mask: Optional[Any] = None, cache: Optional[Any] = None
    ) -> mx.array:
        attn = self._orig
        lq, sd = _capture_subdiag_and_lastq(attn, x, mask, cache)
        self._lastq.append(lq)
        self._subdiag.append(sd)
        return attn(x, mask, cache)


@contextmanager
def rauq_attention_capture(layers: List[Any], target_indices: Dict[str, int]):
    """Wrap target layers' self_attn for the duration of the block.

    Yields (lastq_caps, subdiag_caps): tag -> list of per-forward arrays.
    """
    lastq: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    subdiag: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    originals: Dict[str, Any] = {}
    try:
        for tag, idx in target_indices.items():
            layer = layers[idx]
            originals[tag] = layer.self_attn
            layer.self_attn = _RauqWrap(layer.self_attn, lastq[tag], subdiag[tag])
        yield lastq, subdiag
    finally:
        for tag, idx in target_indices.items():
            layers[idx].self_attn = originals[tag]


# ─────────────────────────────────────────────────────────────────────────────
# RAUQ scoring (pure functions — unit-tested without a model load)
# ─────────────────────────────────────────────────────────────────────────────


def select_heads(prefix_subdiags: List[np.ndarray]) -> Tuple[int, float]:
    """Full-n unsupervised head-select: argmax_h of the grand mean of
    a_{i,i-1} pooled over every position of every sample (labels untouched).

    `prefix_subdiags`: list of (H, T_i-1) arrays, one per usable sample.
    Returns (selected_head_index, selected_head_mean). Pools positions
    across all samples (longer prompts contribute more positions — this
    matches RAUQ's mean-over-tokens definition rather than a per-sample
    mean-of-means).
    """
    if not prefix_subdiags:
        raise ValueError("no prefix sub-diagonals to select a head from")
    n_heads = prefix_subdiags[0].shape[0]
    sums = np.zeros(n_heads, dtype=np.float64)
    count = 0
    for sd in prefix_subdiags:
        if sd.shape[0] != n_heads or sd.shape[1] == 0:
            continue
        sums += sd.astype(np.float64).sum(axis=1)
        count += sd.shape[1]
    if count == 0:
        raise ValueError("all prefix sub-diagonals were empty")
    means = sums / count
    h = int(np.argmax(means))
    return h, float(means[h])


def score_1a(commit_g_head: float) -> float:
    """Commit-only RAUQ: 1 - g_commit (higher = more uncertain)."""
    if not math.isfinite(commit_g_head):
        return float("nan")
    return 1.0 - commit_g_head


def score_1b(prefix_g_seq: np.ndarray, commit_g_head: float, alpha: float = ALPHA) -> float:
    """Prompt-recurrence RAUQ read out at the commit token.

    u_1     = 1 - g_1
    u_t     = α(1-g_t) + (1-α)u_{t-1}     for t = 2 .. T_prefix-1   (prompt)
    u_commit = α(1-g_commit) + (1-α)u_{T_prefix-1}                  (commit step)

    `prefix_g_seq` is the selected head's sub-diagonal over the prompt
    (length T_prefix-1). Returns u_commit (higher = more uncertain).
    """
    g = np.asarray(prefix_g_seq, dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size == 0 or not math.isfinite(commit_g_head):
        return float("nan")
    u = 1.0 - g[0]
    for t in range(1, g.size):
        u = alpha * (1.0 - g[t]) + (1.0 - alpha) * u
    u = alpha * (1.0 - commit_g_head) + (1.0 - alpha) * u
    return float(u)


def _auroc_pair(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Raw (fixed-direction) AUROC + a sign-free sensitivity footnote."""
    a, direction = _raw_auroc(y, np.asarray(scores, dtype=np.float64))
    signfree = float("nan")
    if math.isfinite(a):
        signfree = max(a, 1.0 - a)
    return {
        "auroc": None if not math.isfinite(a) else round(float(a), 6),
        "direction": direction,
        "auroc_signfree": None if not math.isfinite(signfree) else round(signfree, 6),
        "n_scored": int(np.isfinite(scores).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model trace + scoring
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(model_id: str):
    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, projection, layer_indices = pipeline.load_model(model_id, cfg)
    return model, tokenizer, projection, layer_indices


def _trace_features(
    model_id: str, data_path: str, *, limit: int = 0, max_new_tokens: int = 4
) -> Dict[str, Any]:
    """One forward pass per sample; collect, per panel layer per sample:
    the prefix sub-diagonal (all heads) and the commit-token attention to
    the previous token (all heads). Returns everything needed to head-select
    then score both variants.
    """
    prompts, labels, data_hash = _load_calibration_jsonl(data_path)
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    model, tokenizer, projection, layer_indices = _load_model(model_id)
    layers = _find_layers(model)
    target_map = _target_layer_map(len(layers))
    tags = [t for t in PANEL_LAYERS if t in target_map]
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)

    print(f"[rauq] model={model_id}")
    print(f"[rauq] n_layers={len(layers)}  target indices={target_map}")
    print(f"[rauq] {len(prompts)} samples")

    per_sample: List[Dict[str, Any]] = []
    n_trace_failed = 0
    n_prefix_only_eos = 0
    for i, prompt in enumerate(prompts):
        wrapped = prompt_strategy(prompt, tokenizer)
        try:
            with rauq_attention_capture(layers, target_map) as (lastq, subdiag):
                trace = pipeline.trace_sample(
                    model=model, tokenizer=tokenizer, prompt=wrapped,
                    layer_indices=layer_indices, output_projection=projection,
                    max_new_tokens=max_new_tokens, v3_capture=False,
                )
        except Exception as e:  # noqa: BLE001 — mirror diagnose's tolerance
            n_trace_failed += 1
            print(f"[rauq]   sample {i}: trace FAILED ({e})")
            continue

        gen_token_ids = trace.get("gen_token_ids") or []
        expected_calls = 1 + len(gen_token_ids)
        for tag in tags:
            if len(lastq[tag]) != expected_calls or len(subdiag[tag]) != expected_calls:
                raise RuntimeError(
                    f"capture count mismatch at sample {i}, tag={tag}: "
                    f"expected {expected_calls}, got lastq={len(lastq[tag])} "
                    f"subdiag={len(subdiag[tag])}"
                )
        if expected_calls < 2:
            n_prefix_only_eos += 1
            continue

        rec: Dict[str, Any] = {"sample_idx": i, "label": int(labels[i]), "layers": {}}
        for tag in tags:
            prefix_sd = subdiag[tag][0]  # (H, T_prefix-1) — prompt forward
            commit_lq = lastq[tag][1]    # (H, T_k) — first generation forward
            # commit token's attention to the immediately-preceding (last
            # prompt) token = a_{commit, commit-1}: key index -2.
            commit_g = (
                commit_lq[:, -2].astype(np.float64)
                if commit_lq.shape[1] >= 2
                else np.full(commit_lq.shape[0], np.nan)
            )
            rec["layers"][tag] = {
                "prefix_subdiag": prefix_sd.astype(np.float32),
                "commit_g": commit_g.astype(np.float64),
            }
        per_sample.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(prompts):
            print(f"[rauq]   {i+1}/{len(prompts)}")

    return {
        "per_sample": per_sample,
        "tags": tags,
        "n_total": len(prompts),
        "n_usable": len(per_sample),
        "n_trace_failed": n_trace_failed,
        "n_prefix_only_eos": n_prefix_only_eos,
        "data_hash": data_hash,
    }


def _score_model(traced: Dict[str, Any]) -> Dict[str, Any]:
    """Head-select per layer (unsupervised), then score 1a + 1b per layer
    and the max-over-panel-layers aggregate."""
    per_sample = traced["per_sample"]
    tags: List[str] = traced["tags"]
    y = np.array([r["label"] for r in per_sample], dtype=np.float64)

    selected: Dict[str, int] = {}
    selected_mean: Dict[str, float] = {}
    for tag in tags:
        h, m = select_heads([r["layers"][tag]["prefix_subdiag"] for r in per_sample])
        selected[tag] = h
        selected_mean[tag] = m

    s1a = {tag: np.full(len(per_sample), np.nan) for tag in tags}
    s1b = {tag: np.full(len(per_sample), np.nan) for tag in tags}
    for j, r in enumerate(per_sample):
        for tag in tags:
            h = selected[tag]
            commit_g_h = float(r["layers"][tag]["commit_g"][h])
            prefix_seq = r["layers"][tag]["prefix_subdiag"][h]
            s1a[tag][j] = score_1a(commit_g_h)
            s1b[tag][j] = score_1b(prefix_seq, commit_g_h)

    def _panel(scores_by_tag: Dict[str, np.ndarray]) -> Dict[str, Any]:
        per_layer = {tag: _auroc_pair(y, scores_by_tag[tag]) for tag in tags}
        stacked = np.vstack([scores_by_tag[tag] for tag in tags])  # (L, N)
        agg = np.nanmax(stacked, axis=0)
        agg[~np.isfinite(stacked).any(axis=0)] = np.nan
        return {"per_layer": per_layer, "aggregate_max": _auroc_pair(y, agg)}

    return {
        "selected_heads": selected,
        "selected_head_mean_subdiag": {k: round(v, 6) for k, v in selected_mean.items()},
        "results": {
            "1a_commit_only": _panel(s1a),
            "1b_prompt_recurrence": _panel(s1b),
        },
    }


def _run_id_from_out(out_path: Optional[Path]) -> str:
    if out_path is not None:
        for part in reversed(out_path.parts):
            if part.startswith("run-"):
                return part
    return "run-" + uuid.uuid4().hex[:8]


def _atomic_write_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=out_path.parent, prefix=f".{out_path.stem}.",
            suffix=".json.tmp", delete=False,
        ) as f:
            tmp = Path(f.name)
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, out_path)
    except OSError as exc:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise SystemExit(f"failed to write {out_path}: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Invariance probe (parity gate for the RAUQ wrapper specifically)
# ─────────────────────────────────────────────────────────────────────────────


def _invariance_check(model_id: str, data_path: str, n: int, max_new_tokens: int) -> int:
    model, tokenizer, projection, layer_indices = _load_model(model_id)
    layers = _find_layers(model)
    target_map = _target_layer_map(len(layers))
    prompts, _, _ = _load_calibration_jsonl(data_path)
    prompts = prompts[:n]
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)

    print(f"[rauq-invariance] model={model_id}")
    print(f"[rauq-invariance] n_layers={len(layers)}  target_map={target_map}")
    print(f"[rauq-invariance] {len(prompts)} samples × max_new_tokens={max_new_tokens}")
    n_match = n_diff = 0
    for i, prompt in enumerate(prompts):
        wp = prompt_strategy(prompt, tokenizer)
        with rauq_attention_capture(layers, target_map):
            tw = pipeline.trace_sample(
                model=model, tokenizer=tokenizer, prompt=wp,
                layer_indices=layer_indices, output_projection=projection,
                max_new_tokens=max_new_tokens, v3_capture=False,
            )
        ids_w = list(tw.get("gen_token_ids") or [])
        tu = pipeline.trace_sample(
            model=model, tokenizer=tokenizer, prompt=wp,
            layer_indices=layer_indices, output_projection=projection,
            max_new_tokens=max_new_tokens, v3_capture=False,
        )
        ids_u = list(tu.get("gen_token_ids") or [])
        if ids_w == ids_u:
            n_match += 1
            print(f"  sample {i}: MATCH  ids={ids_w}")
        else:
            n_diff += 1
            print(f"  sample {i}: DIFF   wrapped={ids_w}  unwrapped={ids_u}")
    print("=" * 60)
    print(f"[rauq-invariance] result: {n_match}/{len(prompts)} match, {n_diff} differ")
    print("=" * 60)
    return 0 if n_diff == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="RAUQ at the commit step (1a + 1b)")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--limit", type=int, default=0, help="cap n samples (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument(
        "--min-usable-fraction", type=float, default=DEFAULT_MIN_USABLE_FRACTION
    )
    p.add_argument(
        "--invariance-check",
        type=int,
        default=0,
        metavar="N",
        help="run the wrapped-vs-unwrapped parity gate on N samples and exit",
    )
    args = p.parse_args()

    if args.invariance_check > 0:
        return _invariance_check(
            args.model, args.data, args.invariance_check, args.max_new_tokens
        )

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    if out_path is not None and out_path.exists() and out_path.is_dir():
        raise SystemExit(f"--out is a directory, expected a file: {out_path}")

    # Capture at run START, before the (multi-minute per-model) trace — not
    # at payload-assembly, which would record completion time. Provenance
    # field must reflect when the run began. (Greptile PR #14 round-2 finding.)
    run_started_at = datetime.now(timezone.utc).isoformat()
    traced = _trace_features(
        args.model, args.data, limit=args.limit, max_new_tokens=args.max_new_tokens
    )
    n_total, n_usable = traced["n_total"], traced["n_usable"]
    if n_usable == 0:
        raise SystemExit("no usable samples")
    required = min(n_total, max(1, math.ceil(n_total * args.min_usable_fraction)))
    if n_usable < required:
        raise SystemExit(
            f"usable coverage too low: {n_usable}/{n_total} "
            f"(trace_failed={traced['n_trace_failed']}, "
            f"prefix_only_eos={traced['n_prefix_only_eos']}); need ≥ {required}"
        )

    scored = _score_model(traced)
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "model": {"slug": args.model},
        "data": {"path": args.data, "data_hash_sha256": traced["data_hash"]},
        "provenance": {
            "script_hash_sha256": _hash_file(Path(__file__).resolve()),
            "diagnose_helpers_hash_sha256": _hash_file(DIAGNOSE_PATH),
            "pipeline_module_hash_sha256": _hash_file(PIPELINE_PATH),
            "started_at_iso": run_started_at,
            "run_id": _run_id_from_out(out_path),
            "host": socket.gethostname(),
        },
        "config": {
            "alpha": ALPHA,
            "head_select": "full_n_unsupervised_argmax_h_mean_subdiag",
            "aggregate": "max_over_panel_layers",
            "panel_layers": list(traced["tags"]),
            "direction": "fixed_native_primary_signfree_footnote",
            "max_new_tokens": args.max_new_tokens,
        },
        "coverage": {
            "n_total": n_total,
            "n_usable": n_usable,
            "n_trace_failed": traced["n_trace_failed"],
            "n_prefix_only_eos": traced["n_prefix_only_eos"],
        },
        "selected_heads": scored["selected_heads"],
        "selected_head_mean_subdiag": scored["selected_head_mean_subdiag"],
        "results": scored["results"],
    }

    if out_path is not None:
        _atomic_write_json(out_path, payload)
        print(f"[rauq] wrote {out_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    # ── Console summary ──
    print()
    print("=" * 78)
    print(f"  RAUQ-at-commit summary — {args.model}  (n_usable={n_usable}/{n_total})")
    print("=" * 78)
    print(f"  selected heads: {scored['selected_heads']}")
    for variant in ("1a_commit_only", "1b_prompt_recurrence"):
        res = scored["results"][variant]
        print(f"  [{variant}]")
        for tag, ap in res["per_layer"].items():
            print(
                f"    {tag:<13s} AUROC={ap['auroc']}  dir={ap['direction']}  "
                f"signfree={ap['auroc_signfree']}"
            )
        agg = res["aggregate_max"]
        print(
            f"    {'AGG(max)':<13s} AUROC={agg['auroc']}  dir={agg['direction']}  "
            f"signfree={agg['auroc_signfree']}"
        )
    print()
    print("  Fixed-direction AUROC is primary; signfree = max(a, 1-a) is a")
    print("  sensitivity footnote only. Aggregate = max over the 3 panel layers")
    print("  (NOT full-depth RAUQ — see wiki writeup for the limitation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
