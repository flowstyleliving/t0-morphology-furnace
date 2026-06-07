#!/usr/bin/env python3
"""SinkProbe sink-score baseline — honest prior-art baseline for the t0 panel.

SinkProbe's core feature is the *sink score* of key position i for head h at
layer l: the attention mass position i RECEIVES, averaged over the causal
queries that can see it:

    s_i^{l,h} = (1 / (T - i)) · Σ_{u=i}^{T-1} A_{u,i}^{l,h}

This is a CAUSAL COLUMN-SUM over all query rows — fundamentally different from
the calibrator's existing Step-1.5 `v_norm_*` cells, which weight only the
LAST-query row. SinkProbe's headline classifier additionally observes that
sinks with large value-vector norms dominate the attention output, so we also
emit the ‖V‖-weighted variant  sv_i = s_i · ‖V_i‖.

Scored at our setting (commit step = gen_step 1, ANLI R1, the 10-model panel)
as single-cell AUROCs in the same shape as our calibrator cells (so they slot
straight into the Step 2.3 head-to-head), NOT SinkProbe's full logistic probe.

Per (layer), reduced over heads (k = 4, pinned + untuned — tuning on labels
would leak, same discipline as RAUQ's α):
  · sink_bos        = mean_h  s_0^h            (the canonical BOS/pos-0 sink)
  · sink_top1       = mean_h  max_i s_i^h
  · sink_topk_sum   = mean_h  Σ(top-4 s_i^h)
  · sink_bos_vw / sink_top1_vw / sink_topk_sum_vw  — same, on sv_i = s_i·‖V_i‖

Pinned decisions (see wiki/results/rauq-sinkprobe-vs-ours-2026-05-16.md):
  · forward scope: PREFIX forward (the prompt's causal attention at the
    commit layer). SinkProbe is not recurrent → no 1a/1b-style dual variant.
  · NO head-selection (aggregated over heads) → no train-fold / no
    selection-bias question.
  · k = 4, pinned + untuned.
  · scoring: fixed-direction AUROC primary; sign-free max(a,1-a) footnote.
  · sample-inclusion identical to RAUQ/calibrator (a sample counts iff the
    model committed a token) so the n=200 set — and the Step 2.3 join — is
    byte-identical across baselines.

Observational guarantee: the capture wrapper returns attn(x, mask, cache)
UNMODIFIED. Its q/k/rope/scale/mask/precise-softmax + value-projection math
is byte-identical to diagnose_inter_head_disagreement (shared helpers
imported + a copied weights core kept in sync — see the NOTE in
_capture_sink_and_vnorms). Run --invariance-check before trusting any AUROC.

Usage:
  # parity gate (hard prerequisite)
  .venv/bin/python scripts/sinkprobe_baseline.py \\
      --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
      --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
      --invariance-check 10

  # full scoring for one model
  .venv/bin/python scripts/sinkprobe_baseline.py \\
      --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
      --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
      --out experiments/t0-baselines/2026-05-16/run-01/sinkprobe/Mistral-7B-Instruct-v0.3-4bit.sinkprobe.json
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
    _project_values,
    _raw_auroc,
    _target_layer_map,
)

TOP_K = 4  # pinned + untuned (tuning on labels would leak; see module docstring)
PANEL_LAYERS = ("final", "mid", "last_minus_1")
DEFAULT_MIN_USABLE_FRACTION = 0.95
SCHEMA = "sinkprobe_baseline/v1"
DIAGNOSE_PATH = REPO_ROOT / "scripts" / "diagnose_inter_head_disagreement.py"
PIPELINE_PATH = REPO_ROOT / "pri_v2_mlx_pipeline.py"


# ─────────────────────────────────────────────────────────────────────────────
# Capture wrapper — same attention + value math as the inter-head/RAUQ
# wrappers, but retains the causal COLUMN-SUM (received attention per key) plus
# per-position value norms. No existing wrapper keeps the full column-sum.
# ─────────────────────────────────────────────────────────────────────────────


def _capture_sink_and_vnorms(
    attn: Any, x: mx.array, mask: Optional[Any], cache: Optional[Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sink_scores (H, T), value_norms (n_kv_heads, T)).

    sink_scores[h, i] = (1/(T-i)) Σ_u A[h, u, i]  — average attention key i
    receives from the causal queries u ≥ i (masked entries are ~0 after the
    precise softmax, so the plain column-sum equals Σ_{u≥i}).

    NOTE: the weights computation MUST stay byte-identical to
    diagnose_inter_head_disagreement._capture_last_query_weights, and the
    value path to _capture_value_norms (fp32 before square). If either
    changes, mirror here and re-run --invariance-check.
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
    # Value norms — mirror _capture_value_norms exactly (fp32 before square).
    values = _project_values(attn, x)  # (1, n_kv_heads, T, head_dim)
    values_fp32 = values.astype(mx.float32)
    vnorms = mx.sqrt(mx.sum(values_fp32 * values_fp32, axis=-1))  # (1, n_kv, T)
    mx.eval(weights, vnorms)
    w = np.array(weights)[0]  # (H, T_q, T_k)
    t_q, t_k = w.shape[1], w.shape[2]
    colsum = w.sum(axis=1)  # (H, T_k) — Σ over query rows; causal → Σ_{u≥i}
    # denom_i = number of causal queries that can see key i = T_q - i.
    # Square attention in the no-KV-cache trace path (T_k == T_q).
    denom = (t_q - np.arange(t_k)).astype(np.float64)
    denom[denom < 1.0] = 1.0  # defensive; i ≤ T-1 ⇒ T-i ≥ 1 already
    sink = (colsum.astype(np.float64) / denom[None, :])  # (H, T_k)
    return np.ascontiguousarray(sink), np.ascontiguousarray(np.array(vnorms)[0])


class _SinkWrap:
    """Module-shaped proxy: captures sink scores + value norms, returns the
    ORIGINAL attention forward unmodified (observational)."""

    def __init__(
        self,
        orig: Any,
        sink_list: List[np.ndarray],
        vnorm_list: List[np.ndarray],
    ) -> None:
        self._orig = orig
        self._sink = sink_list
        self._vnorm = vnorm_list

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)

    def __call__(
        self, x: mx.array, mask: Optional[Any] = None, cache: Optional[Any] = None
    ) -> mx.array:
        attn = self._orig
        s, v = _capture_sink_and_vnorms(attn, x, mask, cache)
        self._sink.append(s)
        self._vnorm.append(v)
        return attn(x, mask, cache)


@contextmanager
def sink_capture(layers: List[Any], target_indices: Dict[str, int]):
    """Wrap target layers' self_attn for the block. Yields (sink_caps,
    vnorm_caps): tag -> list of per-forward arrays."""
    sink: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    vnorm: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    originals: Dict[str, Any] = {}
    try:
        for tag, idx in target_indices.items():
            layer = layers[idx]
            originals[tag] = layer.self_attn
            layer.self_attn = _SinkWrap(layer.self_attn, sink[tag], vnorm[tag])
        yield sink, vnorm
    finally:
        for tag, idx in target_indices.items():
            layers[idx].self_attn = originals[tag]


# ─────────────────────────────────────────────────────────────────────────────
# Pure metric reductions (unit-tested without a model load)
# ─────────────────────────────────────────────────────────────────────────────


def _topk_sum(row: np.ndarray, k: int) -> float:
    """Sum of the k largest entries (all of them if fewer than k)."""
    r = np.asarray(row, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    if r.size <= k:
        return float(r.sum())
    return float(np.partition(r, -k)[-k:].sum())


def sink_metrics(
    sink: np.ndarray,
    vnorms: np.ndarray,
    n_q: int,
    n_kv: int,
    k: int = TOP_K,
) -> Dict[str, float]:
    """Reduce one sample's prefix sink scores (+ value norms) to the 6
    per-layer SinkProbe cells. `sink` is (H_q, T); `vnorms` is (n_kv, T).

    The ‖V‖-weighted variant expands KV-group value norms to per-Q-head by
    repetition (each Q head sees its KV group's V), matching the calibrator's
    `_lastq_weighted_v_norm` GQA convention. If the head layout is
    inconsistent (n_q % n_kv != 0) the *_vw metrics are NaN but the
    attention-mass metrics still resolve.
    """
    out: Dict[str, float] = {}
    if sink.ndim != 2 or sink.shape[0] < 1 or sink.shape[1] < 1:
        return {m: float("nan") for m in (
            "sink_bos", "sink_top1", "sink_topk_sum",
            "sink_bos_vw", "sink_top1_vw", "sink_topk_sum_vw")}
    s = sink.astype(np.float64)
    out["sink_bos"] = float(np.mean(s[:, 0]))
    out["sink_top1"] = float(np.mean(np.max(s, axis=1)))
    out["sink_topk_sum"] = float(np.mean([_topk_sum(s[h], k) for h in range(s.shape[0])]))

    nan = float("nan")
    if (
        vnorms.ndim != 2
        or n_kv < 1
        or n_q != s.shape[0]
        or n_q % n_kv != 0
        or vnorms.shape[0] != n_kv
        or vnorms.shape[1] != s.shape[1]
    ):
        out["sink_bos_vw"] = out["sink_top1_vw"] = out["sink_topk_sum_vw"] = nan
        return out
    repeats = n_q // n_kv
    v_per_q = np.repeat(vnorms.astype(np.float64), repeats, axis=0)  # (n_q, T)
    sv = s * v_per_q  # (H_q, T)
    out["sink_bos_vw"] = float(np.mean(sv[:, 0]))
    out["sink_top1_vw"] = float(np.mean(np.max(sv, axis=1)))
    out["sink_topk_sum_vw"] = float(
        np.mean([_topk_sum(sv[h], k) for h in range(sv.shape[0])])
    )
    return out


METRIC_NAMES = (
    "sink_bos", "sink_top1", "sink_topk_sum",
    "sink_bos_vw", "sink_top1_vw", "sink_topk_sum_vw",
)


def _auroc_pair(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Raw (fixed-direction) AUROC + a sign-free sensitivity footnote.

    Mirrors rauq_at_commit._auroc_pair; kept local so the two baseline
    scripts stay independent (no cross-script import coupling)."""
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
    return pipeline.load_model(model_id, cfg)


def _trace_features(
    model_id: str, data_path: str, *, limit: int = 0, max_new_tokens: int = 4
) -> Dict[str, Any]:
    prompts, labels, data_hash = _load_calibration_jsonl(data_path)
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    model, tokenizer, projection, layer_indices = _load_model(model_id)
    layers = _find_layers(model)
    target_map = _target_layer_map(len(layers))
    tags = [t for t in PANEL_LAYERS if t in target_map]
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)

    print(f"[sinkprobe] model={model_id}")
    print(f"[sinkprobe] n_layers={len(layers)}  target indices={target_map}")
    print(f"[sinkprobe] {len(prompts)} samples")

    rows: List[Dict[str, Any]] = []
    n_trace_failed = 0
    n_prefix_only_eos = 0
    for i, prompt in enumerate(prompts):
        wrapped = prompt_strategy(prompt, tokenizer)
        try:
            with sink_capture(layers, target_map) as (sink_caps, vnorm_caps):
                trace = pipeline.trace_sample(
                    model=model, tokenizer=tokenizer, prompt=wrapped,
                    layer_indices=layer_indices, output_projection=projection,
                    max_new_tokens=max_new_tokens, v3_capture=False,
                )
        except Exception as e:  # noqa: BLE001 — mirror diagnose's tolerance
            n_trace_failed += 1
            print(f"[sinkprobe]   sample {i}: trace FAILED ({e})")
            continue

        gen_token_ids = trace.get("gen_token_ids") or []
        expected_calls = 1 + len(gen_token_ids)
        for tag in tags:
            if len(sink_caps[tag]) != expected_calls or len(vnorm_caps[tag]) != expected_calls:
                raise RuntimeError(
                    f"capture count mismatch at sample {i}, tag={tag}: "
                    f"expected {expected_calls}, got sink={len(sink_caps[tag])} "
                    f"vnorm={len(vnorm_caps[tag])}"
                )
        # Sample-inclusion identical to RAUQ/calibrator: count iff the model
        # committed a token (≥2 forward calls). Keeps the n=200 set — and the
        # Step 2.3 join — byte-identical across baselines.
        if expected_calls < 2:
            n_prefix_only_eos += 1
            continue

        rec: Dict[str, Any] = {"sample_idx": i, "label": int(labels[i]), "metrics": {}}
        for tag in tags:
            sink0 = sink_caps[tag][0]   # (H_q, T) — prefix forward
            vnorm0 = vnorm_caps[tag][0]  # (n_kv, T)
            attn_mod = layers[target_map[tag]].self_attn
            n_q = int(attn_mod.n_heads)
            n_kv = int(attn_mod.n_kv_heads)
            rec["metrics"][tag] = sink_metrics(sink0, vnorm0, n_q, n_kv)
        rows.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(prompts):
            print(f"[sinkprobe]   {i+1}/{len(prompts)}")

    return {
        "rows": rows,
        "tags": tags,
        "n_total": len(prompts),
        "n_usable": len(rows),
        "n_trace_failed": n_trace_failed,
        "n_prefix_only_eos": n_prefix_only_eos,
        "data_hash": data_hash,
    }


def _score_model(traced: Dict[str, Any]) -> Dict[str, Any]:
    rows = traced["rows"]
    tags: List[str] = traced["tags"]
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    results: Dict[str, Any] = {}
    for tag in tags:
        per_metric: Dict[str, Any] = {}
        for m in METRIC_NAMES:
            scores = np.array(
                [r["metrics"][tag].get(m, float("nan")) for r in rows],
                dtype=np.float64,
            )
            per_metric[m] = _auroc_pair(y, scores)
        results[tag] = per_metric
    return {"results": results}


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
# Invariance probe (parity gate for the SinkProbe wrapper specifically)
# ─────────────────────────────────────────────────────────────────────────────


def _invariance_check(model_id: str, data_path: str, n: int, max_new_tokens: int) -> int:
    model, tokenizer, projection, layer_indices = _load_model(model_id)
    layers = _find_layers(model)
    target_map = _target_layer_map(len(layers))
    prompts, _, _ = _load_calibration_jsonl(data_path)
    prompts = prompts[:n]
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)

    print(f"[sinkprobe-invariance] model={model_id}")
    print(f"[sinkprobe-invariance] n_layers={len(layers)}  target_map={target_map}")
    print(f"[sinkprobe-invariance] {len(prompts)} samples × max_new_tokens={max_new_tokens}")
    n_match = n_diff = 0
    for i, prompt in enumerate(prompts):
        wp = prompt_strategy(prompt, tokenizer)
        with sink_capture(layers, target_map):
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
    print(f"[sinkprobe-invariance] result: {n_match}/{len(prompts)} match, {n_diff} differ")
    print("=" * 60)
    return 0 if n_diff == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="SinkProbe sink-score baseline")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--limit", type=int, default=0, help="cap n samples (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument(
        "--min-usable-fraction", type=float, default=DEFAULT_MIN_USABLE_FRACTION
    )
    p.add_argument(
        "--invariance-check", type=int, default=0, metavar="N",
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

    # Capture at run START (before the multi-minute trace) — not at
    # payload-assembly, which would record completion time.
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
            "top_k": TOP_K,
            "forward_scope": "prefix",
            "head_select": "none_aggregated_over_heads",
            "variants": ["attention_mass", "v_norm_weighted"],
            "direction": "fixed_native_primary_signfree_footnote",
            "panel_layers": list(traced["tags"]),
            "max_new_tokens": args.max_new_tokens,
        },
        "coverage": {
            "n_total": n_total,
            "n_usable": n_usable,
            "n_trace_failed": traced["n_trace_failed"],
            "n_prefix_only_eos": traced["n_prefix_only_eos"],
        },
        "results": scored["results"],
    }

    if out_path is not None:
        _atomic_write_json(out_path, payload)
        print(f"[sinkprobe] wrote {out_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    # ── Console summary ──
    print()
    print("=" * 78)
    print(f"  SinkProbe summary — {args.model}  (n_usable={n_usable}/{n_total})")
    print("=" * 78)
    for tag in traced["tags"]:
        print(f"  [{tag}]")
        for m in METRIC_NAMES:
            ap = scored["results"][tag][m]
            print(
                f"    {m:<18s} AUROC={ap['auroc']}  dir={ap['direction']}  "
                f"signfree={ap['auroc_signfree']}"
            )
    print()
    print("  Fixed-direction AUROC is primary; signfree = max(a, 1-a) is a")
    print(f"  sensitivity footnote only. k={TOP_K} (pinned, untuned). Prefix-")
    print("  forward causal column-sum; *_vw = ‖V‖-weighted variant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
