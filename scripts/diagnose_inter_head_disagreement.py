#!/usr/bin/env python3
"""Inter-head attention disagreement diagnostic — W_u-free rupture candidate.

For each calibration sample, capture per-head attention weights at gen_step=1
across three target decoder blocks (final, mid = N//2, last-1 = N-2). Compute
the Jensen-Shannon radius across heads at each captured layer:

    js_radius_layer{tag} = (1/H) * Σ_h JS(A^h || A_centroid)

where A^h ∈ Δ^T is head h's attention distribution over past positions and
A_centroid = mean over heads. Bounded in [0, log 2].

  · high = heads disagree about what to attend to → hedge → rupture candidate
  · low  = heads converge on the same content → committed

Falsification gates (analyzed in the wiki roll-up, not this script):
  1. AUROC < 0.6 on Mistral → dead at the easy case
  2. AUROC ≈ 0.7 on Mistral but flips raw orientation or collapses on Qwen 2.5
     → ALSO an OOD detector dressed up differently
  3. AUROC ≈ 0.7 across Mistral / Qwen 2.5 / Llama 3B at the SAME raw
     orientation without recalibration → architectural invariant. Scale.

W_u-free, one forward pass. The capture wrapper mirrors each model family's
native attention math closely enough to compare the captured simplex geometry
across architectures without perturbing generation.

Usage:
    .venv/bin/python scripts/diagnose_inter_head_disagreement.py \\
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --data <path>.jsonl \\
        --out <path>.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_mlx_pipeline as pipeline
import pri_v2_io_plugins as io_plugins
from pri_calibrator import _load_calibration_jsonl

MIN_AUROC_SAMPLES = 5
EPS = 1e-12
DEFAULT_MIN_USABLE_FRACTION = 0.95


def _prepare_output_path(output_arg: str) -> Path:
    out_path = Path(output_arg).expanduser().resolve()
    parent = out_path.parent
    if out_path.exists() and out_path.is_dir():
        raise SystemExit(f"output path is a directory, expected a file: {out_path}")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {parent}")
    return out_path


def _find_layers(model) -> List[Any]:
    """Walk a few common nesting patterns to find the decoder block list."""
    for cand in (getattr(model, "model", None), model):
        if cand is None:
            continue
        layers = getattr(cand, "layers", None)
        if layers is not None and len(layers) > 0 and hasattr(layers[0], "self_attn"):
            return layers
    raise RuntimeError("could not find decoder blocks on model")


def _target_layer_map(n_layers: int) -> Dict[str, int]:
    return {
        "final": n_layers - 1,
        "mid": n_layers // 2,
        "last_minus_1": n_layers - 2,
    }


class _WrapAttention:
    """Module-shaped wrapper that captures softmax(Q·Kᵀ/√d) on every call.

    The PRI pipeline does not use a KV-cache; each generation step re-runs the
    full token sequence (prefix + tokens-so-far). So we can't filter "commit
    step" by L==1 — we capture every call and the caller slices captures[1] to
    get gen_step=1's first generation forward.

    On each call, we slice and store only the LAST query row of the attention
    matrix:  weights[0, :, -1, :]  has shape (n_heads, T_kv), the attention of
    the most-recent query position over all key positions. For gen_step=k, this
    is the just-appended token's attention pattern.

    When `v_norm_capture_list` is provided, the wrapper also records per-head
    per-position L2 norms of the value projection (shape (n_kv_heads, T)) into
    that parallel list. Used by the calibrator's V-norm cells (SinkProbe-style
    features); when None, the V projection is not computed (no overhead).

    Why manual weights but native output: mlx.fast.scaled_dot_product_attention
    is a fused kernel that does not return weights. We mirror the attention
    module's q/k preprocessing closely enough to recover the softmax output,
    but still return the ORIGINAL attention module's forward result so the
    diagnostic cannot perturb next-token generation.
    """

    def __init__(
        self,
        orig: Any,
        capture_list: List[np.ndarray],
        v_norm_capture_list: Optional[List[np.ndarray]] = None,
    ) -> None:
        self._orig = orig
        self._capture = capture_list
        self._v_norm_capture = v_norm_capture_list

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)

    def __call__(self, x: mx.array, mask: Optional[Any] = None, cache: Optional[Any] = None) -> mx.array:
        attn = self._orig
        self._capture.append(_capture_last_query_weights(attn, x, mask, cache))
        if self._v_norm_capture is not None:
            self._v_norm_capture.append(_capture_value_norms(attn, x))
        return attn(x, mask, cache)


def _reshape_heads(x: mx.array, batch: int, seqlen: int, n_heads: int) -> mx.array:
    return x.reshape(batch, seqlen, n_heads, -1).transpose(0, 2, 1, 3)


def _project_queries_keys(attn: Any, x: mx.array) -> Tuple[mx.array, mx.array]:
    """Mirror the native attention module's q/k projection path.

    Supports the model families in the scheduled panel:
      - Llama / Mistral / Qwen2: separate q_proj / k_proj
      - Qwen3 / Gemma3: separate projections plus q_norm / k_norm
      - Phi3 / Phi4-mini: packed qkv_proj split into q / k / v slices
    """
    batch, seqlen, _ = x.shape
    if hasattr(attn, "qkv_proj"):
        head_dim = int(getattr(attn, "head_dim"))
        qkv = attn.qkv_proj(x)
        query_pos = attn.n_heads * head_dim
        key_pos = query_pos + attn.n_kv_heads * head_dim
        q_raw, k_raw, _v_raw = mx.split(qkv, [query_pos, key_pos], axis=-1)
    elif hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
        q_raw = attn.q_proj(x)
        k_raw = attn.k_proj(x)
    else:
        raise RuntimeError(
            f"unsupported attention projection layout on {type(attn).__name__}"
        )

    queries = _reshape_heads(q_raw, batch, seqlen, int(attn.n_heads))
    keys = _reshape_heads(k_raw, batch, seqlen, int(attn.n_kv_heads))
    if hasattr(attn, "q_norm"):
        queries = attn.q_norm(queries)
    if hasattr(attn, "k_norm"):
        keys = attn.k_norm(keys)
    return queries, keys


def _apply_attention_mask(scores: mx.array, mask: Optional[Any]) -> mx.array:
    if mask is None:
        return scores
    if isinstance(mask, str):
        qL, kL = scores.shape[-2:]
        q_idx = mx.arange(kL - qL, kL)
        k_idx = mx.arange(kL)
        cmask = q_idx[:, None] >= k_idx[None]
        return mx.where(cmask, scores, mx.finfo(scores.dtype).min)
    if hasattr(mask, "dtype") and mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask


def _capture_last_query_weights(
    attn: Any,
    x: mx.array,
    mask: Optional[Any],
    cache: Optional[Any],
) -> np.ndarray:
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
    # Cast to fp32 BEFORE the matmul to avoid overflow at deep layers.
    # On Qwen 2.5 7B's final block (layer 27), `q @ kᵀ` in float16 produces
    # scores up to ~1800 with sporadic +inf in unmasked positions, which then
    # propagate through softmax → NaN in 180/200 ANLI R1 captures. fp32 keeps
    # the dynamic range safe at trivial wall cost (capture path only — the
    # model's native forward stays in its original dtype because we return
    # `attn(x, mask, cache)` unmodified). Fix dated 2026-05-15.
    queries = queries.astype(mx.float32)
    keys = keys.astype(mx.float32)
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * attn.scale
    scores = _apply_attention_mask(scores, mask)
    weights = mx.softmax(scores, axis=-1, precise=True).astype(mx.float32)
    mx.eval(weights)
    # Slice last-query row only: (B=1, H, L_q, T_k) -> (H, T_k) at q=L_q-1.
    return np.array(weights)[0, :, -1, :]


def _project_values(attn: Any, x: mx.array) -> mx.array:
    """Mirror the native attention module's v projection path.

    Supports the same families as `_project_queries_keys`:
      - separate `v_proj`
      - packed `qkv_proj` (Phi3 / Phi4-mini) — value is the third slice
    Returns the value tensor reshaped to (B, n_kv_heads, T, head_dim).
    """
    batch, seqlen, _ = x.shape
    if hasattr(attn, "qkv_proj"):
        head_dim = int(getattr(attn, "head_dim"))
        qkv = attn.qkv_proj(x)
        query_pos = attn.n_heads * head_dim
        key_pos = query_pos + attn.n_kv_heads * head_dim
        _q_raw, _k_raw, v_raw = mx.split(qkv, [query_pos, key_pos], axis=-1)
    elif hasattr(attn, "v_proj"):
        v_raw = attn.v_proj(x)
    else:
        raise RuntimeError(
            f"unsupported attention projection layout for V on {type(attn).__name__}"
        )
    return _reshape_heads(v_raw, batch, seqlen, int(attn.n_kv_heads))


def _capture_value_norms(
    attn: Any,
    x: mx.array,
) -> np.ndarray:
    """Compute per-head per-position L2 norm of the value vectors for this
    forward call.

    Returns shape (n_kv_heads, T) — value norm for each token position from
    each KV head. GQA isn't expanded here (norms are KV-group-level by
    construction). The caller is responsible for expansion if it wants
    per-Q-head norms.

    Cast to fp32 BEFORE the elementwise square so float16's narrow exponent
    range doesn't clip large value vectors (same overflow class as the
    attention scores fix above).
    """
    batch, _seqlen, _ = x.shape
    if batch != 1:
        raise RuntimeError(f"expected batch size 1 during trace, got {batch}")
    values = _project_values(attn, x)  # (1, n_kv_heads, T, head_dim)
    values_fp32 = values.astype(mx.float32)
    norms = mx.sqrt(mx.sum(values_fp32 * values_fp32, axis=-1))  # (1, n_kv_heads, T)
    mx.eval(norms)
    return np.array(norms)[0]  # (n_kv_heads, T)


@contextmanager
def attention_capture(layers: List[Any], target_indices: Dict[str, int]):
    """Wraps target layers' self_attn with capture proxies for the duration of
    the `with` block. Returns a dict tag -> list of (n_heads, T_kv) arrays.
    """
    captures: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    originals: Dict[str, Any] = {}
    try:
        for tag, idx in target_indices.items():
            layer = layers[idx]
            originals[tag] = layer.self_attn
            layer.self_attn = _WrapAttention(layer.self_attn, captures[tag])
        yield captures
    finally:
        for tag, idx in target_indices.items():
            layers[idx].self_attn = originals[tag]


@contextmanager
def attention_capture_with_values(layers: List[Any], target_indices: Dict[str, int]):
    """Like `attention_capture` but also collects per-head per-position L2
    norms of the value projection on every forward call.

    Yields a 2-tuple `(weights_captures, v_norm_captures)`:
      * `weights_captures[tag][k]` — attention weights at forward call k,
        shape `(n_heads, T_kv)` (last query row only). Same as
        `attention_capture`.
      * `v_norm_captures[tag][k]` — value norms at forward call k, shape
        `(n_kv_heads, T)` — one L2 norm per (KV head, token position).

    Used by the calibrator when V-norm cells are in the panel (SinkProbe-
    style features). Adds one V projection + sqrt-sum-sq per target layer
    per forward call; typically <5% overhead vs `attention_capture` alone.
    """
    captures: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    v_norm_captures: Dict[str, List[np.ndarray]] = {tag: [] for tag in target_indices}
    originals: Dict[str, Any] = {}
    try:
        for tag, idx in target_indices.items():
            layer = layers[idx]
            originals[tag] = layer.self_attn
            layer.self_attn = _WrapAttention(
                layer.self_attn,
                captures[tag],
                v_norm_capture_list=v_norm_captures[tag],
            )
        yield captures, v_norm_captures
    finally:
        for tag, idx in target_indices.items():
            layers[idx].self_attn = originals[tag]


def _js_radius(weights: np.ndarray) -> float:
    """Information radius: (1/H) Σ_h JS(A^h || centroid).

    weights: (H, T) — H head distributions over T past positions.
    JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M = 0.5*(P+Q).
    Bounded in [0, log 2].
    """
    if weights.ndim != 2 or weights.shape[0] < 2:
        return float("nan")
    p = weights.astype(np.float64) + EPS
    p /= p.sum(axis=1, keepdims=True)
    centroid = p.mean(axis=0)  # (T,)
    centroid /= centroid.sum() + EPS

    m = 0.5 * (p + centroid[None, :])  # (H, T)
    kl_pm = np.sum(p * (np.log(p) - np.log(m + EPS)), axis=1)
    kl_cm = np.sum(centroid[None, :] * (np.log(centroid[None, :] + EPS) - np.log(m + EPS)), axis=1)
    js_per_head = 0.5 * (kl_pm + kl_cm)
    return float(js_per_head.mean())


def _drop_bos_and_renorm(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 2 or weights.shape[1] < 2:
        return np.empty((0, 0), dtype=np.float64)
    trimmed = weights[:, 1:].astype(np.float64)
    denom = trimmed.sum(axis=1, keepdims=True)
    valid = denom.squeeze(-1) > 0
    if not np.all(valid):
        return np.empty((0, 0), dtype=np.float64)
    trimmed /= denom
    return trimmed


def _js_radius_no_bos(weights: np.ndarray) -> float:
    trimmed = _drop_bos_and_renorm(weights)
    if trimmed.size == 0:
        return float("nan")
    return _js_radius(trimmed)


def _js_radius_kv_groups(weights: np.ndarray, n_kv_heads: int) -> float:
    """Collapse Q heads that share a KV group before computing JS-radius.

    This de-biases the cross-head disagreement metric for grouped-query
    attention, where models with larger Q:KV repeat factors can look
    artificially more "converged" simply because more heads share the same K/V.
    """
    if weights.ndim != 2 or weights.shape[0] < 2 or n_kv_heads < 1:
        return float("nan")
    n_heads, width = weights.shape
    if n_heads % n_kv_heads != 0:
        return float("nan")
    repeats = n_heads // n_kv_heads
    p = weights.astype(np.float64)
    denom = p.sum(axis=1, keepdims=True)
    valid = denom.squeeze(-1) > 0
    if not np.all(valid):
        return float("nan")
    p /= denom
    grouped = p.reshape(n_kv_heads, repeats, width).mean(axis=1)
    grouped /= grouped.sum(axis=1, keepdims=True)
    return _js_radius(grouped)


def _mean_bos_mass(weights: np.ndarray) -> float:
    if weights.ndim != 2 or weights.shape[1] < 1:
        return float("nan")
    p = weights.astype(np.float64)
    denom = p.sum(axis=1, keepdims=True)
    valid = denom.squeeze(-1) > 0
    if not np.all(valid):
        return float("nan")
    p /= denom
    return float(p[:, 0].mean())


def _attention_entropy(weights: np.ndarray) -> float:
    """Mean per-head entropy of the attention distribution. Sanity-check
    feature: very low entropy = attention concentrated on one position
    (committed), very high entropy = spread evenly (uncertain). This is a
    PER-HEAD property, complementary to the cross-head JS-radius.
    """
    if weights.ndim != 2:
        return float("nan")
    p = weights.astype(np.float64) + EPS
    p /= p.sum(axis=1, keepdims=True)
    H = -np.sum(p * np.log(p), axis=1)
    return float(H.mean())


# ─── V-norm metrics (SinkProbe refinement; added 2026-05-15 evening) ─────────
# Captured V-norms have shape (n_kv_heads, T) — one L2 norm per (KV head,
# token position). Below helpers reduce these to a single scalar per sample.

def _mean_v_norm_bos(v_norms: np.ndarray) -> float:
    """Mean over KV heads of the L2 norm of the value vector at token 0
    (the BOS sink). Single scalar per layer per forward.

    SinkProbe motivation: high ‖V_0‖ + high attention mass on token 0 →
    BOS sink is computationally active → hallucination-correlated.
    """
    if v_norms.ndim != 2 or v_norms.shape[1] < 1:
        return float("nan")
    return float(v_norms[:, 0].astype(np.float64).mean())


def _mean_v_norm_max(v_norms: np.ndarray) -> float:
    """Mean over KV heads of the maximum L2 norm across all token
    positions. Captures "is there ANY computationally active sink in this
    head's value sequence". Robust to sink-position drift across samples.
    """
    if v_norms.ndim != 2 or v_norms.shape[1] < 1:
        return float("nan")
    return float(v_norms.astype(np.float64).max(axis=1).mean())


def _lastq_weighted_v_norm(weights: np.ndarray, v_norms: np.ndarray) -> float:
    """Σ_i A^h_{q=-1, i} · ‖V_i^h‖ averaged over Q heads.

    The closest single-scalar analog to SinkProbe's "sinks with large
    value-vector norms dominate the attention output". For each Q head h,
    weight the V norm at each key position by the head's attention to that
    position from the last (committed-token) query. Then average across
    heads. For GQA models, V norms are expanded from KV-group level to
    Q-head level by repetition (each Q head sees its KV group's V).
    """
    if weights.ndim != 2 or v_norms.ndim != 2:
        return float("nan")
    n_q, t_kv = weights.shape
    n_kv, t_v = v_norms.shape
    if t_kv != t_v or n_q < 1 or n_kv < 1:
        return float("nan")
    if n_q % n_kv != 0:
        return float("nan")
    repeats = n_q // n_kv
    # Expand v_norms to per-Q-head: (n_kv, T) → (n_kv, repeats, T) →
    # (n_q, T) preserving the within-group ordering.
    v_per_q = np.repeat(v_norms.astype(np.float64), repeats, axis=0)
    p = weights.astype(np.float64)
    denom = p.sum(axis=1, keepdims=True)
    valid = denom.squeeze(-1) > 0
    if not np.all(valid):
        return float("nan")
    p /= denom
    weighted = np.sum(p * v_per_q, axis=1)  # (n_q,)
    return float(weighted.mean())


def _raw_auroc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, str]:
    """Raw AUROC on the score as defined, without post-hoc orientation flips."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < MIN_AUROC_SAMPLES:
        return float("nan"), "?"
    s = scores[mask]
    y = labels[mask].astype(int)
    if len(np.unique(y)) < 2 or np.isclose(s.std(), 0.0):
        return float("nan"), "?"
    a = pipeline.safe_auroc(y, s)
    if np.isclose(a, 0.5):
        return a, "="
    return a, ("hi" if a > 0.5 else "lo")


def _write_rows_csv(out_path: Path, rows: List[Dict[str, Any]], tags: List[str]) -> None:
    header = ["sample_idx", "label", "surprise_gen1"]
    for tag in tags:
        header.append(f"js_radius_{tag}")
        header.append(f"js_radius_kv_groups_{tag}")
        header.append(f"js_radius_no_bos_{tag}")
        header.append(f"bos_mass_{tag}")
        header.append(f"attn_entropy_{tag}")
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=out_path.parent,
            prefix=f".{out_path.stem}.", suffix=f"{out_path.suffix or '.csv'}.tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            w = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL)
            w.writerow(header)
            for row in rows:
                surprise = row.get("surprise_gen1", float("nan"))
                surprise_s = f"{surprise:.6f}" if isinstance(surprise, float) and math.isfinite(surprise) else "nan"
                rec = [row["sample_idx"], row["label"], surprise_s]
                for tag in tags:
                    for col in (
                        f"js_radius_{tag}",
                        f"js_radius_kv_groups_{tag}",
                        f"js_radius_no_bos_{tag}",
                        f"bos_mass_{tag}",
                        f"attn_entropy_{tag}",
                    ):
                        v = row.get(col, float("nan"))
                        rec.append(f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else "nan")
                w.writerow(rec)
        os.replace(tmp_path, out_path)
    except OSError as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise SystemExit(f"failed to write {out_path}: {exc}") from exc


def main() -> int:
    p = argparse.ArgumentParser(description="Inter-head attention disagreement at commit step")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="/tmp/inter_head_disagreement.csv")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap n samples (default: all)")
    p.add_argument(
        "--min-usable-fraction",
        type=float,
        default=DEFAULT_MIN_USABLE_FRACTION,
        help="fail if usable samples fall below this fraction of requested rows",
    )
    args = p.parse_args()

    out_path = _prepare_output_path(args.out)

    prompts, labels, _ = _load_calibration_jsonl(args.data)
    if args.limit:
        prompts, labels = prompts[: args.limit], labels[: args.limit]

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, projection, layer_indices = pipeline.load_model(args.model, cfg)

    layers = _find_layers(model)
    target_map = _target_layer_map(len(layers))
    tags = list(target_map.keys())
    print(f"[head-disagree] model={args.model}")
    print(f"[head-disagree] n_layers={len(layers)}  target indices={target_map}")
    print(f"[head-disagree] {len(prompts)} samples")
    prompt_strategy = io_plugins.get_prompt_strategy(args.model)

    rows: List[Dict[str, Any]] = []
    n_trace_failed = 0
    n_prefix_only_eos = 0
    print(f"[head-disagree] tracing {len(prompts)} samples ...")
    for i, prompt in enumerate(prompts):
        wrapped = prompt_strategy(prompt, tokenizer)
        try:
            with attention_capture(layers, target_map) as captures:
                trace = pipeline.trace_sample(
                    model=model, tokenizer=tokenizer, prompt=wrapped,
                    layer_indices=layer_indices, output_projection=projection,
                    max_new_tokens=args.max_new_tokens, v3_capture=False,
                )
        except Exception as e:
            n_trace_failed += 1
            print(f"[head-disagree]   sample {i}: trace FAILED ({e})")
            continue

        # gen_step=1 surprise = -log(p_t[generated_token]) at the commit step.
        # trace_sample emits gen_surprises as a list aligned with gen tokens.
        surprise_gen1 = float("nan")
        try:
            gs = trace.get("gen_surprises") or []
            if gs:
                v = float(gs[0])
                if math.isfinite(v):
                    surprise_gen1 = v
        except (AttributeError, TypeError, IndexError):
            pass

        row: Dict[str, Any] = {
            "sample_idx": i,
            "label": int(labels[i]),
            "surprise_gen1": surprise_gen1,
        }
        gen_token_ids = trace.get("gen_token_ids") or []
        expected_calls = 1 + len(gen_token_ids)
        usable = True
        for tag in tags:
            caps = captures[tag]
            if len(caps) != expected_calls:
                raise RuntimeError(
                    f"capture count mismatch at sample {i}, tag={tag}: "
                    f"expected {expected_calls}, got {len(caps)}"
                )
            # captures[0] = prefix forward (T_prefix queries); captures[1] is
            # the first generation forward when at least one token was
            # committed. That last-query row is gen_step=1's commit attention.
            if expected_calls < 2:
                usable = False
                break
            w = caps[1]
            n_kv_heads = int(layers[target_map[tag]].self_attn.n_kv_heads)
            row[f"js_radius_{tag}"] = _js_radius(w)
            row[f"js_radius_kv_groups_{tag}"] = _js_radius_kv_groups(w, n_kv_heads)
            row[f"js_radius_no_bos_{tag}"] = _js_radius_no_bos(w)
            row[f"bos_mass_{tag}"] = _mean_bos_mass(w)
            row[f"attn_entropy_{tag}"] = _attention_entropy(w)
        if not usable:
            n_prefix_only_eos += 1
            continue
        rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            print(f"[head-disagree]   {i+1}/{len(prompts)}")

    if not rows:
        raise SystemExit("no usable samples")

    n_total = len(prompts)
    n_usable = len(rows)
    required_usable = min(n_total, max(1, math.ceil(n_total * args.min_usable_fraction)))
    if n_usable < required_usable:
        raise SystemExit(
            f"usable coverage too low: {n_usable}/{n_total} "
            f"(trace_failed={n_trace_failed}, prefix_only_eos={n_prefix_only_eos}); "
            f"need at least {required_usable}"
        )

    _write_rows_csv(out_path, rows, tags)
    print(f"[head-disagree] wrote {len(rows)} rows to {out_path}")
    print(
        f"[head-disagree] coverage usable={n_usable}/{n_total}  "
        f"trace_failed={n_trace_failed}  prefix_only_eos={n_prefix_only_eos}"
    )

    # ── Summary ──
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    print()
    print("=" * 80)
    print(f"  Head-disagreement summary (n={len(rows)})")
    print("=" * 80)
    print(f"  layer         | raw AUROC JS   raw AUROC JS-KV-groups   raw AUROC JS-no-BOS   raw AUROC BOS-mass   raw AUROC attn-entropy")
    print(f"  --------------+-----------------------------------------------------------------------------------------------------")
    for tag in tags:
        js = np.array([r[f"js_radius_{tag}"] for r in rows])
        js_kv = np.array([r[f"js_radius_kv_groups_{tag}"] for r in rows])
        js_no_bos = np.array([r[f"js_radius_no_bos_{tag}"] for r in rows])
        bos_mass = np.array([r[f"bos_mass_{tag}"] for r in rows])
        en = np.array([r[f"attn_entropy_{tag}"] for r in rows])
        au_js, dir_js = _raw_auroc(y, js)
        au_js_kv, dir_js_kv = _raw_auroc(y, js_kv)
        au_js_no_bos, dir_js_no_bos = _raw_auroc(y, js_no_bos)
        au_bos, dir_bos = _raw_auroc(y, bos_mass)
        au_en, dir_en = _raw_auroc(y, en)
        print(
            f"  {tag:<13s} | {au_js:.4f} ({dir_js})      "
            f"{au_js_kv:.4f} ({dir_js_kv})      "
            f"{au_js_no_bos:.4f} ({dir_js_no_bos})      "
            f"{au_bos:.4f} ({dir_bos})      "
            f"{au_en:.4f} ({dir_en})"
        )
    print()
    print("  Raw AUROC is evaluated on the metric as defined: 'hi' means higher predicts contradiction,")
    print("  'lo' means lower predicts contradiction, and '=' is chance. No post-hoc sign flipping is applied.")
    print("  JS-KV-groups collapses heads within each shared-KV group before scoring, which makes cross-GQA comparisons fairer.")
    print("  JS-radius is the headline. JS-no-BOS and BOS-mass make the sink-token falsification check visible in the run output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
