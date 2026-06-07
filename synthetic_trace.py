"""
Trace collection and event-aligned analysis for synthetic contradiction experiments.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import mlx.core as mx
import numpy as np

import pri_metrics
import config


PRIMARY_SIGNALS = (
    "pri",
    "delta_sigma_jsd",
    "acr_mid_mean",
    "momentum_pc1_ratio",
    "momentum_mean_cos",
)


def _to_token_ids(encoded: Any) -> List[int]:
    if isinstance(encoded, mx.array):
        return [int(v) for v in encoded.tolist()]
    if isinstance(encoded, list):
        return [int(v) for v in encoded]
    return [int(v) for v in list(encoded)]


def tokenize_prompt(tokenizer: Any, prompt: str) -> List[int]:
    return _to_token_ids(tokenizer.encode(prompt))


def resolve_anchor_token_index(
    tokenizer: Any,
    prompt: str,
    anchor_char_index: int,
) -> int:
    full_ids = tokenize_prompt(tokenizer, prompt)
    if len(full_ids) < 2:
        return 0
    prefix_ids = tokenize_prompt(tokenizer, prompt[: max(0, int(anchor_char_index))])
    anchor = int(len(prefix_ids))
    anchor = max(1, min(anchor, len(full_ids) - 1))
    return anchor


def _compute_acr(adapter: Any) -> Tuple[float, float, Dict[int, float], bool]:
    acr_full_mean = 0.0
    acr_mid_mean = 0.0
    acr_by_layer: Dict[int, float] = {}
    if getattr(adapter, "acr_collector", None) is None:
        return acr_mid_mean, acr_full_mean, acr_by_layer, False

    full_range = (0, len(adapter.layers))
    acr_by_layer = adapter.acr_collector.compute_acr_by_layer(layer_range=full_range)
    if not acr_by_layer:
        return acr_mid_mean, acr_full_mean, acr_by_layer, False

    acr_full_mean = float(sum(acr_by_layer.values()) / len(acr_by_layer))
    layer_range = adapter.acr_collector.layer_range
    if layer_range is None:
        mid_values = list(acr_by_layer.values())
    else:
        start, end = layer_range
        mid_values = [v for layer_idx, v in acr_by_layer.items() if start <= layer_idx < end]
    if mid_values:
        acr_mid_mean = float(sum(mid_values) / len(mid_values))
    return acr_mid_mean, acr_full_mean, acr_by_layer, True


def _compute_canonical_metrics(
    probs: mx.array,
    hidden_vectors: List[mx.array],
    selected_token: int,
    previous_hidden_final: mx.array | None,
    adapter: Any,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> Tuple[Dict[str, Any], mx.array]:
    cfg = cfg_obj if cfg_obj is not None else config.DEFAULT_UNCERTAINTY_CONFIG
    delta_sigma_jsd = pri_metrics.compute_delta_sigma_mean_cross_layer_jsd(
        hidden_vectors,
        cfg_obj=cfg,
    )
    surprise = pri_metrics.compute_surprise(
        probs,
        selected_token,
        cfg_obj=cfg,
    )
    current_hidden_final = hidden_vectors[-1]
    if previous_hidden_final is None:
        delta_h = 0.0
        pri = float(surprise)
    else:
        delta_h = pri_metrics.compute_cosine_distance(
            current_hidden_final,
            previous_hidden_final,
            cfg_obj=cfg,
        )
        pri = pri_metrics.compute_pri(
            surprise,
            delta_h,
            alpha=cfg.pri_alpha,
            cfg_obj=cfg,
        )

    svd_features = pri_metrics.compute_svd_spectrum_features(
        hidden_vectors,
        cfg_obj=cfg,
    )
    momentum_features = pri_metrics.compute_momentum_features(
        hidden_vectors,
        cfg_obj=cfg,
    )
    acr_mid_mean, acr_full_mean, acr_by_layer, acr_valid = _compute_acr(adapter)

    metrics = {
        "delta_sigma_jsd": float(delta_sigma_jsd),
        "surprise": float(surprise),
        "delta_h": float(delta_h),
        "pri": float(pri),
        "acr_mid_mean": float(acr_mid_mean),
        "acr_full_mean": float(acr_full_mean),
        "acr_valid": bool(acr_valid),
        "acr_by_layer": {int(k): float(v) for k, v in acr_by_layer.items()},
        "pc1_ratio": float(svd_features["pc1_ratio"]),
        "effective_rank": float(svd_features["effective_rank"]),
        "spectral_entropy": float(svd_features["spectral_entropy"]),
        "momentum_pc1_ratio": float(momentum_features["momentum_pc1_ratio"]),
        "momentum_mean_cos": float(momentum_features["momentum_mean_cos"]),
    }
    return metrics, current_hidden_final


def collect_prefix_trace(
    adapter: Any,
    tokenizer: Any,
    prompt: str,
    anchor_token_index: int,
    window_wide: int = 12,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> List[Dict[str, Any]]:
    token_ids = tokenize_prompt(tokenizer, prompt)
    if len(token_ids) < 2:
        return []

    left = max(1, int(anchor_token_index) - int(window_wide))
    right = min(len(token_ids) - 1, int(anchor_token_index) + int(window_wide))

    trace: List[Dict[str, Any]] = []
    previous_hidden_final = None
    for token_index in range(left, right + 1):
        prefix_ids = token_ids[:token_index]
        selected_token = int(token_ids[token_index])

        logits = adapter.next_token_logits(mx.array(prefix_ids))
        probs = mx.softmax(logits, axis=-1)
        hidden_vectors = adapter.collector.get_all_blocks()
        metrics, previous_hidden_final = _compute_canonical_metrics(
            probs=probs,
            hidden_vectors=hidden_vectors,
            selected_token=selected_token,
            previous_hidden_final=previous_hidden_final,
            adapter=adapter,
            cfg_obj=cfg_obj,
        )
        metrics["token_index"] = int(token_index)
        metrics["selected_token"] = int(selected_token)
        metrics["relative_offset"] = int(token_index - anchor_token_index)
        trace.append(metrics)
    return trace


def collect_generation_trace(
    adapter: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 12,
    temperature: float = 0.0,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> Dict[str, Any]:
    input_ids = tokenize_prompt(tokenizer, prompt)
    if not input_ids:
        return {
            "tokens": [],
            "text": "",
            "halt_reason": "empty_prompt",
            "halted": True,
            "trajectory": [],
        }

    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
    generated_tokens: List[int] = []
    trajectory: List[Dict[str, Any]] = []
    previous_hidden_final = None
    halt_reason = "max_tokens reached"

    for step in range(int(max_tokens)):
        logits = adapter.next_token_logits(mx.array(input_ids))
        probs = mx.softmax(logits, axis=-1)
        hidden_vectors = adapter.collector.get_all_blocks()

        if float(temperature) == 0.0:
            next_token = int(mx.argmax(logits).item())
        else:
            temp_probs = mx.softmax(logits / float(temperature), axis=-1)
            next_token = int(mx.random.categorical(temp_probs).item())

        metrics, previous_hidden_final = _compute_canonical_metrics(
            probs=probs,
            hidden_vectors=hidden_vectors,
            selected_token=next_token,
            previous_hidden_final=previous_hidden_final,
            adapter=adapter,
            cfg_obj=cfg_obj,
        )
        metrics["step"] = int(step)
        metrics["selected_token"] = int(next_token)
        trajectory.append(metrics)

        generated_tokens.append(next_token)
        input_ids.append(next_token)

        if eos_token_id is not None and next_token == int(eos_token_id):
            halt_reason = "EOS token generated"
            break

    generated_text = tokenizer.decode(generated_tokens) if generated_tokens else ""
    return {
        "tokens": [int(t) for t in generated_tokens],
        "text": generated_text,
        "halt_reason": halt_reason,
        "halted": bool(halt_reason != "EOS token generated" and halt_reason != "empty_prompt"),
        "trajectory": trajectory,
    }


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    if not group_a or not group_b:
        return 0.0
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    pooled = np.sqrt((float(np.var(a)) + float(np.var(b))) / 2.0)
    if pooled <= 0.0:
        return 0.0
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def compute_window_summaries(
    prefix_trace: List[Dict[str, Any]],
    windows: Iterable[int] = (5, 12),
    signal_keys: Iterable[str] = PRIMARY_SIGNALS,
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for window in windows:
        points = [
            p for p in prefix_trace
            if abs(int(p.get("relative_offset", 0))) <= int(window)
        ]
        pre = [p for p in points if int(p.get("relative_offset", 0)) < 0]
        post = [p for p in points if int(p.get("relative_offset", 0)) >= 0]
        signal_summary: Dict[str, Any] = {}
        for key in signal_keys:
            pre_vals = [float(p.get(key, 0.0)) for p in pre]
            post_vals = [float(p.get(key, 0.0)) for p in post]
            pre_mean = _safe_mean(pre_vals)
            post_mean = _safe_mean(post_vals)
            delta = float(post_mean - pre_mean)
            peak_offset = 0
            peak_value = 0.0
            if points:
                peak_point = max(points, key=lambda x: float(x.get(key, 0.0)))
                peak_offset = int(peak_point.get("relative_offset", 0))
                peak_value = float(peak_point.get(key, 0.0))
            signal_summary[key] = {
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "delta_post_minus_pre": delta,
                "peak_value": peak_value,
                "peak_offset": int(peak_offset),
                "abs_peak_offset": int(abs(peak_offset)),
                "n_points": int(len(points)),
            }
        summaries[f"w{int(window)}"] = signal_summary
    return summaries


def extract_peak_offsets(
    sample_records: List[Dict[str, Any]],
    signal_key: str,
    window_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sample_records:
        win = row.get("window_summaries", {}).get(window_key, {})
        signal_stats = win.get(signal_key)
        if not signal_stats:
            continue
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "chain_steps": int(row.get("chain_steps", 0)),
                "has_contradiction": bool(row.get("has_contradiction", False)),
                "abs_peak_offset": float(signal_stats.get("abs_peak_offset", 0.0)),
                "peak_offset": float(signal_stats.get("peak_offset", 0.0)),
            }
        )
    return rows


def permutation_test_peak_offset(
    peak_rows: List[Dict[str, Any]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    if not peak_rows:
        return {
            "mean_abs_offset_contradiction": 0.0,
            "mean_abs_offset_control": 0.0,
            "observed_diff": 0.0,
            "p_value_two_sided": 1.0,
            "cohens_d": 0.0,
            "n_samples": 0,
            "n_permutations": int(n_permutations),
        }

    by_chain: Dict[int, List[Dict[str, Any]]] = {}
    for row in peak_rows:
        by_chain.setdefault(int(row["chain_steps"]), []).append(row)

    contr_all = [float(r["abs_peak_offset"]) for r in peak_rows if r["has_contradiction"]]
    ctrl_all = [float(r["abs_peak_offset"]) for r in peak_rows if not r["has_contradiction"]]
    mean_contr = _safe_mean(contr_all)
    mean_ctrl = _safe_mean(ctrl_all)
    observed_diff = float(mean_contr - mean_ctrl)

    rng = random.Random(seed)
    perm_diffs: List[float] = []
    for _ in range(int(n_permutations)):
        total_contr_sum = 0.0
        total_ctrl_sum = 0.0
        total_contr_n = 0
        total_ctrl_n = 0
        for rows in by_chain.values():
            vals = [float(r["abs_peak_offset"]) for r in rows]
            labels = [bool(r["has_contradiction"]) for r in rows]
            n_contr = sum(1 for x in labels if x)
            perm_labels = labels[:]
            rng.shuffle(perm_labels)
            contr_vals = [v for v, lab in zip(vals, perm_labels) if lab]
            ctrl_vals = [v for v, lab in zip(vals, perm_labels) if not lab]
            total_contr_sum += float(sum(contr_vals))
            total_ctrl_sum += float(sum(ctrl_vals))
            total_contr_n += int(len(contr_vals))
            total_ctrl_n += int(len(ctrl_vals))
            if n_contr == 0 or n_contr == len(labels):
                break
        if total_contr_n == 0 or total_ctrl_n == 0:
            perm_diffs.append(0.0)
        else:
            perm_diffs.append(
                float((total_contr_sum / total_contr_n) - (total_ctrl_sum / total_ctrl_n))
            )

    perm_abs = np.abs(np.asarray(perm_diffs, dtype=np.float64))
    p_two = float(np.mean(perm_abs >= abs(observed_diff))) if len(perm_abs) else 1.0
    return {
        "mean_abs_offset_contradiction": float(mean_contr),
        "mean_abs_offset_control": float(mean_ctrl),
        "observed_diff": float(observed_diff),
        "p_value_two_sided": float(p_two),
        "cohens_d": float(_cohens_d(contr_all, ctrl_all)),
        "n_samples": int(len(peak_rows)),
        "n_permutations": int(n_permutations),
    }


def summarize_signal_effects(
    sample_records: List[Dict[str, Any]],
    windows: Iterable[int] = (5, 12),
    signal_keys: Iterable[str] = PRIMARY_SIGNALS,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for window in windows:
        w_key = f"w{int(window)}"
        out[w_key] = {}
        for signal in signal_keys:
            rows = extract_peak_offsets(sample_records, signal_key=signal, window_key=w_key)
            peak_test = permutation_test_peak_offset(
                rows,
                n_permutations=n_permutations,
                seed=seed,
            )
            deltas_contr = []
            deltas_ctrl = []
            for sample in sample_records:
                stats = sample.get("window_summaries", {}).get(w_key, {}).get(signal, {})
                delta = float(stats.get("delta_post_minus_pre", 0.0))
                if sample.get("has_contradiction"):
                    deltas_contr.append(delta)
                else:
                    deltas_ctrl.append(delta)
            out[w_key][signal] = {
                "peak_offset_test": peak_test,
                "delta_post_minus_pre": {
                    "mean_contradiction": _safe_mean(deltas_contr),
                    "mean_control": _safe_mean(deltas_ctrl),
                    "observed_diff": float(_safe_mean(deltas_contr) - _safe_mean(deltas_ctrl)),
                    "cohens_d": float(_cohens_d(deltas_contr, deltas_ctrl)),
                    "n_contradiction": int(len(deltas_contr)),
                    "n_control": int(len(deltas_ctrl)),
                },
            }
    return out
