#!/usr/bin/env python3
"""Comprehensive shadow-ambiguity build harness.

Per invocation runs one (model, benchmark) pair and writes one JSON artifact.
The full all-models x all-benchmarks run is intentionally orchestrated outside
this script as separate processes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - scipy is present in the t0 venv
    student_t = None

HERE = Path(__file__).resolve().parent
REPO = HERE
while REPO != REPO.parent:
    if (REPO / "pri_runtime.py").exists():
        break
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))

import mlx.core as mx  # noqa: E402
import pri_runtime as pipeline  # noqa: E402
import pri_v2_io_plugins as io_plugins  # noqa: E402
from pri_calibrator import _load_calibration_jsonl  # noqa: E402
from test_shadow_ambiguity import (  # noqa: E402
    fisher_eff_rank,
    fisher_spectral_entropy,
    shadow_logvol_post_rank,
)

SCHEMA = "shadow_ambiguity_comprehensive/v2"
META_SCHEMA = "shadow_ambiguity_comprehensive_meta/v2"
FRESH_SEED = 20260611
K_SUPPORT_DEFAULT = 512
META_MIN_ENDPOINT_SE = 1e-8
PRIMARY_STAT = "fisher_eff_rank"
SECONDARY_STATS = ("neg_shadow_logvol_r1", "spectral_entropy")
ALL_STATS = (PRIMARY_STAT,) + SECONDARY_STATS
BASE_SURPRISE = ("surprise",)
BASE_SURPRISE_NULL = ("surprise", "null_ratio_post_rank1")
BASE_FULL = ("surprise", "null_ratio_post_rank1", "p_max")
BASE_CONFIDENCE = ("surprise", "p_max")
KNOWN_SKIP_HINTS = {
    "mlx-community/gpt-oss-20b-MXFP4-Q4": "known too heavy for this harness on local MLX cache",
}
KNOWN_GATE_RISKS = {
    "mlx-community/gemma-3-1b-it-4bit": "known harness-gate risk; attempted if explicitly selected",
    "mlx-community/dolphin-2.9.3-mistral-nemo-12b-4bit": "known harness-gate risk; attempted if explicitly selected",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(":", "_")


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".json.tmp",
            delete=False,
        ) as f:
            tmp = Path(f.name)
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def discover_models(cache_root: Path = Path.home() / ".cache/huggingface/hub") -> List[str]:
    models = set()
    if cache_root.exists():
        for p in cache_root.glob("models--mlx-community--*"):
            name = p.name.replace("models--mlx-community--", "", 1)
            if name:
                models.add(f"mlx-community/{name}")
    return sorted(models)


def _benchmark_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("anli_r1"):
        return "anli_r1"
    if stem.startswith("anli_r2"):
        return "anli_r2"
    if stem.startswith("anli_r3"):
        return "anli_r3"
    if stem.startswith("triviaqa_paired"):
        return "triviaqa_paired"
    return stem


def discover_benchmarks() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted((REPO / "experiments/t0-sealed").glob("*/data/*.jsonl")):
        out.setdefault(_benchmark_name(p), p)
    return out


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (float(np.sum(e)) + 1e-300)


def fc_full_spectrum(W_s: np.ndarray, p_s: np.ndarray, d: int) -> np.ndarray:
    """Full d-dimensional centered-Fisher spectrum via the K x K dual."""
    p_s = np.asarray(p_s, dtype=np.float64)
    p_s = p_s / (float(np.sum(p_s)) + 1e-300)
    B = np.diag(p_s) - np.outer(p_s, p_s)
    B = 0.5 * (B + B.T)
    wB, QB = np.linalg.eigh(B)
    wB = np.clip(wB, 0.0, None)
    R = np.einsum("ij,jk->ik", QB * np.sqrt(wB), QB.T, optimize=True)
    gram = np.einsum("ik,jk->ij", W_s, W_s, optimize=True)
    M = np.einsum("ij,jk,kl->il", R, gram, R, optimize=True)
    M = 0.5 * (M + M.T)
    eig = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    pad = max(int(d) - eig.size, 0)
    return np.concatenate([eig, np.zeros(pad, dtype=np.float64)]) if pad else eig


def _topk_indices(p: np.ndarray, k_support: int) -> np.ndarray:
    k = int(min(k_support, p.shape[0]))
    return np.argpartition(-p, kth=k - 1)[:k].astype(np.int32)


def _extract_final_norm_gamma(model: Any) -> Optional[np.ndarray]:
    fn = getattr(pipeline, "_extract_final_rmsnorm_gamma", None)
    if fn is None:
        return None
    gamma = fn(model)
    if gamma is None:
        return None
    return np.asarray(gamma, dtype=np.float32)


def _fit_final_layer_name(layer_indices: Dict[str, int]) -> str:
    if "final" in layer_indices:
        return "final"
    return max(layer_indices, key=lambda k: layer_indices[k])


def _pinned_late_layers(n_layers: int) -> Dict[str, Any]:
    count = int(math.ceil(float(n_layers) / 4.0))
    start = int(n_layers - count)
    layers = list(range(start, int(n_layers)))
    return {
        "n_layers": int(n_layers),
        "rule": "zero-indexed blocks start=B-ceil(B/4), stop=B-1; aggregate includes these blocks plus readout",
        "late_count": count,
        "start": start,
        "stop_inclusive": int(n_layers - 1),
        "layers": layers,
        "includes_readout": True,
    }


def _spectrum_stats(spec: np.ndarray) -> Dict[str, float]:
    raw_logvol = float(shadow_logvol_post_rank(spec, r=1))
    return {
        "fisher_eff_rank": float(fisher_eff_rank(spec)),
        "neg_shadow_logvol_r1": float(-raw_logvol),
        "spectral_entropy": float(fisher_spectral_entropy(spec)),
        "shadow_logvol_r1_raw": raw_logvol,
    }


def _support_spectrum(
    projection: Any,
    p: np.ndarray,
    d_model: int,
    k_support: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    idx = _topk_indices(p, k_support)
    W_s = projection.get_rows(idx)
    if W_s is None or W_s.ndim != 2:
        return None, idx, None
    W_s = W_s.astype(np.float64)
    if not np.isfinite(W_s).all():
        return None, idx, W_s
    spec = fc_full_spectrum(W_s, p[idx], d_model)
    return spec, idx, W_s


def _rotation_invariance_delta(
    W_s: np.ndarray,
    p_s: np.ndarray,
    d_model: int,
    seed: int,
    reference: Dict[str, float],
) -> float:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=W_s.shape[1]).astype(np.float64)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0 or not math.isfinite(norm):
        return float("nan")
    v /= norm
    # Dense Householder reflection: an orthogonal hidden-coordinate transform
    # stronger than coordinate sign flips while avoiding a full d x d QR.
    W_reflected = W_s - 2.0 * np.outer(W_s @ v, v)
    spec_rot = fc_full_spectrum(W_reflected, p_s, d_model)
    rotated = _spectrum_stats(spec_rot)
    deltas = [
        abs(float(rotated["fisher_eff_rank"]) - float(reference["fisher_eff_rank"])),
        abs(float(rotated["neg_shadow_logvol_r1"]) - float(reference["neg_shadow_logvol_r1"])),
        abs(float(rotated["spectral_entropy"]) - float(reference["spectral_entropy"])),
    ]
    return float(max(deltas))


@dataclass
class FeatureRun:
    rows: List[Dict[str, Any]]
    drops: Dict[str, int]
    diagnostics: Dict[str, Any]


def trace_pair_features(
    model_id: str,
    benchmark: str,
    data_path: Path,
    *,
    limit: int,
    max_new_tokens: int,
    k_support: int,
    seed: int,
) -> FeatureRun:
    prompts, labels, data_hash = _load_calibration_jsonl(str(data_path))
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = True
    cfg.v3_all_layers_for_first_n_steps = max(1, int(max_new_tokens))
    cfg.v3_capture_raw = False
    cfg.v3_capture_centered = False
    model, tokenizer, projection, layer_indices = pipeline.load_model(model_id, cfg)
    final_layer = _fit_final_layer_name(layer_indices)
    gamma = _extract_final_norm_gamma(model)
    if gamma is None:
        raise RuntimeError("final RMSNorm gamma unavailable; null-ratio and logit-lens paths cannot run")
    readout_core = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)
    d_model, vocab = int(projection.hidden_size), int(projection.vocab_size)

    drops = {
        "trace_failed": 0,
        "no_gen_step1": 0,
        "missing_readout": 0,
        "nonfinite_readout_probability_or_surprise": 0,
        "nonfinite_readout_null_ratio": 0,
        "missing_layer_capture": 0,
        "nonfinite_layer_hidden": 0,
        "nonfinite_logitlens_logits": 0,
        "nonfinite_layer_probability": 0,
        "support_rows_missing": 0,
        "nonfinite_support_rows": 0,
        "nonfinite_spectrum": 0,
        "incomplete_pinned_aggregate": 0,
    }
    rows: List[Dict[str, Any]] = []
    first_valid: Dict[str, Any] = {}
    layer_window: Optional[Dict[str, Any]] = None

    print(f"[inventory] benchmark={benchmark} path={_rel(data_path)}")
    print(f"[trace] model={model_id}")
    print(f"[trace] samples={len(prompts)} K={k_support} final_layer={final_layer} d={d_model} V={vocab}")
    print("[trace] commit instant: gen_step=1; pinned aggregate = final 25% block logit lens plus readout")

    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        wrapped = prompt_strategy(prompt, tokenizer)
        try:
            trace = pipeline.trace_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=wrapped,
                layer_indices=layer_indices,
                output_projection=projection,
                max_new_tokens=max_new_tokens,
                v3_capture=True,
                v3_all_for_first_n_steps=max(1, int(max_new_tokens)),
                v3_probe_fallback=["final"],
            )
        except Exception as exc:  # noqa: BLE001
            drops["trace_failed"] += 1
            print(f"[trace]   sample {i}: trace failed: {exc}", flush=True)
            continue

        captures_by_step = trace.get("gen_captures_by_step") or []
        gen_ids = trace.get("gen_token_ids") or []
        gen_probs = trace.get("gen_probs") or []
        gen_surprises = trace.get("gen_surprises") or []
        gen_hidden = (trace.get("gen_hidden") or {}).get(final_layer, [])
        if not captures_by_step or not gen_ids or not gen_probs or not gen_surprises:
            drops["no_gen_step1"] += 1
            continue
        if not gen_hidden or "last_prefix_hidden" not in trace:
            drops["missing_readout"] += 1
            continue

        p_t = np.asarray(gen_probs[0], dtype=np.float64)
        surprise = float(gen_surprises[0])
        if p_t.ndim != 1 or not np.isfinite(p_t).all() or not math.isfinite(surprise) or float(np.sum(p_t)) <= 0.0:
            drops["nonfinite_readout_probability_or_surprise"] += 1
            continue
        p_t = p_t / (float(np.sum(p_t)) + 1e-300)

        h_t = np.asarray(gen_hidden[0], dtype=np.float32)
        h_prev = np.asarray(trace["last_prefix_hidden"][final_layer], dtype=np.float32)
        comp = readout_core.compute_step(
            h_t=h_t,
            h_prev=h_prev,
            p_t=p_t,
            S_t=surprise,
            alpha=1.0,
            topk_values=[32],
            lowrank_values=[32],
            v3_rank_values=[1],
            v3_capture_raw=False,
            v3_capture_centered=False,
        )
        null_ratio = float(comp.get("null_ratio_post_rank1", float("nan")))
        if not math.isfinite(null_ratio):
            drops["nonfinite_readout_null_ratio"] += 1
            continue

        readout_spec, readout_idx, readout_W = _support_spectrum(projection, p_t, d_model, k_support)
        if readout_spec is None:
            if readout_W is None:
                drops["support_rows_missing"] += 1
            else:
                drops["nonfinite_support_rows"] += 1
            continue
        if not np.isfinite(readout_spec).all():
            drops["nonfinite_spectrum"] += 1
            continue
        readout_stats = _spectrum_stats(readout_spec)
        readout_rot_delta = _rotation_invariance_delta(
            readout_W,
            p_t[readout_idx],
            d_model,
            seed + 100_003 + i,
            readout_stats,
        )

        step_caps = captures_by_step[0]
        step_layer_indices = (trace.get("gen_layer_indices_by_step") or [{}])[0]
        n_layers = int(trace.get("n_layers") or len(step_layer_indices))
        if layer_window is None:
            layer_window = _pinned_late_layers(n_layers)
            print(f"[trace] n_layers={n_layers}; pinned_late_layers={layer_window['layers']}")

        idx_to_name = {int(v): k for k, v in step_layer_indices.items()}
        hidden_by_index: Dict[int, np.ndarray] = {}
        for li, lname in idx_to_name.items():
            cap = step_caps.get(lname)
            if cap is not None and "h_t" in cap:
                hidden_by_index[li] = np.asarray(cap["h_t"], dtype=np.float32)

        source_stats: List[Dict[str, float]] = []
        source_stats.append({**readout_stats, "p_max": float(np.max(p_t)), "source": "readout"})  # type: ignore[arg-type]
        layer_rows: List[Dict[str, Any]] = []
        rot_deltas = [readout_rot_delta]
        for layer_idx in (layer_window or {}).get("layers", []):
            h_l = hidden_by_index.get(int(layer_idx))
            # Shadow stats are logit-lens p_l-only (F_c(p_l)); no per-layer delta-h
            # is used, so the previous-layer hidden / embedding capture is unneeded.
            if h_l is None:
                drops["missing_layer_capture"] += 1
                continue
            if not np.isfinite(h_l).all():
                drops["nonfinite_layer_hidden"] += 1
                continue

            h_l_post = readout_core.rmsnorm(h_l, gamma)
            logits_l = projection.project(h_l_post)
            if logits_l.ndim != 1 or not np.isfinite(logits_l).all():
                drops["nonfinite_logitlens_logits"] += 1
                continue
            p_l = softmax_np(logits_l)
            if p_l.ndim != 1 or not np.isfinite(p_l).all() or float(np.sum(p_l)) <= 0.0:
                drops["nonfinite_layer_probability"] += 1
                continue

            spec_l, idx_l, W_l = _support_spectrum(projection, p_l, d_model, k_support)
            if spec_l is None:
                if W_l is None:
                    drops["support_rows_missing"] += 1
                else:
                    drops["nonfinite_support_rows"] += 1
                continue
            if not np.isfinite(spec_l).all():
                drops["nonfinite_spectrum"] += 1
                continue
            stats_l = _spectrum_stats(spec_l)
            rot_deltas.append(
                _rotation_invariance_delta(W_l, p_l[idx_l], d_model, seed + 200_003 + 37 * i + int(layer_idx), stats_l)
            )
            source_stats.append({**stats_l, "p_max": float(np.max(p_l)), "source": f"block_{layer_idx}"})  # type: ignore[arg-type]
            layer_rows.append(
                {
                    "layer": int(layer_idx),
                    "layer_name": idx_to_name.get(int(layer_idx), f"layer_{layer_idx}"),
                    "p_max": float(np.max(p_l)),
                    **stats_l,
                }
            )

        expected_sources = 1 + len((layer_window or {}).get("layers", []))
        if len(source_stats) != expected_sources:
            drops["incomplete_pinned_aggregate"] += 1
            continue

        agg: Dict[str, float] = {}
        for name in ("fisher_eff_rank", "neg_shadow_logvol_r1", "spectral_entropy", "shadow_logvol_r1_raw", "p_max"):
            vals = np.asarray([float(s[name]) for s in source_stats], dtype=np.float64)
            if not np.isfinite(vals).all():
                drops["nonfinite_spectrum"] += 1
                continue
            agg[name] = float(np.mean(vals))
        if set(ALL_STATS) - set(agg):
            drops["nonfinite_spectrum"] += 1
            continue

        row = {
            "sample_idx": int(i),
            "label": int(label),
            "surprise": surprise,
            "p_max": float(np.max(p_t)),
            "p_max_late_readout_mean": agg["p_max"],
            "null_ratio_post_rank1": null_ratio,
            "fisher_eff_rank": agg["fisher_eff_rank"],
            "neg_shadow_logvol_r1": agg["neg_shadow_logvol_r1"],
            "spectral_entropy": agg["spectral_entropy"],
            "readout": {
                "p_max": float(np.max(p_t)),
                "fisher_eff_rank": readout_stats["fisher_eff_rank"],
                "neg_shadow_logvol_r1": readout_stats["neg_shadow_logvol_r1"],
                "spectral_entropy": readout_stats["spectral_entropy"],
                "shadow_logvol_r1_raw": readout_stats["shadow_logvol_r1_raw"],
            },
            "late_layers": layer_rows,
            "gen_token_id": int(gen_ids[0]),
            "random_rotation_max_abs_stat_delta": float(max(rot_deltas)),
        }
        if not all(math.isfinite(float(row[k])) for k in ("surprise", "p_max", "null_ratio_post_rank1") + ALL_STATS):
            drops["nonfinite_spectrum"] += 1
            continue
        rows.append(row)

        if not first_valid:
            first_valid = {
                "sample_idx": int(i),
                "gen_token_id": int(gen_ids[0]),
                "generated_text_prefix": str(trace.get("generated_text") or "")[:80],
                "readout_p_max": float(np.max(p_t)),
                "aggregate_stats": {k: row[k] for k in ALL_STATS},
                "random_rotation_max_abs_stat_delta": float(max(rot_deltas)),
            }
        print(f"[trace]   {i + 1}/{len(prompts)} processed; usable={len(rows)}", flush=True)

    diagnostics = {
        "data_hash_sha256": data_hash,
        "n_requested": int(len(prompts)),
        "n_usable": int(len(rows)),
        "model_dims": {"d_model": d_model, "vocab": vocab},
        "layer_indices": layer_indices,
        "final_layer_used": final_layer,
        "pinned_layer_window": layer_window,
        "k_support": int(k_support),
        "null_ratio_support_note": "rank-1 null-ratio baseline uses the inherited centered-Fisher core with v3_rank_values=[1].",
        "feature_locus": "shadow stats are mean over pinned late-window blocks plus readout; surprise/p_max/null_ratio are readout commit features",
        "first_valid_sample": first_valid,
    }
    try:
        del model, tokenizer, projection, readout_core
        pipeline.clear_mlx_cache()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass
    return FeatureRun(rows=rows, drops=drops, diagnostics=diagnostics)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores) & np.isfinite(y)
    if int(mask.sum()) < 4 or len(np.unique(y[mask])) < 2 or np.isclose(float(np.nanstd(scores[mask])), 0.0):
        return float("nan")
    yv = y[mask]
    sv = scores[mask]
    order = np.argsort(sv, kind="mergesort")
    sorted_scores = sv[order]
    ranks_sorted = np.empty(sorted_scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    pos = yv == 1
    n_pos = int(np.sum(pos))
    n_neg = int(yv.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(np.sum(ranks[pos]))
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _safe_pearson(x: np.ndarray, z: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    xm = x - float(np.mean(x))
    zm = z - float(np.mean(z))
    denom = math.sqrt(float(np.sum(xm * xm) * np.sum(zm * zm)))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(xm * zm) / denom)


def _feature_arrays(rows: List[Dict[str, Any]], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    if not rows:
        X = np.empty((0, len(names)), dtype=np.float64)
    else:
        X = np.asarray([[float(r[n]) for n in names] for r in rows], dtype=np.float64)
    return y, X


def _repeat_auc_summary(repeat_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = np.asarray(
        [float(item["auroc"]) for item in repeat_scores if item.get("auroc") is not None],
        dtype=np.float64,
    )
    if vals.size == 0:
        return {"n_repeats_used": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n_repeats_used": int(vals.size),
        "mean": float(np.mean(vals)),
        "sd": None if vals.size < 2 else float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def make_repeated_folds(
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int32)
    counts = np.bincount(y) if y.size else np.asarray([], dtype=np.int64)
    positive_min = int(counts[counts > 0].min()) if counts.size and np.any(counts > 0) else 0
    k = int(min(max(2, n_splits), positive_min)) if positive_min >= 2 else 0
    if k < 2 or len(np.unique(y)) < 2:
        return {"folds": [], "n_splits_effective": 0, "n_repeats": 0}
    folds: List[Dict[str, Any]] = []
    for rep in range(int(n_repeats)):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + 1009 * rep)
        for fold, (train, test) in enumerate(skf.split(np.zeros(len(y)), y)):
            folds.append({"repeat": rep, "fold": fold, "train": train, "test": test})
    return {"folds": folds, "n_splits_effective": k, "n_repeats": int(n_repeats)}


def cv_logit_oof_from_folds(y: np.ndarray, X: np.ndarray, fold_info: Dict[str, Any], seed: int) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int32)
    X = np.asarray(X, dtype=np.float64)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not valid.all():
        yv = y[valid]
        Xv = X[valid]
        index_map = np.where(valid)[0]
        local_fold_info = make_repeated_folds(
            yv,
            n_splits=int(fold_info.get("n_splits_effective") or 5),
            n_repeats=int(fold_info.get("n_repeats") or 10),
            seed=seed,
        )
    else:
        yv = y
        Xv = X
        index_map = np.arange(len(y))
        local_fold_info = fold_info

    if len(yv) < 4 or len(np.unique(yv)) < 2 or not local_fold_info.get("folds"):
        return {
            "scores": np.full(len(y), np.nan),
            "auroc": None,
            "n": int(len(yv)),
            "valid_mask": valid.tolist(),
            "repeat_scores": [],
            "repeat_aurocs": [],
            "repeat_auc_summary": _repeat_auc_summary([]),
        }

    pred_sum = np.zeros(len(yv), dtype=np.float64)
    pred_count = np.zeros(len(yv), dtype=np.int32)
    repeat_ids = sorted({int(item.get("repeat", 0)) for item in local_fold_info["folds"]})
    repeat_pred_sum = {rep: np.zeros(len(yv), dtype=np.float64) for rep in repeat_ids}
    repeat_pred_count = {rep: np.zeros(len(yv), dtype=np.int32) for rep in repeat_ids}
    skipped = 0
    for item in local_fold_info["folds"]:
        train = np.asarray(item["train"], dtype=np.int64)
        test = np.asarray(item["test"], dtype=np.int64)
        if len(np.unique(yv[train])) < 2:
            skipped += 1
            continue
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xv[train])
        Xh = scaler.transform(Xv[test])
        clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=seed)
        clf.fit(Xt, yv[train])
        pred = clf.predict_proba(Xh)[:, 1].astype(np.float64)
        pred_sum[test] += pred
        pred_count[test] += 1
        rep = int(item.get("repeat", 0))
        repeat_pred_sum[rep][test] += pred
        repeat_pred_count[rep][test] += 1
    oof_v = np.full(len(yv), np.nan, dtype=np.float64)
    ok = pred_count > 0
    oof_v[ok] = pred_sum[ok] / pred_count[ok]
    auc = _safe_auc(yv, oof_v)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    scores[index_map] = oof_v
    repeat_scores: List[Dict[str, Any]] = []
    repeat_aurocs: List[Dict[str, Any]] = []
    for rep in repeat_ids:
        rep_v = np.full(len(yv), np.nan, dtype=np.float64)
        rep_ok = repeat_pred_count[rep] > 0
        rep_v[rep_ok] = repeat_pred_sum[rep][rep_ok] / repeat_pred_count[rep][rep_ok]
        rep_scores_full = np.full(len(y), np.nan, dtype=np.float64)
        rep_scores_full[index_map] = rep_v
        rep_auc = _safe_auc(y, rep_scores_full)
        item = {
            "repeat": int(rep),
            "scores": rep_scores_full,
            "auroc": None if not math.isfinite(rep_auc) else float(rep_auc),
            "n": int(np.isfinite(rep_scores_full).sum()),
        }
        repeat_scores.append(item)
        repeat_aurocs.append({k: v for k, v in item.items() if k != "scores"})
    return {
        "scores": scores,
        "auroc": None if not math.isfinite(auc) else float(auc),
        "n": int(np.isfinite(oof_v).sum()),
        "valid_mask": valid.tolist(),
        "skipped_folds": int(skipped),
        "repeat_scores": repeat_scores,
        "repeat_aurocs": repeat_aurocs,
        "repeat_auc_summary": _repeat_auc_summary(repeat_scores),
        "oof_aggregation": "mean predicted probability across repeated out-of-fold assignments",
    }


def cv_locked_marginal_auc_from_folds(
    y: np.ndarray,
    scores: np.ndarray,
    fold_info: Dict[str, Any],
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores) & np.isfinite(y)
    if finite.sum() < 4 or len(np.unique(y[finite])) < 2 or not fold_info.get("folds"):
        return {"auroc": None, "n": int(finite.sum()), "fold_signs": []}
    sv = scores
    out_sum = np.zeros(len(y), dtype=np.float64)
    out_count = np.zeros(len(y), dtype=np.int32)
    signs: List[int] = []
    for item in fold_info["folds"]:
        train = np.asarray(item["train"], dtype=np.int64)
        test = np.asarray(item["test"], dtype=np.int64)
        train_auc = _safe_auc(y[train], sv[train])
        sign = 1 if (math.isfinite(train_auc) and train_auc >= 0.5) else -1
        signs.append(sign)
        out_sum[test] += sign * sv[test]
        out_count[test] += 1
    oof = np.full(len(y), np.nan, dtype=np.float64)
    ok = out_count > 0
    oof[ok] = out_sum[ok] / out_count[ok]
    auc = _safe_auc(y, oof)
    return {
        "auroc": None if not math.isfinite(auc) else float(auc),
        "n": int(np.isfinite(oof).sum()),
        "fold_signs": signs,
        "orientation_source": "train-fold AUROC sign only; no test-fold sign fitting",
    }


def paired_bootstrap_auc_diff(
    y: np.ndarray,
    base_scores: np.ndarray,
    aug_scores: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    mask = np.isfinite(base_scores) & np.isfinite(aug_scores) & np.isfinite(y)
    yv = y[mask].astype(np.int32)
    bv = base_scores[mask].astype(np.float64)
    av = aug_scores[mask].astype(np.float64)
    if len(yv) < 4 or len(np.unique(yv)) < 2:
        return {"diff": None, "ci_lo": None, "ci_hi": None, "p_one_sided_le_zero": None, "n_boot_used": 0, "n": int(len(yv))}
    base_auc = _safe_auc(yv, bv)
    aug_auc = _safe_auc(yv, av)
    diff = aug_auc - base_auc
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    n = len(yv)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(yv[idx])) < 2:
            continue
        db = _safe_auc(yv[idx], bv[idx])
        da = _safe_auc(yv[idx], av[idx])
        if math.isfinite(db) and math.isfinite(da):
            diffs.append(float(da - db))
    if not diffs:
        return {
            "base_auroc": None if not math.isfinite(base_auc) else float(base_auc),
            "augmented_auroc": None if not math.isfinite(aug_auc) else float(aug_auc),
            "diff": None if not math.isfinite(diff) else float(diff),
            "ci_lo": None,
            "ci_hi": None,
            "p_one_sided_le_zero": None,
            "n_boot_used": 0,
            "n": int(n),
        }
    arr = np.asarray(diffs, dtype=np.float64)
    p_le_zero = (1.0 + float(np.sum(arr <= 0.0))) / (1.0 + float(arr.size))
    return {
        "base_auroc": None if not math.isfinite(base_auc) else float(base_auc),
        "augmented_auroc": None if not math.isfinite(aug_auc) else float(aug_auc),
        "diff": None if not math.isfinite(diff) else float(diff),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "p_one_sided_le_zero": float(p_le_zero),
        "n_boot_used": int(arr.size),
        "n": int(n),
    }


def _repeat_diff_summary(y: np.ndarray, base: Dict[str, Any], aug: Dict[str, Any]) -> Dict[str, Any]:
    by_rep_base = {int(item["repeat"]): item for item in base.get("repeat_scores") or []}
    by_rep_aug = {int(item["repeat"]): item for item in aug.get("repeat_scores") or []}
    diffs: List[float] = []
    items: List[Dict[str, Any]] = []
    for rep in sorted(set(by_rep_base) & set(by_rep_aug)):
        b = np.asarray(by_rep_base[rep]["scores"], dtype=np.float64)
        a = np.asarray(by_rep_aug[rep]["scores"], dtype=np.float64)
        mask = np.isfinite(b) & np.isfinite(a) & np.isfinite(y)
        if int(mask.sum()) < 4 or len(np.unique(y[mask])) < 2:
            continue
        ba = _safe_auc(y[mask], b[mask])
        aa = _safe_auc(y[mask], a[mask])
        if not (math.isfinite(ba) and math.isfinite(aa)):
            continue
        diff = float(aa - ba)
        diffs.append(diff)
        items.append(
            {
                "repeat": int(rep),
                "base_auroc": float(ba),
                "augmented_auroc": float(aa),
                "diff": diff,
            }
        )
    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "n_repeats_used": int(arr.size),
        "mean_diff": None if arr.size == 0 else float(np.mean(arr)),
        "sd_diff": None if arr.size < 2 else float(np.std(arr, ddof=1)),
        "min_diff": None if arr.size == 0 else float(np.min(arr)),
        "max_diff": None if arr.size == 0 else float(np.max(arr)),
        "repeat_diffs": items,
        "note": "Diagnostic only: primary AUROC uses averaged repeated-OOF scores; this reports split-repeat instability.",
    }


def bootstrap_corr(x: np.ndarray, z: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(z)
    xv = x[mask]
    zv = z[mask]
    if len(xv) < 4 or np.isclose(float(np.std(xv)), 0.0) or np.isclose(float(np.std(zv)), 0.0):
        return {"pearson_r": None, "ci_lo": None, "ci_hi": None, "n": int(len(xv)), "n_boot_used": 0}

    def one(idx: np.ndarray) -> float:
        xx = xv[idx]
        zz = zv[idx]
        if np.isclose(float(np.std(xx)), 0.0) or np.isclose(float(np.std(zz)), 0.0):
            return float("nan")
        return _safe_pearson(xx, zz)

    idx0 = np.arange(len(xv))
    r0 = one(idx0)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(xv), size=len(xv))
        r = one(idx)
        if math.isfinite(r):
            vals.append(r)
    if not vals:
        return {"pearson_r": None if not math.isfinite(r0) else float(r0), "ci_lo": None, "ci_hi": None, "n": int(len(xv)), "n_boot_used": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "pearson_r": None if not math.isfinite(r0) else float(r0),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": int(len(xv)),
        "n_boot_used": int(arr.size),
    }


def partial_association(
    rows: List[Dict[str, Any]],
    stat: str,
    controls: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    y, X = _feature_arrays(rows, tuple(controls) + (stat,))
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    yv = y[mask].astype(np.float64)
    Xv = X[mask]
    if len(yv) < 4 or len(np.unique(yv)) < 2:
        return {"partial_pearson_r": None, "ci_lo": None, "ci_hi": None, "n": int(len(yv)), "controls": list(controls)}
    C = Xv[:, : len(controls)]
    s = Xv[:, len(controls)]

    def one(idx: np.ndarray) -> float:
        yy = yv[idx]
        ss = s[idx]
        CC = C[idx]
        if len(np.unique(yy)) < 2 or np.isclose(float(np.std(ss)), 0.0):
            return float("nan")
        design = np.column_stack([np.ones(len(idx), dtype=np.float64), CC])
        try:
            y_beta = np.linalg.lstsq(design, yy, rcond=None)[0]
            s_beta = np.linalg.lstsq(design, ss, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float("nan")
        y_res = yy - design @ y_beta
        s_res = ss - design @ s_beta
        if np.isclose(float(np.std(y_res)), 0.0) or np.isclose(float(np.std(s_res)), 0.0):
            return float("nan")
        return _safe_pearson(y_res, s_res)

    idx0 = np.arange(len(yv))
    r0 = one(idx0)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(yv), size=len(yv))
        r = one(idx)
        if math.isfinite(r):
            vals.append(r)
    if not vals:
        return {
            "partial_pearson_r": None if not math.isfinite(r0) else float(r0),
            "ci_lo": None,
            "ci_hi": None,
            "n": int(len(yv)),
            "n_boot_used": 0,
            "controls": list(controls),
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "partial_pearson_r": None if not math.isfinite(r0) else float(r0),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": int(len(yv)),
        "n_boot_used": int(arr.size),
        "controls": list(controls),
    }


def _fitset_scores(
    cache: Dict[Tuple[str, ...], Dict[str, Any]],
    rows: List[Dict[str, Any]],
    features: Sequence[str],
    fold_info: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    key = tuple(features)
    if key not in cache:
        y, X = _feature_arrays(rows, features)
        cache[key] = cv_logit_oof_from_folds(y, X, fold_info, seed)
        cache[key]["features"] = list(features)
    return cache[key]


def _increment(
    y: np.ndarray,
    base: Dict[str, Any],
    aug: Dict[str, Any],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    diff = paired_bootstrap_auc_diff(y, base["scores"], aug["scores"], n_boot=n_boot, seed=seed)
    return {
        "base_features": base.get("features"),
        "augmented_features": aug.get("features"),
        "base_auroc": base.get("auroc"),
        "augmented_auroc": aug.get("auroc"),
        "cv_repeat_diff_summary": _repeat_diff_summary(y, base, aug),
        **diff,
    }


def _label_permutation_controls(
    rows: List[Dict[str, Any]],
    *,
    n_permutations: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    y_real = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    if len(y_real) < 4 or len(np.unique(y_real)) < 2:
        return {
            "n_permutations": 0,
            "minimum_attainable_empirical_p": None,
            "primary_p_empirical": None,
            "max_stat_diffs": [],
        }
    stat_diffs: Dict[str, List[float]] = {s: [] for s in ALL_STATS}
    max_diffs: List[float] = []
    rng = np.random.default_rng(seed)
    feature_rows = [dict(r) for r in rows]
    for j in range(int(n_permutations)):
        perm_y = rng.permutation(y_real)
        for idx, yy in enumerate(perm_y):
            feature_rows[idx]["label"] = int(yy)
        fold_info = make_repeated_folds(perm_y, n_splits=n_splits, n_repeats=n_repeats, seed=seed + 7919 * (j + 1))
        cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        base = _fitset_scores(cache, feature_rows, BASE_SURPRISE, fold_info, seed + j)
        per_perm: List[float] = []
        for stat in ALL_STATS:
            aug = _fitset_scores(cache, feature_rows, BASE_SURPRISE + (stat,), fold_info, seed + 13 * j)
            base_auc = base.get("auroc")
            aug_auc = aug.get("auroc")
            diff = float("nan") if base_auc is None or aug_auc is None else float(aug_auc) - float(base_auc)
            if math.isfinite(diff):
                stat_diffs[stat].append(diff)
                per_perm.append(diff)
        if per_perm:
            max_diffs.append(float(max(per_perm)))
    return {
        "n_permutations": int(n_permutations),
        "minimum_attainable_empirical_p": float(1.0 / (1.0 + max(0, int(n_permutations)))),
        "stat_diffs_over_surprise": {k: [float(x) for x in v] for k, v in stat_diffs.items()},
        "max_stat_diffs_over_surprise": max_diffs,
        "note": "Empirical null from shuffled labels; family max over carried stats.",
    }


def analyze_rows(
    rows: List[Dict[str, Any]],
    *,
    n_splits: int,
    n_repeats: int,
    n_boot: int,
    n_permutations: int,
    seed: int,
    label_perm_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    fold_info = make_repeated_folds(y, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    base_surprise = _fitset_scores(cache, rows, BASE_SURPRISE, fold_info, seed)
    base_sn = _fitset_scores(cache, rows, BASE_SURPRISE_NULL, fold_info, seed)
    base_full = _fitset_scores(cache, rows, BASE_FULL, fold_info, seed)
    base_conf = _fitset_scores(cache, rows, BASE_CONFIDENCE, fold_info, seed)
    pmax_aug = _fitset_scores(cache, rows, BASE_SURPRISE + ("p_max",), fold_info, seed)

    marginal: Dict[str, Any] = {}
    for name in ("surprise", "p_max", "null_ratio_post_rank1") + ALL_STATS:
        scores = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
        marginal[name] = cv_locked_marginal_auc_from_folds(y, scores, fold_info)

    incremental: Dict[str, Any] = {}
    for i, stat in enumerate(ALL_STATS):
        aug_surprise = _fitset_scores(cache, rows, BASE_SURPRISE + (stat,), fold_info, seed)
        aug_sn = _fitset_scores(cache, rows, BASE_SURPRISE_NULL + (stat,), fold_info, seed)
        aug_full = _fitset_scores(cache, rows, BASE_FULL + (stat,), fold_info, seed)
        incremental[stat] = {
            "over_surprise": _increment(y, base_surprise, aug_surprise, n_boot=n_boot, seed=seed + 101 + i),
            "over_surprise_null_ratio": _increment(y, base_sn, aug_sn, n_boot=n_boot, seed=seed + 211 + i),
            "over_surprise_null_ratio_pmax": _increment(y, base_full, aug_full, n_boot=n_boot, seed=seed + 307 + i),
        }

    partial = {
        stat: {
            "controlling_surprise": partial_association(rows, stat, BASE_SURPRISE, n_boot=n_boot, seed=seed + 401 + j),
            "controlling_surprise_null_ratio": partial_association(rows, stat, BASE_SURPRISE_NULL, n_boot=n_boot, seed=seed + 503 + j),
            "controlling_surprise_null_ratio_pmax": partial_association(rows, stat, BASE_FULL, n_boot=n_boot, seed=seed + 601 + j),
        }
        for j, stat in enumerate(ALL_STATS)
    }
    brittleness = {
        stat: {
            "vs_p_max": bootstrap_corr(
                np.asarray([float(r[stat]) for r in rows]),
                np.asarray([float(r["p_max"]) for r in rows]),
                n_boot=n_boot,
                seed=seed + 701 + j,
            ),
            "vs_surprise": bootstrap_corr(
                np.asarray([float(r[stat]) for r in rows]),
                np.asarray([float(r["surprise"]) for r in rows]),
                n_boot=n_boot,
                seed=seed + 809 + j,
            ),
        }
        for j, stat in enumerate(ALL_STATS)
    }
    if label_perm_override is None:
        label_perm = _label_permutation_controls(
            rows,
            n_permutations=n_permutations,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed + 9001,
        )
    else:
        label_perm = dict(label_perm_override)
        label_perm["n_permutations"] = int(n_permutations)
        label_perm["reused_from_prior_analysis"] = True
    primary_obs = incremental[PRIMARY_STAT]["over_surprise"].get("diff")
    max_null = np.asarray(label_perm.get("max_stat_diffs_over_surprise") or [], dtype=np.float64)
    empirical_fwer = None
    if primary_obs is not None and math.isfinite(float(primary_obs)) and max_null.size:
        empirical_fwer = float((1.0 + np.sum(max_null >= float(primary_obs))) / (1.0 + max_null.size))

    base_sn_auc = base_sn.get("auroc")
    base_surprise_auc = base_surprise.get("auroc")
    degraded = None
    if base_sn_auc is not None and base_surprise_auc is not None:
        degraded = bool(float(base_sn_auc) < float(base_surprise_auc))

    rot_deltas = np.asarray([float(r["random_rotation_max_abs_stat_delta"]) for r in rows], dtype=np.float64)
    return {
        "cv": {
            "requested_n_splits": int(n_splits),
            "effective_n_splits": int(fold_info.get("n_splits_effective") or 0),
            "n_repeats": int(fold_info.get("n_repeats") or 0),
            "same_fold_assignments_for_all_feature_sets": True,
            "identical_feature_sets_share_cached_oof_scores": True,
            "oof_repeat_aggregation": "mean predicted probability across repeats before AUROC/bootstrap",
            "fixed_logistic": {"solver": "liblinear", "C": 1.0, "max_iter": 1000},
        },
        "orientation": {
            "fisher_eff_rank": "pre_registered_positive_direction_higher_to_contradiction",
            "neg_shadow_logvol_r1": "pre_registered_sign_flip_of_shadow_logvol_r1_higher_to_contradiction",
            "spectral_entropy": "pre_registered_positive_direction_higher_to_contradiction",
            "marginal_sign_source": "train folds only",
            "assert_not_learned_from_test": True,
        },
        "base_models": {
            "surprise": {
                "features": list(BASE_SURPRISE),
                "auroc": base_surprise.get("auroc"),
                "repeat_auc_summary": base_surprise.get("repeat_auc_summary"),
            },
            "surprise_null_ratio": {
                "features": list(BASE_SURPRISE_NULL),
                "auroc": base_sn.get("auroc"),
                "repeat_auc_summary": base_sn.get("repeat_auc_summary"),
            },
            "surprise_null_ratio_pmax": {
                "features": list(BASE_FULL),
                "auroc": base_full.get("auroc"),
                "repeat_auc_summary": base_full.get("repeat_auc_summary"),
            },
            "surprise_pmax": {
                "features": list(BASE_CONFIDENCE),
                "auroc": base_conf.get("auroc"),
                "repeat_auc_summary": base_conf.get("repeat_auc_summary"),
            },
        },
        "degraded_base_flag": {
            "base_surprise_null_ratio_below_base_surprise": degraded,
            "base_surprise_auroc": base_surprise_auc,
            "base_surprise_null_ratio_auroc": base_sn_auc,
        },
        "marginal_train_locked_auroc": marginal,
        "incremental_logistic_repeated_cv": incremental,
        "primary_endpoint": {
            "stat": PRIMARY_STAT,
            "base": list(BASE_SURPRISE),
            **incremental[PRIMARY_STAT]["over_surprise"],
            "minimum_practical_effect": 0.02,
            "empirical_familywise_p_from_shuffled_max_stat": empirical_fwer,
        },
        "partial_association": partial,
        "brittleness": brittleness,
        "controls": {
            "shuffled_label_permutation": label_perm,
            "temperature_matched_confidence": {
                "operationalization": "confidence-only p_max augmentation and required primary increment after p_max/null_ratio controls",
                "p_max_over_surprise": _increment(y, base_surprise, pmax_aug, n_boot=n_boot, seed=seed + 10007),
                "base_surprise_pmax_auroc": base_conf.get("auroc"),
                "primary_over_surprise_null_ratio_pmax": incremental[PRIMARY_STAT]["over_surprise_null_ratio_pmax"],
            },
            "random_rotation_invariance": {
                "orthogonal_transform": "deterministic random Householder reflection in hidden coordinates",
                "max_abs_stat_delta": None if rot_deltas.size == 0 else float(np.max(rot_deltas)),
                "mean_abs_stat_delta": None if rot_deltas.size == 0 else float(np.mean(rot_deltas)),
                "expected": "near zero for spectral statistics",
            },
        },
        "multiplicity": {
            "one_uncorrected_primary": True,
            "primary_spec": "fisher_eff_rank x pinned late-window-plus-readout aggregate x base{surprise} x random-effects meta-rule",
            "secondary_stats": list(SECONDARY_STATS),
            "per_pair_secondary_test_count": int((len(ALL_STATS) * 3) - 1),
        },
    }


def model_family(model_id: str) -> str:
    low = model_id.lower()
    if "qwen" in low:
        return "qwen"
    if "llama" in low:
        return "llama"
    if "mistral" in low:
        return "mistral"
    if "gemma" in low:
        return "gemma"
    if "phi" in low:
        return "phi"
    if "deepseek" in low:
        return "deepseek"
    if "gpt-oss" in low:
        return "gpt-oss"
    return "other"


def environment_report() -> Dict[str, Any]:
    return {
        "timestamp_utc": _now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(REPO),
    }


def provenance(data_path: Path) -> Dict[str, Any]:
    files = [
        HERE / "PRE_REGISTRATION_DRAFT.md",
        HERE / "comprehensive_run.py",
        HERE / "test_shadow_ambiguity.py",
        REPO / "pri_runtime.py",
        REPO / "pri_calibrator.py",
        REPO / "pri_v2_io_plugins.py",
    ]
    return {
        "fresh_seed": FRESH_SEED,
        "pilot_seed_not_reused": 20260607,
        "data_file_sha256": _file_sha256(data_path),
        "code_hashes_sha256": {_rel(p): _file_sha256(p) for p in files},
    }


def run_one(args: argparse.Namespace) -> Path:
    models = discover_models()
    benchmarks = discover_benchmarks()
    print("[inventory] discovered_models:")
    for m in models:
        suffix = ""
        if m in KNOWN_SKIP_HINTS:
            suffix = f"  [known-skip: {KNOWN_SKIP_HINTS[m]}]"
        elif m in KNOWN_GATE_RISKS:
            suffix = f"  [gate-risk: {KNOWN_GATE_RISKS[m]}]"
        print(f"  - {m}{suffix}")
    print("[inventory] discovered_benchmarks:")
    for name, path in benchmarks.items():
        print(f"  - {name}: {_rel(path)}")

    if args.model not in models:
        print(f"[coverage] requested model is not in mlx-community cache inventory: {args.model}")
    if args.benchmark not in benchmarks:
        raise SystemExit(f"benchmark not found: {args.benchmark}; available={sorted(benchmarks)}")

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"shadow_v2__{_slug(args.model)}__{args.benchmark}__limit{args.limit or 'all'}.json"
    coverage: Dict[str, Any] = {
        "discovered_models": models,
        "discovered_benchmarks": {k: _rel(v) for k, v in benchmarks.items()},
        "requested": {"model": args.model, "benchmark": args.benchmark, "limit": args.limit},
        "known_skip_hints": KNOWN_SKIP_HINTS,
        "known_gate_risks": KNOWN_GATE_RISKS,
        "attempt_status": "started",
        "skip_reason": None,
    }

    if args.model in KNOWN_SKIP_HINTS:
        coverage["attempt_status"] = "skipped_before_load"
        coverage["skip_reason"] = KNOWN_SKIP_HINTS[args.model]
        payload = {
            "schema": SCHEMA,
            "model": args.model,
            "model_family": model_family(args.model),
            "benchmark": args.benchmark,
            "data_path": _rel(benchmarks[args.benchmark]),
            "coverage": coverage,
            "environment": environment_report(),
            "provenance": provenance(benchmarks[args.benchmark]),
        }
        _atomic_write_json(out_path, payload)
        print(f"[coverage] skipped {args.model}: {coverage['skip_reason']}")
        print(f"[write] {out_path}")
        return out_path

    try:
        feature_run = trace_pair_features(
            args.model,
            args.benchmark,
            benchmarks[args.benchmark],
            limit=int(args.limit or 0),
            max_new_tokens=int(args.max_new_tokens),
            k_support=int(args.k_support),
            seed=int(args.seed),
        )
        analysis = analyze_rows(
            feature_run.rows,
            n_splits=int(args.n_splits),
            n_repeats=int(args.n_repeats),
            n_boot=int(args.n_boot),
            n_permutations=int(args.n_permutations),
            seed=int(args.seed),
        )
        coverage["attempt_status"] = "completed"
        coverage["drops"] = feature_run.drops
        coverage["nonfinite_drop_count"] = int(sum(v for k, v in feature_run.drops.items() if "nonfinite" in k))
        coverage["silent_drop_count"] = 0
        coverage["n_usable"] = len(feature_run.rows)
        payload = {
            "schema": SCHEMA,
            "model": args.model,
            "model_family": model_family(args.model),
            "benchmark": args.benchmark,
            "data_path": _rel(benchmarks[args.benchmark]),
            "rows": feature_run.rows,
            "diagnostics": feature_run.diagnostics,
            "analysis": analysis,
            "coverage": coverage,
            "environment": environment_report(),
            "provenance": provenance(benchmarks[args.benchmark]),
        }
    except Exception as exc:  # noqa: BLE001
        coverage["attempt_status"] = "failed"
        coverage["skip_reason"] = str(exc)
        payload = {
            "schema": SCHEMA,
            "model": args.model,
            "model_family": model_family(args.model),
            "benchmark": args.benchmark,
            "data_path": _rel(benchmarks[args.benchmark]),
            "coverage": coverage,
            "environment": environment_report(),
            "provenance": provenance(benchmarks[args.benchmark]),
        }
        _atomic_write_json(out_path, payload)
        print(f"[coverage] failed {args.model} {args.benchmark}: {exc}")
        print(f"[write] {out_path}")
        raise

    _atomic_write_json(out_path, payload)
    print(f"[analysis] primary diff={payload['analysis']['primary_endpoint'].get('diff')} ci=[{payload['analysis']['primary_endpoint'].get('ci_lo')}, {payload['analysis']['primary_endpoint'].get('ci_hi')}]")
    print(f"[coverage] drops={feature_run.drops}")
    print(f"[coverage] nonfinite_drop_count={coverage['nonfinite_drop_count']}")
    if coverage["nonfinite_drop_count"] == 0:
        print("[coverage] nonfinite warnings: zero")
    print(f"[write] {out_path}")
    return out_path


def _normal_p_from_z(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _endpoint_se(endpoint: Dict[str, Any]) -> Optional[float]:
    lo = endpoint.get("ci_lo")
    hi = endpoint.get("ci_hi")
    if lo is None or hi is None:
        return None
    se = (float(hi) - float(lo)) / (2.0 * 1.96)
    if math.isfinite(se) and se > META_MIN_ENDPOINT_SE:
        return float(se)
    repeat_summary = endpoint.get("cv_repeat_diff_summary") or {}
    repeat_sd = repeat_summary.get("sd_diff")
    n_repeats = repeat_summary.get("n_repeats_used")
    if repeat_sd is not None and n_repeats is not None:
        repeat_se = float(repeat_sd) / math.sqrt(max(1.0, float(n_repeats)))
        if math.isfinite(repeat_se) and repeat_se > META_MIN_ENDPOINT_SE:
            return float(repeat_se)
    if se <= 0 or not math.isfinite(se):
        return None
    return None


def _endpoint_se_source(endpoint: Dict[str, Any]) -> Optional[str]:
    lo = endpoint.get("ci_lo")
    hi = endpoint.get("ci_hi")
    if lo is None or hi is None:
        return None
    se = (float(hi) - float(lo)) / (2.0 * 1.96)
    if math.isfinite(se) and se > META_MIN_ENDPOINT_SE:
        return "bootstrap_ci"
    repeat_summary = endpoint.get("cv_repeat_diff_summary") or {}
    repeat_sd = repeat_summary.get("sd_diff")
    n_repeats = repeat_summary.get("n_repeats_used")
    if repeat_sd is not None and n_repeats is not None:
        repeat_se = float(repeat_sd) / math.sqrt(max(1.0, float(n_repeats)))
        if math.isfinite(repeat_se) and repeat_se > META_MIN_ENDPOINT_SE:
            return "repeat_diff_summary"
    return None


def random_effects_meta(effects: Sequence[float], ses: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(effects, dtype=np.float64)
    se = np.asarray(ses, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y = y[mask]
    se = se[mask]
    if y.size == 0:
        return {
            "k": 0,
            "mean": None,
            "ci_lo": None,
            "ci_hi": None,
            "tau2": None,
            "p_one_sided_le_zero": None,
            "ci_method": None,
        }
    v = se * se
    w = 1.0 / v
    fixed = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - fixed) ** 2))
    c = float(np.sum(w) - (np.sum(w * w) / np.sum(w)))
    tau2 = max(0.0, (q - (y.size - 1.0)) / c) if y.size > 1 and c > 0 else 0.0
    wr = 1.0 / (v + tau2)
    mean = float(np.sum(wr * y) / np.sum(wr))
    se_mean = math.sqrt(float(1.0 / np.sum(wr)))
    z = mean / se_mean if se_mean > 0 else float("nan")
    normal_ci = {
        "ci_lo": float(mean - 1.96 * se_mean),
        "ci_hi": float(mean + 1.96 * se_mean),
        "p_one_sided_le_zero": None if not math.isfinite(z) else float(_normal_p_from_z(z)),
    }
    if y.size == 1:
        return {
            "k": 1,
            "mean": mean,
            "se": se_mean,
            "ci_lo": None,
            "ci_hi": None,
            "tau2": float(tau2),
            "p_one_sided_le_zero": None,
            "ci_method": "insufficient_k_for_random_effects_ci",
            "normal_dersimonian_laird": {"se": se_mean, **normal_ci},
            "modified_knapp_hartung_t": {"available": False},
            "small_k_rule": "k=1 is reported without a selected meta CI; use modified Knapp-Hartung/t when 2 <= k < 10",
        }
    hksj: Dict[str, Any] = {"available": False}
    if y.size >= 2 and student_t is not None:
        q_re = float(np.sum(wr * (y - mean) ** 2) / max(1, y.size - 1))
        scale = max(1.0, q_re)
        se_hksj = math.sqrt(float(scale / np.sum(wr)))
        df = int(y.size - 1)
        crit = float(student_t.ppf(0.975, df))
        t_stat = mean / se_hksj if se_hksj > 0 else float("nan")
        hksj = {
            "available": True,
            "df": df,
            "q_scale": float(q_re),
            "modified_scale_used": float(scale),
            "se": float(se_hksj),
            "ci_lo": float(mean - crit * se_hksj),
            "ci_hi": float(mean + crit * se_hksj),
            "p_one_sided_le_zero": None
            if not math.isfinite(t_stat)
            else float(student_t.sf(t_stat, df)),
            "method": "modified_knapp_hartung_t",
        }
    use_hksj = bool(y.size < 10 and hksj.get("available"))
    selected = hksj if use_hksj else normal_ci
    return {
        "k": int(y.size),
        "mean": mean,
        "se": se_mean,
        "ci_lo": selected.get("ci_lo"),
        "ci_hi": selected.get("ci_hi"),
        "tau2": float(tau2),
        "p_one_sided_le_zero": selected.get("p_one_sided_le_zero"),
        "ci_method": "modified_knapp_hartung_t" if use_hksj else "normal_dersimonian_laird",
        "normal_dersimonian_laird": {"se": se_mean, **normal_ci},
        "modified_knapp_hartung_t": hksj,
        "small_k_rule": "use modified Knapp-Hartung/t interval when k < 10; otherwise report normal DL as selected",
    }


def holm(pvals: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    clean = [(name, float(p)) for name, p in pvals if p is not None and math.isfinite(float(p))]
    ordered = sorted(clean, key=lambda x: x[1])
    m = len(ordered)
    out = []
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out.append({"test": name, "p": p, "holm_p": adj})
    return out


def _analysis_int(analysis: Dict[str, Any], path: Tuple[str, ...], default: int) -> int:
    obj: Any = analysis
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return int(default)
        obj = obj[key]
    try:
        return int(obj)
    except (TypeError, ValueError):
        return int(default)


def _reanalyze_path(path: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text())
    if data.get("schema") != SCHEMA:
        return {"path": path, "status": "skipped", "reason": "schema mismatch"}
    rows = data.get("rows") or []
    if not rows:
        return {"path": path, "status": "skipped", "reason": "no stored rows"}
    analysis = data.get("analysis") or {}
    n_splits = int(opts.get("n_splits") or _analysis_int(analysis, ("cv", "requested_n_splits"), 5))
    n_repeats = int(opts.get("n_repeats") or _analysis_int(analysis, ("cv", "n_repeats"), 10))
    n_permutations = int(
        opts["n_permutations"]
        if opts.get("n_permutations") is not None
        else _analysis_int(analysis, ("controls", "shuffled_label_permutation", "n_permutations"), 1000)
    )
    seed = int(opts.get("seed") or data.get("provenance", {}).get("fresh_seed") or FRESH_SEED)
    prior_label_perm = (analysis.get("controls") or {}).get("shuffled_label_permutation")
    reuse_label_perm = (
        opts.get("n_permutations") is None
        and isinstance(prior_label_perm, dict)
        and int(prior_label_perm.get("n_permutations") or -1) == n_permutations
    )
    new_analysis = analyze_rows(
        rows,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_boot=int(opts["n_boot"]),
        n_permutations=n_permutations,
        seed=seed,
        label_perm_override=prior_label_perm if reuse_label_perm else None,
    )
    data["analysis"] = new_analysis
    data["reanalysis"] = {
        "timestamp_utc": _now(),
        "mode": "stored_rows_only_no_model_trace",
        "rows_reused": int(len(rows)),
        "n_boot": int(opts["n_boot"]),
        "n_permutations": n_permutations,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": seed,
        "permutation_controls": "reused_from_prior_analysis" if reuse_label_perm else "recomputed",
        "previous_analysis_replaced": True,
    }
    _atomic_write_json(p, data)
    return {
        "path": path,
        "status": "rewritten",
        "rows": int(len(rows)),
        "n_boot": int(opts["n_boot"]),
        "n_permutations": n_permutations,
    }


def run_reanalyze(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    paths = sorted(str(p) for p in in_dir.glob("shadow_v2__*.json"))
    opts = {
        "n_boot": int(args.n_boot),
        "n_permutations": args.n_permutations,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
    }
    rewritten = 0
    skipped = 0
    jobs = max(1, int(args.jobs))
    if jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_reanalyze_path, path, opts) for path in paths]
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                name = Path(result["path"]).name
                if result["status"] == "rewritten":
                    rewritten += 1
                    print(
                        f"[reanalyze] wrote {name}: rows={result['rows']} "
                        f"n_boot={result['n_boot']} n_perm={result['n_permutations']}",
                        flush=True,
                    )
                else:
                    skipped += 1
                    print(f"[reanalyze] skip {name}: {result['reason']}", flush=True)
        print(f"[reanalyze] rewritten={rewritten} skipped={skipped}", flush=True)
        return

    for path in paths:
        result = _reanalyze_path(path, opts)
        p = Path(result["path"])
        if result["status"] != "rewritten":
            skipped += 1
            print(f"[reanalyze] skip {p.name}: {result['reason']}", flush=True)
            continue
        rewritten += 1
        print(
            f"[reanalyze] wrote {p.name}: rows={result['rows']} "
            f"n_boot={result['n_boot']} n_perm={result['n_permutations']}",
            flush=True,
        )
    print(f"[reanalyze] rewritten={rewritten} skipped={skipped}", flush=True)


def run_meta(args: argparse.Namespace) -> Path:
    in_dir = Path(args.in_dir)
    paths = sorted(in_dir.glob("shadow_v2__*.json"))
    records: List[Dict[str, Any]] = []
    for p in paths:
        data = json.loads(p.read_text())
        if data.get("schema") != SCHEMA:
            continue
        if data.get("coverage", {}).get("attempt_status") != "completed":
            continue
        records.append(data)

    primary_effects: List[float] = []
    primary_ses: List[float] = []
    primary_base_nr_effects: List[float] = []
    primary_base_nr_ses: List[float] = []
    primary_base_b_effects: List[float] = []
    primary_base_b_ses: List[float] = []
    pvals: List[Tuple[str, float]] = []
    per_pair: List[Dict[str, Any]] = []
    for rec in records:
        inc = rec.get("analysis", {}).get("incremental_logistic_repeated_cv", {})
        base_nr_ep = inc.get(PRIMARY_STAT, {}).get("over_surprise_null_ratio", {})
        base_nr_diff = base_nr_ep.get("diff")
        base_nr_se = _endpoint_se(base_nr_ep)
        if base_nr_diff is not None and base_nr_se is not None:
            primary_base_nr_effects.append(float(base_nr_diff))
            primary_base_nr_ses.append(float(base_nr_se))
        base_b_ep = inc.get(PRIMARY_STAT, {}).get("over_surprise_null_ratio_pmax", {})
        base_b_diff = base_b_ep.get("diff")
        base_b_se = _endpoint_se(base_b_ep)
        if base_b_diff is not None and base_b_se is not None:
            primary_base_b_effects.append(float(base_b_diff))
            primary_base_b_ses.append(float(base_b_se))
        ep = rec.get("analysis", {}).get("primary_endpoint", {})
        diff = ep.get("diff")
        se = _endpoint_se(ep)
        if diff is not None and se is not None:
            primary_effects.append(float(diff))
            primary_ses.append(float(se))
            name = f"{rec['model']}::{rec['benchmark']}::{PRIMARY_STAT}::over_surprise"
            p = ep.get("p_one_sided_le_zero")
            if p is not None:
                pvals.append((name, float(p)))
            per_pair.append(
                {
                    "model": rec["model"],
                    "family": rec.get("model_family"),
                    "benchmark": rec["benchmark"],
                    "primary_diff": float(diff),
                    "primary_se": float(se),
                    "primary_ci": [ep.get("ci_lo"), ep.get("ci_hi")],
                    "primary_se_source": _endpoint_se_source(ep),
                    "primary_base_null_ratio_diff": None if base_nr_diff is None else float(base_nr_diff),
                    "primary_base_null_ratio_se": None if base_nr_se is None else float(base_nr_se),
                    "primary_base_null_ratio_se_source": _endpoint_se_source(base_nr_ep),
                    "primary_base_null_ratio_ci": [base_nr_ep.get("ci_lo"), base_nr_ep.get("ci_hi")],
                    "primary_base_b_diff": None if base_b_diff is None else float(base_b_diff),
                    "primary_base_b_se": None if base_b_se is None else float(base_b_se),
                    "primary_base_b_se_source": _endpoint_se_source(base_b_ep),
                    "primary_base_b_ci": [base_b_ep.get("ci_lo"), base_b_ep.get("ci_hi")],
                    "base_surprise": rec.get("analysis", {}).get("base_models", {}).get("surprise", {}).get("auroc"),
                    "base_surprise_null_ratio": rec.get("analysis", {}).get("base_models", {}).get("surprise_null_ratio", {}).get("auroc"),
                    "base_surprise_null_ratio_pmax": rec.get("analysis", {}).get("base_models", {}).get("surprise_null_ratio_pmax", {}).get("auroc"),
                    "brittleness": rec.get("analysis", {}).get("brittleness", {}).get(PRIMARY_STAT),
                }
            )

        for stat, stat_obj in inc.items():
            for endpoint_name, endpoint in stat_obj.items():
                if stat == PRIMARY_STAT and endpoint_name == "over_surprise":
                    continue
                p = endpoint.get("p_one_sided_le_zero")
                if p is not None:
                    pvals.append((f"{rec['model']}::{rec['benchmark']}::{stat}::{endpoint_name}", float(p)))

    meta_primary = random_effects_meta(primary_effects, primary_ses)
    meta_primary_base_nr = random_effects_meta(primary_base_nr_effects, primary_base_nr_ses)
    meta_primary_base_b = random_effects_meta(primary_base_b_effects, primary_base_b_ses)

    # H2: weighted linear interaction of increment over {surprise,null_ratio}
    # against null-ratio weakness, defined from null-ratio's own marginal AUROC.
    xs: List[float] = []
    ys: List[float] = []
    ws: List[float] = []
    for rec in records:
        endpoint = rec.get("analysis", {}).get("incremental_logistic_repeated_cv", {}).get(PRIMARY_STAT, {}).get("over_surprise_null_ratio", {})
        diff = endpoint.get("diff")
        se = _endpoint_se(endpoint)
        marg_null = (
            rec.get("analysis", {})
            .get("marginal_train_locked_auroc", {})
            .get("null_ratio_post_rank1", {})
            .get("auroc")
        )
        if diff is None or se is None or marg_null is None:
            continue
        # null-ratio weakness from its OWN train-locked marginal power, not the
        # {surprise, null_ratio} combined base (which conflates surprise strength).
        xs.append(float(1.0 - float(marg_null)))
        ys.append(float(diff))
        ws.append(float(1.0 / (se * se)))
    h2: Dict[str, Any]
    if len(xs) >= 3:
        X = np.column_stack([np.ones(len(xs)), np.asarray(xs)])
        W = np.diag(np.asarray(ws))
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ np.asarray(ys))
        h2 = {
            "n": int(len(xs)),
            "predictor": "null_ratio_weakness = 1 - AUROC(marginal null_ratio_post_rank1, train-locked)",
            "weighted_intercept": float(beta[0]),
            "weighted_slope": float(beta[1]),
        }
    else:
        h2 = {"n": int(len(xs)), "status": "insufficient completed pairs for interaction"}

    positive_pairs = [
        p
        for p in per_pair
        if p["primary_diff"] >= 0.02
        and p["primary_ci"][0] is not None
        and float(p["primary_ci"][0]) > 0.0
        and p["primary_base_b_diff"] is not None
        and float(p["primary_base_b_diff"]) >= 0.02
        and p["primary_base_b_ci"][0] is not None
        and float(p["primary_base_b_ci"][0]) > 0.0
    ]
    positive_families = sorted({str(p["family"]) for p in positive_pairs})
    family_verdict = "null_or_underpowered"
    if len(positive_pairs) >= 2 and len(positive_families) >= 2:
        family_verdict = "eligible_for_general_claim"
    elif positive_pairs and set(positive_families) == {"qwen"}:
        family_verdict = "qwen_family_null_ratio_rescue_only"

    brittle_fail = False
    for p in per_pair:
        b = p.get("brittleness") or {}
        for key in ("vs_p_max", "vs_surprise"):
            hi = (b.get(key) or {}).get("ci_hi")
            if hi is not None and float(hi) >= 0.75:
                brittle_fail = True
    primary_a_ci_excludes_zero = bool(
        meta_primary.get("ci_lo") is not None and float(meta_primary["ci_lo"]) > 0.0
    )
    primary_b_ci_excludes_zero = bool(
        meta_primary_base_b.get("ci_lo") is not None and float(meta_primary_base_b["ci_lo"]) > 0.0
    )
    primary_a_min_effect = bool(meta_primary.get("mean") is not None and float(meta_primary["mean"]) >= 0.02)
    primary_b_min_effect = bool(
        meta_primary_base_b.get("mean") is not None and float(meta_primary_base_b["mean"]) >= 0.02
    )
    h1_passes_registered_gates = bool(
        primary_a_ci_excludes_zero
        and primary_b_ci_excludes_zero
        and primary_a_min_effect
        and primary_b_min_effect
        and not brittle_fail
    )
    if h1_passes_registered_gates and family_verdict == "eligible_for_general_claim":
        final_claim_verdict = "go_general_claim"
    elif h1_passes_registered_gates and family_verdict == "qwen_family_null_ratio_rescue_only":
        final_claim_verdict = "go_qwen_family_only"
    else:
        final_claim_verdict = "no_go_registered_claim"

    payload = {
        "schema": META_SCHEMA,
        "input_dir": str(in_dir),
        "n_completed_pairs": int(len(records)),
        "primary_random_effects": meta_primary,
        "primary_base_null_ratio_random_effects": {
            "base": list(BASE_SURPRISE_NULL),
            "stat": PRIMARY_STAT,
            **meta_primary_base_nr,
        },
        "primary_base_b_random_effects": {
            "base": list(BASE_FULL),
            "stat": PRIMARY_STAT,
            **meta_primary_base_b,
        },
        "primary_min_practical_effect": 0.02,
        "primary_passes_min_effect": bool(primary_a_min_effect),
        "primary_ci_excludes_zero": bool(primary_a_ci_excludes_zero and primary_b_ci_excludes_zero),
        "primary_brittleness_any_upper_ci_ge_0_75": brittle_fail,
        "h1_gate_summary": {
            "base_a": {
                "base": list(BASE_SURPRISE),
                "ci_excludes_zero": primary_a_ci_excludes_zero,
                "passes_min_effect": primary_a_min_effect,
            },
            "base_b": {
                "base": list(BASE_FULL),
                "ci_excludes_zero": primary_b_ci_excludes_zero,
                "passes_min_effect": primary_b_min_effect,
            },
            "passes_brittleness_gate": not brittle_fail,
            "passes_registered_h1_gates": h1_passes_registered_gates,
        },
        "multiplicity": {
            "total_test_count": int(len(pvals)),
            "holm_secondary_and_pair_tests": holm(pvals),
            "note": "The registered meta primary is listed for transparency; secondary claims use corrected p-values.",
        },
        "h2_regime_interaction": h2,
        "family_spanning_verdict": {
            "verdict": family_verdict,
            "final_claim_verdict_after_h1_gates": final_claim_verdict,
            "positive_pair_definition": "per-pair primary must clear base A and base B with diff >= 0.02 and lower CI > 0",
            "positive_primary_pairs": positive_pairs,
            "positive_families": positive_families,
        },
        "per_pair": per_pair,
        "coverage": {
            "all_json_count": int(len(paths)),
            "completed_json_count": int(len(records)),
            "skipped_or_failed": [
                {
                    "path": str(p),
                    "status": json.loads(p.read_text()).get("coverage", {}).get("attempt_status"),
                    "reason": json.loads(p.read_text()).get("coverage", {}).get("skip_reason"),
                }
                for p in paths
                if json.loads(p.read_text()).get("coverage", {}).get("attempt_status") != "completed"
            ],
        },
        "environment": environment_report(),
    }
    out_path = Path(args.out) if args.out else in_dir / "shadow_v2_meta.json"
    _atomic_write_json(out_path, payload)
    print(f"[meta] completed_pairs={len(records)} primary_mean={meta_primary.get('mean')} ci=[{meta_primary.get('ci_lo')}, {meta_primary.get('ci_hi')}]")
    print(f"[meta] primary_base_b_mean={meta_primary_base_b.get('mean')} ci=[{meta_primary_base_b.get('ci_lo')}, {meta_primary_base_b.get('ci_hi')}]")
    print(f"[meta] total_test_count={len(pvals)} family_verdict={family_verdict}")
    print(f"[write] {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run one model x benchmark pair")
    run.add_argument("--model", required=True)
    run.add_argument("--benchmark", required=True)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--out-dir", default=str(HERE / "comprehensive_outputs"))
    run.add_argument("--max-new-tokens", type=int, default=1)
    run.add_argument("--k-support", type=int, default=K_SUPPORT_DEFAULT)
    run.add_argument("--n-splits", type=int, default=5)
    run.add_argument("--n-repeats", type=int, default=10)
    run.add_argument("--n-boot", type=int, default=1000)
    run.add_argument("--n-permutations", type=int, default=1000)
    run.add_argument("--seed", type=int, default=FRESH_SEED)

    meta = sub.add_parser("meta", help="aggregate completed per-pair JSON files")
    meta.add_argument("--in-dir", required=True)
    meta.add_argument("--out", default=None)

    reanalyze = sub.add_parser("reanalyze", help="recompute analysis blocks from stored per-sample rows")
    reanalyze.add_argument("--in-dir", required=True)
    reanalyze.add_argument("--n-boot", type=int, default=10000)
    reanalyze.add_argument("--n-permutations", type=int, default=None)
    reanalyze.add_argument("--n-splits", type=int, default=None)
    reanalyze.add_argument("--n-repeats", type=int, default=None)
    reanalyze.add_argument("--seed", type=int, default=None)
    reanalyze.add_argument("--jobs", type=int, default=1)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    raw = sys.argv[1:] if argv is None else list(argv)
    # Backward-compatible convenience for the requested CLI shape without a
    # subcommand: comprehensive_run.py --model ... --benchmark ...
    if raw and raw[0] not in {"run", "meta", "reanalyze", "-h", "--help"}:
        raw = ["run", *raw]
    args = parser.parse_args(raw)
    if args.cmd is None:
        parser.print_help()
        raise SystemExit(2)
    if args.cmd == "run":
        run_one(args)
    elif args.cmd == "meta":
        run_meta(args)
    elif args.cmd == "reanalyze":
        run_reanalyze(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
