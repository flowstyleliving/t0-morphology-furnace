#!/usr/bin/env python3
"""Labeled ANLI pilot for research-candidate #10 (shadow-ambiguity).

This is an exploratory-only replay of the sealed ANLI commit plane. It computes
readout-morphology statistics at the same gen_step=1/final-layer locus as the
v3 calibrator path, then asks whether each shadow statistic adds AUROC beyond
the sealed pair {surprise, null_ratio_post_rank1}.

Artifacts are written next to this file; sealed repo modules/data are imported
and read, never modified.

Run from repo root:
    .venv/bin/python exploratory/shadow-ambiguity/labeled_pilot_anli.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
from pri_calibrator import _hash_file, _load_calibration_jsonl  # noqa: E402
from test_shadow_ambiguity import (  # noqa: E402
    fisher_eff_rank,
    fisher_spectral_entropy,
    participation_ratio,
    shadow_logvol_post_rank,
)

SCHEMA = "shadow_ambiguity_labeled_pilot/v1"
DEFAULT_DATA = REPO / "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl"
ABSENT_PINNED_N100 = REPO / "experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl"
DEFAULT_MODELS = (
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
)
FEATURES = (
    "surprise",
    "null_ratio_post_rank1",
    "fisher_eff_rank",
    "spectral_entropy",
    "shadow_logvol_r1",
    "participation_ratio",
)
SHADOW_STATS = (
    "fisher_eff_rank",
    "spectral_entropy",
    "shadow_logvol_r1",
    "participation_ratio",
)
K_SUPPORT_DEFAULT = 512
SEED_DEFAULT = 20260607


def _sha256_jsonl_label_prompt(path: Path) -> str:
    _, _, h = _load_calibration_jsonl(str(path))
    return h


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(":", "_")


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


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-300)


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
    RG = np.einsum("ij,jk->ik", R, gram, optimize=True)
    M = np.einsum("ij,jk->ik", RG, R, optimize=True)
    M = 0.5 * (M + M.T)
    eig = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    pad = max(int(d) - eig.size, 0)
    return np.concatenate([eig, np.zeros(pad, dtype=np.float64)]) if pad else eig


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


@dataclass
class FeatureRun:
    rows: List[Dict[str, Any]]
    drops: Dict[str, int]
    diagnostics: Dict[str, Any]


def trace_model_features(
    model_id: str,
    data_path: Path,
    *,
    limit: int,
    max_new_tokens: int,
    k_support: int,
) -> FeatureRun:
    prompts, labels, data_hash = _load_calibration_jsonl(str(data_path))
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, projection, layer_indices = pipeline.load_model(model_id, cfg)
    final_layer = _fit_final_layer_name(layer_indices)
    gamma = _extract_final_norm_gamma(model)
    if gamma is None:
        raise RuntimeError("final RMSNorm gamma unavailable; null_ratio path cannot run")
    pri_comp = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)
    d, V = int(projection.hidden_size), int(projection.vocab_size)

    print(f"[trace] model={model_id}")
    print(f"[trace] samples={len(prompts)} final_layer={final_layer} d={d} V={V} K={k_support}")
    print("[trace] commit instant: gen_step=1, final-layer h_t vs last_prefix_hidden; p_t=trace.gen_probs[0]")

    rows: List[Dict[str, Any]] = []
    drops = {
        "trace_failed": 0,
        "no_gen_step1": 0,
        "nonfinite_prob_or_surprise": 0,
        "nonfinite_null_ratio": 0,
        "support_rows_missing": 0,
        "nonfinite_support_rows": 0,
        "nonfinite_spectrum": 0,
    }
    first_diag: Dict[str, Any] = {}

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
                v3_capture=False,
            )
        except Exception as exc:  # noqa: BLE001 - exploratory replay, count and continue.
            drops["trace_failed"] += 1
            print(f"[trace]   sample {i}: trace FAILED ({exc})")
            continue

        gen_hidden = trace.get("gen_hidden", {}).get(final_layer, [])
        gen_probs = trace.get("gen_probs", [])
        gen_surprises = trace.get("gen_surprises", [])
        if not gen_hidden or not gen_probs or not gen_surprises:
            drops["no_gen_step1"] += 1
            continue

        h_t = np.asarray(gen_hidden[0], dtype=np.float32)
        h_prev = np.asarray(trace["last_prefix_hidden"][final_layer], dtype=np.float32)
        p_t = np.asarray(gen_probs[0], dtype=np.float64)
        surprise = float(gen_surprises[0])
        if (
            p_t.ndim != 1
            or not np.isfinite(p_t).all()
            or not math.isfinite(surprise)
            or float(np.sum(p_t)) <= 0.0
        ):
            drops["nonfinite_prob_or_surprise"] += 1
            continue
        p_t = p_t / (float(np.sum(p_t)) + 1e-300)

        comp = pri_comp.compute_step(
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
            drops["nonfinite_null_ratio"] += 1
            continue

        top_k = int(min(k_support, p_t.shape[0]))
        idx = np.argpartition(-p_t, kth=top_k - 1)[:top_k].astype(np.int32)
        W_s = projection.get_rows(idx)
        if W_s is None or W_s.ndim != 2:
            drops["support_rows_missing"] += 1
            continue
        W_s = W_s.astype(np.float64)
        if not np.isfinite(W_s).all():
            drops["nonfinite_support_rows"] += 1
            continue
        spec = fc_full_spectrum(W_s, p_t[idx], d)
        if not np.isfinite(spec).all():
            drops["nonfinite_spectrum"] += 1
            continue

        if not first_diag:
            first_diag = {
                "sample_idx": i,
                "gen_token_id": int((trace.get("gen_token_ids") or [None])[0]),
                "generated_text_prefix": str(trace.get("generated_text") or "")[:80],
                "p_t_sum": float(np.sum(p_t)),
                "p_t_argmax": int(np.argmax(p_t)),
                "p_t_max": float(np.max(p_t)),
                "W_s_shape": list(W_s.shape),
                "W_s_absmax": float(np.max(np.abs(W_s))),
                "spectrum_len": int(spec.size),
                "spectrum_sum": float(np.sum(spec)),
            }

        rows.append(
            {
                "sample_idx": int(i),
                "label": int(label),
                "gold": "NO" if int(label) == 1 else "YES",
                "surprise": surprise,
                "null_ratio_post_rank1": null_ratio,
                "fisher_eff_rank": fisher_eff_rank(spec),
                "spectral_entropy": fisher_spectral_entropy(spec),
                "shadow_logvol_r1": shadow_logvol_post_rank(spec, r=1),
                "participation_ratio": participation_ratio(spec),
                "gen_token_id": int((trace.get("gen_token_ids") or [-1])[0]),
            }
        )
        if (i + 1) % 25 == 0 or i + 1 == len(prompts):
            print(f"[trace]   {i + 1}/{len(prompts)} processed; usable={len(rows)}")

    diagnostics = {
        "data_hash_sha256": data_hash,
        "n_requested": int(len(prompts)),
        "n_usable": int(len(rows)),
        "model_dims": {"d_model": d, "vocab": V},
        "layer_indices": layer_indices,
        "final_layer_used": final_layer,
        "first_valid_sample": first_diag,
    }
    try:
        del model, tokenizer, projection, pri_comp
        pipeline.clear_mlx_cache()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass
    return FeatureRun(rows=rows, drops=drops, diagnostics=diagnostics)


def _feature_arrays(rows: List[Dict[str, Any]], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    X = np.array([[float(r[name]) for name in feature_names] for r in rows], dtype=np.float64)
    return y, X


def _valid_mask(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(X).all(axis=1)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if finite.sum() < 4 or len(np.unique(y[finite])) < 2 or np.isclose(np.nanstd(scores[finite]), 0.0):
        return float("nan")
    return float(roc_auc_score(y[finite], scores[finite]))


def cv_locked_marginal_auc(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    yv, sv = y[finite], scores[finite]
    if len(yv) < 8 or len(np.unique(yv)) < 2:
        return {"auroc": None, "n": int(len(yv)), "fold_signs": []}

    k = int(min(n_splits, np.bincount(yv).min()))
    if k < 2:
        return {"auroc": None, "n": int(len(yv)), "fold_signs": []}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(yv), np.nan, dtype=np.float64)
    signs: List[int] = []
    fold_aucs: List[float] = []
    for train, test in skf.split(np.zeros(len(yv)), yv):
        train_auc = _safe_auc(yv[train], sv[train])
        sign = 1 if (math.isfinite(train_auc) and train_auc >= 0.5) else -1
        signs.append(sign)
        oof[test] = sign * sv[test]
        fold_aucs.append(_safe_auc(yv[test], oof[test]))
    auc = _safe_auc(yv, oof)
    return {
        "auroc": None if not math.isfinite(auc) else float(auc),
        "n": int(len(yv)),
        "fold_signs": signs,
        "fold_aurocs": [None if not math.isfinite(a) else float(a) for a in fold_aucs],
    }


def cv_logit_oof(
    y: np.ndarray,
    X: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    mask = _valid_mask(y, X)
    yv, Xv = y[mask], X[mask]
    if len(yv) < 12 or len(np.unique(yv)) < 2:
        return {"scores": np.full(len(y), np.nan), "auroc": float("nan"), "mask": mask, "n": int(len(yv))}
    k = int(min(n_splits, np.bincount(yv).min()))
    if k < 2:
        return {"scores": np.full(len(y), np.nan), "auroc": float("nan"), "mask": mask, "n": int(len(yv))}
    oof_v = np.full(len(yv), np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train, test in skf.split(Xv, yv):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xv[train])
        Xh = scaler.transform(Xv[test])
        clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed)
        clf.fit(Xt, yv[train])
        oof_v[test] = clf.predict_proba(Xh)[:, 1]
    auc = _safe_auc(yv, oof_v)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    scores[np.where(mask)[0]] = oof_v
    return {"scores": scores, "auroc": auc, "mask": mask, "n": int(len(yv))}


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
    if len(yv) < 12 or len(np.unique(yv)) < 2:
        return {"diff": None, "ci_lo": None, "ci_hi": None, "n_boot_used": 0, "n": int(len(yv))}
    base_auc = _safe_auc(yv, bv)
    aug_auc = _safe_auc(yv, av)
    diff = aug_auc - base_auc
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    n = len(yv)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(yv[idx])) < 2:
            continue
        db = _safe_auc(yv[idx], bv[idx])
        da = _safe_auc(yv[idx], av[idx])
        if math.isfinite(db) and math.isfinite(da):
            diffs.append(da - db)
    if not diffs:
        return {
            "base_auroc": base_auc,
            "augmented_auroc": aug_auc,
            "diff": diff,
            "ci_lo": None,
            "ci_hi": None,
            "n_boot_used": 0,
            "n": int(n),
        }
    arr = np.asarray(diffs)
    return {
        "base_auroc": float(base_auc),
        "augmented_auroc": float(aug_auc),
        "diff": float(diff),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_boot_used": int(len(arr)),
        "n": int(n),
    }


def incremental_analysis(
    rows: List[Dict[str, Any]],
    stat: str,
    *,
    n_splits: int,
    n_boot: int,
    seed: int,
    shuffle_labels: bool = False,
) -> Dict[str, Any]:
    names_base = ("surprise", "null_ratio_post_rank1")
    names_aug = ("surprise", "null_ratio_post_rank1", stat)
    y, X_aug = _feature_arrays(rows, names_aug)
    X_base = X_aug[:, :2]
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)
    base = cv_logit_oof(y, X_base, n_splits=n_splits, seed=seed)
    aug = cv_logit_oof(y, X_aug, n_splits=n_splits, seed=seed)
    diff = paired_bootstrap_auc_diff(
        y,
        base["scores"],
        aug["scores"],
        n_boot=n_boot,
        seed=seed + 101,
    )
    return {
        "base_features": list(names_base),
        "augmented_feature": stat,
        "base_auroc": None if not math.isfinite(base["auroc"]) else float(base["auroc"]),
        "augmented_auroc": None if not math.isfinite(aug["auroc"]) else float(aug["auroc"]),
        "diff": diff.get("diff"),
        "ci_lo": diff.get("ci_lo"),
        "ci_hi": diff.get("ci_hi"),
        "n": int(diff.get("n", 0)),
        "n_boot_used": int(diff.get("n_boot_used", 0)),
        "shuffle_labels": bool(shuffle_labels),
    }


def partial_association(
    rows: List[Dict[str, Any]],
    stat: str,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    y, X = _feature_arrays(rows, ("surprise", "null_ratio_post_rank1", stat))
    mask = _valid_mask(y, X)
    yv, Xv = y[mask].astype(np.float64), X[mask]
    if len(yv) < 12 or len(np.unique(yv)) < 2:
        return {"partial_pearson_r": None, "ci_lo": None, "ci_hi": None, "n": int(len(yv))}
    controls = Xv[:, :2]
    stat_v = Xv[:, 2]

    def one(idx: np.ndarray) -> float:
        C = controls[idx]
        yy = yv[idx]
        ss = stat_v[idx]
        if len(np.unique(yy)) < 2 or np.isclose(np.std(ss), 0.0):
            return float("nan")
        scaler = StandardScaler()
        Cz = scaler.fit_transform(C)
        y_res = yy - LinearRegression().fit(Cz, yy).predict(Cz)
        s_res = ss - LinearRegression().fit(Cz, ss).predict(Cz)
        denom = float(np.std(y_res) * np.std(s_res))
        if denom <= 0.0:
            return float("nan")
        return float(np.corrcoef(y_res, s_res)[0, 1])

    idx0 = np.arange(len(yv))
    r0 = one(idx0)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), size=len(yv))
        r = one(idx)
        if math.isfinite(r):
            vals.append(r)
    if not vals:
        return {"partial_pearson_r": r0, "ci_lo": None, "ci_hi": None, "n": int(len(yv)), "n_boot_used": 0}
    arr = np.asarray(vals)
    return {
        "partial_pearson_r": None if not math.isfinite(r0) else float(r0),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": int(len(yv)),
        "n_boot_used": int(len(arr)),
    }


def dispersion(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name in FEATURES:
        vals = np.array([float(r[name]) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[name] = {"n": 0, "mean": None, "std": None, "iqr": None, "min": None, "max": None}
            continue
        out[name] = {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def analyze_rows(
    rows: List[Dict[str, Any]],
    *,
    n_splits: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    marginal: Dict[str, Any] = {}
    for j, name in enumerate(FEATURES):
        scores = np.array([float(r[name]) for r in rows], dtype=np.float64)
        marginal[name] = cv_locked_marginal_auc(y, scores, n_splits=n_splits, seed=seed + j)

    incremental = {
        stat: incremental_analysis(rows, stat, n_splits=n_splits, n_boot=n_boot, seed=seed + 11 * i)
        for i, stat in enumerate(SHADOW_STATS, start=1)
    }
    shuffled = {
        stat: incremental_analysis(
            rows,
            stat,
            n_splits=n_splits,
            n_boot=n_boot,
            seed=seed + 1000 + 11 * i,
            shuffle_labels=True,
        )
        for i, stat in enumerate(SHADOW_STATS, start=1)
    }
    partial = {
        stat: partial_association(rows, stat, n_boot=n_boot, seed=seed + 2000 + 13 * i)
        for i, stat in enumerate(SHADOW_STATS, start=1)
    }
    return {
        "marginal_cv_locked_auroc": marginal,
        "incremental_logistic_cv": incremental,
        "negative_control_shuffled_labels": shuffled,
        "partial_association_controlling_surprise_null_ratio": partial,
        "dispersion": dispersion(rows),
    }


def verify_setup(data_path: Path, models: Sequence[str]) -> Dict[str, Any]:
    if not data_path.exists():
        raise SystemExit(f"data path not found: {data_path}")
    prompts, labels, data_hash = _load_calibration_jsonl(str(data_path))
    counts = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    profiles_dir = REPO / "experiments/t0-sealed/2026-05-26/profiles/anli"
    profile_hits = {}
    for model in models:
        p = profiles_dir / f"{_slug(model)}.profile.json"
        profile_hits[model] = str(p.relative_to(REPO)) if p.exists() else None
    setup = {
        "data_path": str(data_path.relative_to(REPO) if data_path.is_relative_to(REPO) else data_path),
        "data_exists": True,
        "data_hash_sha256": data_hash,
        "n_available": int(len(prompts)),
        "n_preferred": 200 if len(prompts) >= 200 else int(len(prompts)),
        "absent_original_n100_path": str(ABSENT_PINNED_N100.relative_to(REPO)),
        "original_n100_exists": ABSENT_PINNED_N100.exists(),
        "label_definition": {
            "1": "contradiction target class; prompt gold answer NO",
            "0": "entailed/consistent non-contradiction class; prompt gold answer YES",
        },
        "label_counts": counts,
        "commit_instant": {
            "name": "sealed v3/calibrator plane",
            "gen_step": 1,
            "layer": "final",
            "h_prev": "last_prefix_hidden[final]",
            "h_t": "gen_hidden[final][0]",
            "p_t_for_null_ratio_and_shadow": "trace.gen_probs[0]",
            "surprise": "trace.gen_surprises[0]",
        },
        "requested_models": list(models),
        "sealed_profile_trace_presence_read_only": profile_hits,
    }
    print("=== STEP 0 SETUP VERIFY ===")
    print(f"data_path: {setup['data_path']}")
    print(f"n_available: {setup['n_available']} (using preferred n={setup['n_preferred']} unless --limit caps it)")
    print(f"label_counts: {setup['label_counts']}")
    print("label_definition: 1=contradiction/gold NO, 0=entailed/gold YES")
    print("commit_instant: gen_step=1 final layer; null_ratio uses PRIComputer.null_ratio_and_energy")
    print(f"original pinned n100 path exists: {setup['original_n100_exists']} ({setup['absent_original_n100_path']})")
    print("sealed profile traces present for requested models:")
    for model, hit in profile_hits.items():
        print(f"  {model}: {hit or 'not found'}")
    print("=== END STEP 0 ===")
    return setup


def run(args: argparse.Namespace) -> int:
    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    models = list(args.models or DEFAULT_MODELS)
    setup = verify_setup(data_path, models)
    _atomic_write_json(out_dir / "labeled_pilot_setup.json", setup)
    if args.verify_only:
        return 0

    all_results: Dict[str, Any] = {
        "schema": SCHEMA,
        "setup": setup,
        "config": {
            "models": models,
            "limit": int(args.limit),
            "max_new_tokens": int(args.max_new_tokens),
            "k_support": int(args.k_support),
            "cv_folds": int(args.cv_folds),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "provenance": {
            "started_at_iso": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "script_hash_sha256": _hash_file(Path(__file__).resolve()),
            "pri_runtime_hash_sha256": _file_sha256(REPO / "pri_runtime.py"),
            "data_hash_sha256": setup["data_hash_sha256"],
        },
        "models": {},
    }

    for mi, model_id in enumerate(models):
        out_path = out_dir / f"labeled_pilot__{_slug(model_id)}.json"
        print()
        print("=" * 78)
        print(f"[pilot] running {model_id}")
        print("=" * 78)
        try:
            fr = trace_model_features(
                model_id,
                data_path,
                limit=int(args.limit),
                max_new_tokens=int(args.max_new_tokens),
                k_support=int(args.k_support),
            )
            analysis = analyze_rows(
                fr.rows,
                n_splits=int(args.cv_folds),
                n_boot=int(args.bootstrap),
                seed=int(args.seed) + 97 * mi,
            )
            payload = {
                "schema": SCHEMA,
                "model": model_id,
                "setup": setup,
                "config": all_results["config"],
                "coverage": {
                    **fr.diagnostics,
                    "drops": fr.drops,
                    "n_dropped_total": int(sum(fr.drops.values())),
                },
                "features": fr.rows,
                "analysis": analysis,
            }
            _atomic_write_json(out_path, payload)
            all_results["models"][model_id] = {
                "result_path": str(out_path.relative_to(REPO) if out_path.is_relative_to(REPO) else out_path),
                "coverage": payload["coverage"],
                "analysis": analysis,
            }
            print(f"[pilot] wrote {out_path}")
        except Exception as exc:  # noqa: BLE001 - keep panel coverage explicit.
            print(f"[pilot] MODEL FAILED: {model_id}: {exc}")
            all_results["models"][model_id] = {"error": str(exc)}
            _atomic_write_json(
                out_path,
                {
                    "schema": SCHEMA,
                    "model": model_id,
                    "setup": setup,
                    "config": all_results["config"],
                    "error": str(exc),
                },
            )

    all_results["completed_at_iso"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(out_dir / "labeled_pilot_summary.json", all_results)
    print(f"[pilot] wrote {out_dir / 'labeled_pilot_summary.json'}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Labeled ANLI pilot for shadow-ambiguity stats.")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--limit", type=int, default=0, help="cap samples; default 0 uses all")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--k-support", type=int, default=K_SUPPORT_DEFAULT)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
