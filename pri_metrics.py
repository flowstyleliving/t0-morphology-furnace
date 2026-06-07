"""
Core metrics for PRI-at-commitment experiments.

This module intentionally excludes semantic-uncertainty / hbar_s computations.
"""

from __future__ import annotations

import math
from typing import List

import mlx.core as mx
import numpy as np

import config


def _cfg(cfg_obj: config.UncertaintyConfig | None) -> config.UncertaintyConfig:
    return cfg_obj if cfg_obj is not None else config.DEFAULT_UNCERTAINTY_CONFIG


def normalize_vector(
    v: mx.array,
    epsilon: float | None = None,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> mx.array:
    cfg = _cfg(cfg_obj)
    eps = cfg.cosine_epsilon if epsilon is None else float(epsilon)
    norm = mx.sqrt(mx.sum(v * v) + eps)
    return v / norm


def compute_cosine_distance(
    v1: mx.array,
    v2: mx.array,
    epsilon: float | None = None,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> float:
    cfg = _cfg(cfg_obj)
    eps = cfg.cosine_epsilon if epsilon is None else float(epsilon)
    v1_norm = normalize_vector(v1, eps, cfg_obj=cfg_obj)
    v2_norm = normalize_vector(v2, eps, cfg_obj=cfg_obj)
    cos_sim = float(mx.sum(v1_norm * v2_norm).item())
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(1.0 - cos_sim)


def compute_surprise(
    probs: mx.array,
    selected_token: int,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> float:
    cfg = _cfg(cfg_obj)
    selected_prob = float(probs[selected_token].item())
    p_lo, p_hi = cfg.selected_prob_clamp
    selected_prob = max(float(p_lo), min(float(p_hi), selected_prob))
    return float(-math.log(selected_prob))


def compute_pri(
    surprise: float,
    delta_h: float,
    alpha: float | None = None,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> float:
    cfg = _cfg(cfg_obj)
    alpha_eff = cfg.pri_alpha if alpha is None else float(alpha)
    return float(surprise * (1.0 + alpha_eff * delta_h))


def compute_delta_sigma_mean_cross_layer_jsd(
    hidden_vectors: List[mx.array],
    epsilon: float | None = None,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> float:
    cfg = _cfg(cfg_obj)
    eps = cfg.delta_sigma_jsd_epsilon if epsilon is None else float(epsilon)
    if len(hidden_vectors) < 2:
        return 0.0

    layer_probs: List[np.ndarray] = []
    for h in hidden_vectors:
        p = mx.softmax(h, axis=0)
        arr = np.asarray(p.tolist(), dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            continue
        arr = arr / np.clip(np.sum(arr), eps, None)
        layer_probs.append(arr)

    if len(layer_probs) < 2:
        return 0.0

    probs_mat = np.vstack(layer_probs)
    mean_p = np.mean(probs_mat, axis=0)
    mean_p = mean_p / np.clip(np.sum(mean_p), eps, None)
    kls = np.sum(
        probs_mat * (np.log(probs_mat + eps) - np.log(mean_p[None, :] + eps)),
        axis=1,
    )
    jsd = float(np.mean(kls))
    denom = max(float(np.log(probs_mat.shape[1])), eps)
    return float(max(0.0, min(1.0, jsd / denom)))


def compute_momentum_features(
    hidden_vectors: List[mx.array],
    cfg_obj: config.UncertaintyConfig | None = None,
) -> dict:
    """
    Residual-stream momentum across transformer layers at a single generation step.

    Computes layer increments inc_l = h_l - h_{l-1} and measures how
    coherently they point in the same direction — the "rush to commitment."

    momentum_pc1_ratio: PC1 energy fraction of the unit-increment matrix.
        High = all layers pushing the same way (coordinated rush).
    momentum_mean_cos: Mean cosine similarity between consecutive layer increments.
        High = each layer continues the previous layer's direction.
    """
    default = {"momentum_pc1_ratio": 0.0, "momentum_mean_cos": 0.0}
    if len(hidden_vectors) < 3:
        return default

    eps = 1e-8

    vecs: list = []
    for h in hidden_vectors:
        arr = np.asarray(h.tolist(), dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            return default
        vecs.append(arr)

    incs = [vecs[i] - vecs[i - 1] for i in range(1, len(vecs))]
    norms = np.array([np.linalg.norm(inc) for inc in incs], dtype=np.float64)
    valid = norms > eps
    if int(valid.sum()) < 2:
        return default

    unit_incs = [incs[i] / norms[i] for i in range(len(incs)) if valid[i]]

    mat = np.vstack(unit_incs)
    mat_centered = mat - mat.mean(axis=0, keepdims=True)
    if np.allclose(mat_centered, 0.0):
        # All increments identical → perfect coherence
        pc1_ratio = 1.0
    else:
        try:
            _, s, _ = np.linalg.svd(mat_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return default
        s_sq = s ** 2
        s_total = float(np.sum(s_sq))
        pc1_ratio = float(s_sq[0] / s_total) if s_total > eps else 0.0

    cos_vals = [
        float(np.dot(unit_incs[i], unit_incs[i + 1]))
        for i in range(len(unit_incs) - 1)
    ]
    mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0

    return {
        "momentum_pc1_ratio": float(np.clip(pc1_ratio, 0.0, 1.0)),
        "momentum_mean_cos": float(np.clip(mean_cos, -1.0, 1.0)),
    }


def compute_svd_spectrum_features(
    hidden_vectors: List[mx.array],
    epsilon: float | None = None,
    cfg_obj: config.UncertaintyConfig | None = None,
) -> dict:
    cfg = _cfg(cfg_obj)
    eps = cfg.svd_epsilon if epsilon is None else float(epsilon)
    if len(hidden_vectors) < 2:
        return {"pc1_ratio": 0.0, "effective_rank": 0.0, "spectral_entropy": 0.0}

    normalized = []
    for h in hidden_vectors:
        norm = mx.sqrt(mx.sum(h * h) + eps)
        normalized.append(h / norm)

    stacked = mx.stack(normalized, axis=0)
    mat = np.asarray(stacked.tolist(), dtype=np.float64)
    if mat.ndim != 2:
        return {"pc1_ratio": 0.0, "effective_rank": 0.0, "spectral_entropy": 0.0}
    mat = mat - mat.mean(axis=0, keepdims=True)
    if np.allclose(mat, 0.0):
        return {"pc1_ratio": 0.0, "effective_rank": 0.0, "spectral_entropy": 0.0}

    try:
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"pc1_ratio": 0.0, "effective_rank": 0.0, "spectral_entropy": 0.0}

    s_sum = float(np.sum(s))
    if s_sum <= eps:
        return {"pc1_ratio": 0.0, "effective_rank": 0.0, "spectral_entropy": 0.0}

    weights = s / (s_sum + eps)
    pc1_ratio = float(np.clip(weights[0], 0.0, 1.0))
    entropy = float(-np.sum(weights * np.log(weights + eps)))
    rank = max(int(len(weights)), 1)
    entropy_norm = float(entropy / max(np.log(rank), eps))
    effective_rank = float(np.exp(entropy))
    return {
        "pc1_ratio": pc1_ratio,
        "effective_rank": effective_rank,
        "spectral_entropy": float(np.clip(entropy_norm, 0.0, 1.0)),
    }
