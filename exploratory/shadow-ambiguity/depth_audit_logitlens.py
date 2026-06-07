#!/usr/bin/env python3
"""Layer-wise logit-lens depth audit for shadow-ambiguity on ANLI R1.

Exploratory-only research-candidate #10 audit. It replays the sealed ANLI R1
commit plane at gen_step=1, captures every transformer-block hidden state, and
compares the rank-1 v3 readout against full-spectrum centered-Fisher shadow
statistics as a function of depth.

Geometry note: all per-layer measurements here are functions of (Delta h_l,
p_l), where Delta h_l is the adjacent-layer residual update at the generated
commit position. These readouts compare rank-1 vs full-spectrum visibility only;
they do not attribute anything to attention-vs-MLP sub-layer structure.

Run from repo root:
    .venv/bin/python exploratory/shadow-ambiguity/depth_audit_logitlens.py \
      --limit 3 --models mlx-community/Qwen3-8B-4bit
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    shadow_logvol_post_rank,
)

SCHEMA = "shadow_ambiguity_depth_audit_logitlens/v1"
DEFAULT_DATA = REPO / "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl"
DEFAULT_MODELS = ("mlx-community/Qwen3-8B-4bit",)
K_SUPPORT_DEFAULT = 512
SEED_DEFAULT = 20260607
SPOTLIGHT_LAYERS = tuple(range(20, 29))
FEATURES = (
    "null_ratio_post_rank1",
    "shadow_logvol",
    "spectral_entropy",
    "eff_rank",
)


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(":", "_")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
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


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (float(np.sum(e)) + 1e-300)


def fc_full_spectrum(W_s: np.ndarray, p_s: np.ndarray, d: int) -> np.ndarray:
    """Full d-dimensional centered-Fisher spectrum via the K x K dual.

    This is the einsum implementation from the temperature pre-check: it builds
    the exact nonzero dual spectrum for F_c on the top-K support and pads the
    d-dimensional structural zero tail for log-volume accounting.
    """
    p_s = np.asarray(p_s, dtype=np.float64).reshape(-1)
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


def _extract_final_norm_gamma(model: Any) -> np.ndarray:
    fn = getattr(pipeline, "_extract_final_rmsnorm_gamma", None)
    if fn is None:
        raise RuntimeError("inherited-core _extract_final_rmsnorm_gamma is unavailable")
    gamma = fn(model)
    if gamma is None:
        raise RuntimeError("final RMSNorm gamma unavailable")
    return np.asarray(gamma, dtype=np.float32)


def _parse_layers(spec: str, n_layers: int) -> List[int]:
    text = (spec or "all").strip().lower()
    if text in {"all", "*"}:
        return list(range(n_layers))
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    clean = sorted(set(i for i in out if 0 <= i < n_layers))
    if not clean:
        raise ValueError(f"no valid layer indices selected by --layers={spec!r}")
    return clean


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if finite.sum() < 2 or len(np.unique(y[finite])) < 2 or np.isclose(np.nanstd(scores[finite]), 0.0):
        return float("nan")
    return float(roc_auc_score(y[finite], scores[finite]))


def cv_locked_marginal_auc(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> Dict[str, Any]:
    """Same sign-locked marginal AUROC protocol as labeled_pilot_anli.py."""
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


def direct_auc_for_smoke(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    auc = _safe_auc(y, scores)
    return None if not math.isfinite(auc) else float(auc)


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> Optional[float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if xv.size < 3 or np.isclose(np.std(xv), 0.0) or np.isclose(np.std(yv), 0.0):
        return None
    try:
        stat = spearmanr(xv, yv).statistic if kind == "spearman" else pearsonr(xv, yv).statistic
    except Exception:
        return None
    return None if not math.isfinite(float(stat)) else float(stat)


def _topk_indices(p: np.ndarray, k: int) -> np.ndarray:
    kk = int(min(k, p.shape[0]))
    return np.argpartition(-p, kth=kk - 1)[:kk].astype(np.int32)


def _commit_embedding_last(
    model: Any,
    tokenizer: Any,
    wrapped_prompt: str,
    gen_token_id: int,
) -> np.ndarray:
    """Return the post-embedding/pre-block vector at the generated position."""
    from model_adapters import post_embed_scale

    core = model.model if hasattr(model, "model") else model
    token_ids = pipeline.encode_text(tokenizer, wrapped_prompt)
    token_ids.append(int(gen_token_id))
    x = mx.array(np.asarray(token_ids, dtype=np.int32)[None, :])
    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("Could not locate token embedding layer on model")
    h = post_embed_scale(core, h)
    mx.eval(h)
    return pipeline.to_numpy(h).astype(np.float32)[0, -1]


@dataclass
class ModelRun:
    samples: List[Dict[str, Any]]
    layer_rows: List[Dict[str, Any]]
    drops: Dict[str, int]
    diagnostics: Dict[str, Any]


def trace_depth_features(
    model_id: str,
    data_path: Path,
    *,
    limit: int,
    layers_spec: str,
    k_support: int,
    max_new_tokens: int,
) -> ModelRun:
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
    gamma = _extract_final_norm_gamma(model)
    pri_comp = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(model_id)
    d, vocab = int(projection.hidden_size), int(projection.vocab_size)

    drops = {
        "trace_failed": 0,
        "no_gen_step1": 0,
        "missing_layer_capture": 0,
        "nonfinite_hidden_or_delta": 0,
        "nonfinite_logitlens_logits": 0,
        "nonfinite_probability": 0,
        "support_rows_missing": 0,
        "nonfinite_support_rows": 0,
        "nonfinite_spectrum": 0,
        "nonfinite_null_ratio": 0,
    }
    samples: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    first_valid: Dict[str, Any] = {}
    selected_layers: Optional[List[int]] = None
    n_layers_seen: Optional[int] = None
    force_rank_for_512 = max(32, int(math.ceil(int(k_support) / 16.0)))
    null_rank_values = sorted({1, force_rank_for_512})

    print(f"[depth] model={model_id}")
    print(f"[depth] samples={len(prompts)} K={k_support} null_rank_values={null_rank_values}")
    print("[depth] commit instant: gen_step=1 (trace step_idx=0 after first generated token)")
    print("[depth] logit lens: p_l = softmax(OutputProjection.project(final_rmsnorm(h_l)))")

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
        except Exception as exc:  # noqa: BLE001 - exploratory replay, count and continue.
            drops["trace_failed"] += 1
            print(f"[depth]   sample {i}: trace FAILED ({exc})", flush=True)
            continue

        captures_by_step = trace.get("gen_captures_by_step") or []
        gen_ids = trace.get("gen_token_ids") or []
        if not captures_by_step or not gen_ids:
            drops["no_gen_step1"] += 1
            continue

        step_caps = captures_by_step[0]
        step_layer_indices = (trace.get("gen_layer_indices_by_step") or [{}])[0]
        n_layers = int(trace.get("n_layers") or len(step_layer_indices))
        if selected_layers is None:
            n_layers_seen = n_layers
            selected_layers = _parse_layers(layers_spec, n_layers)
            print(f"[depth] n_layers={n_layers}; selected_layers={selected_layers[:8]}{'...' if len(selected_layers) > 8 else ''}")

        idx_to_name = {int(v): k for k, v in step_layer_indices.items()}
        hidden_by_index: Dict[int, np.ndarray] = {}
        for li, lname in idx_to_name.items():
            cap = step_caps.get(lname)
            if cap is not None and "h_t" in cap:
                hidden_by_index[li] = np.asarray(cap["h_t"], dtype=np.float32)

        try:
            embed_prev = _commit_embedding_last(model, tokenizer, wrapped, int(gen_ids[0]))
        except Exception as exc:  # noqa: BLE001
            drops["missing_layer_capture"] += 1
            print(f"[depth]   sample {i}: embedding capture FAILED ({exc})", flush=True)
            continue

        sample_layer_rows: List[Dict[str, Any]] = []
        for layer_idx in selected_layers or []:
            h_l = hidden_by_index.get(layer_idx)
            h_prev_layer = embed_prev if layer_idx == 0 else hidden_by_index.get(layer_idx - 1)
            if h_l is None or h_prev_layer is None:
                drops["missing_layer_capture"] += 1
                continue
            if not np.isfinite(h_l).all() or not np.isfinite(h_prev_layer).all():
                drops["nonfinite_hidden_or_delta"] += 1
                continue

            h_l_post = pri_comp.rmsnorm(h_l, gamma)
            h_prev_post = pri_comp.rmsnorm(h_prev_layer, gamma)
            dh_post = h_l_post - h_prev_post
            if not np.isfinite(dh_post).all():
                drops["nonfinite_hidden_or_delta"] += 1
                continue

            logits_l = projection.project(h_l_post)
            if logits_l.ndim != 1 or not np.isfinite(logits_l).all():
                drops["nonfinite_logitlens_logits"] += 1
                continue
            p_l = softmax_np(logits_l)
            if p_l.ndim != 1 or not np.isfinite(p_l).all() or float(np.sum(p_l)) <= 0.0:
                drops["nonfinite_probability"] += 1
                continue
            p_l = p_l / (float(np.sum(p_l)) + 1e-300)

            null_out = pri_comp.null_ratio_and_energy(
                dh_post,
                p_l,
                rank_values=null_rank_values,
            )
            null_ratio = float(null_out.get("null_ratio_post_rank1", float("nan")))
            if not math.isfinite(null_ratio):
                drops["nonfinite_null_ratio"] += 1
                continue

            idx = _topk_indices(p_l, k_support)
            W_s = projection.get_rows(idx)
            if W_s is None or W_s.ndim != 2:
                drops["support_rows_missing"] += 1
                continue
            W_s = W_s.astype(np.float64)
            if not np.isfinite(W_s).all():
                drops["nonfinite_support_rows"] += 1
                continue
            spec = fc_full_spectrum(W_s, p_l[idx], d)
            if not np.isfinite(spec).all():
                drops["nonfinite_spectrum"] += 1
                continue

            row = {
                "sample_idx": int(i),
                "label": int(label),
                "gold": "NO" if int(label) == 1 else "YES",
                "layer": int(layer_idx),
                "layer_name": idx_to_name.get(layer_idx, f"layer_{layer_idx}"),
                "null_ratio_post_rank1": null_ratio,
                "shadow_logvol": shadow_logvol_post_rank(spec, r=1),
                "spectral_entropy": fisher_spectral_entropy(spec),
                "eff_rank": fisher_eff_rank(spec),
                "p_max": float(np.max(p_l)),
                "p_argmax": int(np.argmax(p_l)),
                "gen_token_id": int(gen_ids[0]),
            }
            sample_layer_rows.append(row)

        if sample_layer_rows:
            layer_rows.extend(sample_layer_rows)
            samples.append(
                {
                    "sample_idx": int(i),
                    "label": int(label),
                    "gold": "NO" if int(label) == 1 else "YES",
                    "gen_token_id": int(gen_ids[0]),
                    "generated_text_prefix": str(trace.get("generated_text") or "")[:80],
                    "n_layer_rows": int(len(sample_layer_rows)),
                }
            )
            if not first_valid:
                first_valid = {
                    "sample_idx": int(i),
                    "gen_token_id": int(gen_ids[0]),
                    "generated_text_prefix": str(trace.get("generated_text") or "")[:80],
                    "first_layer_row": sample_layer_rows[0],
                    "spotlight_rows": [
                        r for r in sample_layer_rows if int(r["layer"]) in SPOTLIGHT_LAYERS
                    ][:3],
                    "step0_sanity": trace.get("step0_sanity"),
                }

        print(
            f"[depth]   {i + 1}/{len(prompts)} processed; "
            f"usable_samples={len(samples)} layer_rows={len(layer_rows)}",
            flush=True,
        )

    diagnostics = {
        "data_hash_sha256": data_hash,
        "n_requested": int(len(prompts)),
        "n_usable_samples": int(len(samples)),
        "n_layer_rows": int(len(layer_rows)),
        "n_layers": None if n_layers_seen is None else int(n_layers_seen),
        "selected_layers": selected_layers or [],
        "model_dims": {"d_model": d, "vocab": vocab},
        "layer_indices_from_load_model": layer_indices,
        "k_support": int(k_support),
        "null_ratio_support_note": (
            "pri_runtime.PRIComputer.null_ratio_and_energy chooses support "
            "as max(256, max_rank*16); this audit requests rank 32 alongside "
            "rank 1 so null_ratio_post_rank1 is computed on a 512-row support."
        ),
        "first_valid_sample": first_valid,
    }
    try:
        del model, tokenizer, projection, pri_comp
        pipeline.clear_mlx_cache()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass
    return ModelRun(samples=samples, layer_rows=layer_rows, drops=drops, diagnostics=diagnostics)


def analyze_layers(rows: List[Dict[str, Any]], *, n_splits: int, seed: int) -> Dict[str, Any]:
    if not rows:
        return {"per_layer": [], "crossover_candidates": [], "spotlight_20_28": []}
    layers = sorted({int(r["layer"]) for r in rows})
    per_layer: List[Dict[str, Any]] = []
    for li in layers:
        lr = [r for r in rows if int(r["layer"]) == li]
        y = np.asarray([int(r["label"]) for r in lr], dtype=np.int32)
        item: Dict[str, Any] = {
            "layer": int(li),
            "n": int(len(lr)),
            "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        }
        for j, feat in enumerate(FEATURES):
            scores = np.asarray([float(r[feat]) for r in lr], dtype=np.float64)
            item[f"{feat}_cv_locked_auroc"] = cv_locked_marginal_auc(
                y,
                scores,
                n_splits=n_splits,
                seed=seed + 97 * li + j,
            )
            item[f"{feat}_direct_auc_smoke"] = direct_auc_for_smoke(y, scores)
        pmax = np.asarray([float(r["p_max"]) for r in lr], dtype=np.float64)
        shadow = np.asarray([float(r["shadow_logvol"]) for r in lr], dtype=np.float64)
        item["brittleness_pmax_shadow_logvol_spearman"] = _safe_corr(pmax, shadow, "spearman")
        item["brittleness_pmax_shadow_logvol_pearson"] = _safe_corr(pmax, shadow, "pearson")
        item["means"] = {
            feat: float(np.nanmean([float(r[feat]) for r in lr]))
            for feat in FEATURES + ("p_max",)
        }
        per_layer.append(item)

    candidates: List[Dict[str, Any]] = []
    for item in per_layer:
        null_auc = item["null_ratio_post_rank1_cv_locked_auroc"]["auroc"]
        shadow_aucs = [
            item["shadow_logvol_cv_locked_auroc"]["auroc"],
            item["spectral_entropy_cv_locked_auroc"]["auroc"],
            item["eff_rank_cv_locked_auroc"]["auroc"],
        ]
        if null_auc is None or any(a is None for a in shadow_aucs):
            continue
        shadow_best = max(float(a) for a in shadow_aucs if a is not None)
        if abs(float(null_auc) - 0.5) <= 0.05 and shadow_best >= 0.6:
            candidates.append(
                {
                    "layer": int(item["layer"]),
                    "null_ratio_post_rank1_cv_locked_auroc": float(null_auc),
                    "best_shadow_cv_locked_auroc": float(shadow_best),
                }
            )

    return {
        "per_layer": per_layer,
        "crossover_candidates": candidates,
        "spotlight_20_28": [x for x in per_layer if int(x["layer"]) in SPOTLIGHT_LAYERS],
    }


def verify_setup(data_path: Path, models: Sequence[str]) -> Dict[str, Any]:
    if not data_path.exists():
        raise SystemExit(f"data path not found: {data_path}")
    prompts, labels, data_hash = _load_calibration_jsonl(str(data_path))
    counts = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    setup = {
        "data_path": str(data_path.relative_to(REPO) if data_path.is_relative_to(REPO) else data_path),
        "data_hash_sha256": data_hash,
        "n_available": int(len(prompts)),
        "label_counts": counts,
        "label_definition": {
            "1": "contradiction target class; prompt gold answer NO",
            "0": "entailed/consistent non-contradiction class; prompt gold answer YES",
        },
        "commit_instant": {
            "name": "sealed ANLI R1 first generated-token commit",
            "gen_step": 1,
            "trace_step_idx": 0,
            "capture": "trace_sample(..., v3_capture=True).gen_captures_by_step[0][layer_NN]['h_t']",
            "layer_delta": "Delta h_l = h_l - h_{l-1}; layer 0 uses the post-embedding/pre-block vector as h_{-1}",
            "logit_lens": "p_l = softmax(OutputProjection.project(final_rmsnorm(h_l)))",
        },
        "requested_models": list(models),
    }
    print("=== DEPTH AUDIT SETUP ===")
    print(f"data_path: {setup['data_path']}")
    print(f"n_available: {setup['n_available']} label_counts={setup['label_counts']}")
    print("label_definition: 1=contradiction/gold NO, 0=entailed/gold YES")
    print("commit_instant: gen_step=1, trace step_idx=0 after first generated token")
    print("=== END SETUP ===")
    return setup


def run(args: argparse.Namespace) -> int:
    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    models = list(args.models or DEFAULT_MODELS)
    setup = verify_setup(data_path, models)
    if args.verify_only:
        return 0

    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "setup": setup,
        "config": {
            "models": models,
            "layers": str(args.layers),
            "limit": int(args.limit),
            "max_new_tokens": int(args.max_new_tokens),
            "k_support": int(args.k_support),
            "cv_folds": int(args.cv_folds),
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
        out_path = out_dir / f"depth_audit_logitlens__{_slug(model_id)}.json"
        print()
        print("=" * 78)
        print(f"[depth] running {model_id}")
        print("=" * 78)
        try:
            mr = trace_depth_features(
                model_id,
                data_path,
                limit=int(args.limit),
                layers_spec=str(args.layers),
                k_support=int(args.k_support),
                max_new_tokens=int(args.max_new_tokens),
            )
            analysis = analyze_layers(
                mr.layer_rows,
                n_splits=int(args.cv_folds),
                seed=int(args.seed) + 193 * mi,
            )
            payload = {
                "schema": SCHEMA,
                "model": model_id,
                "setup": setup,
                "config": summary["config"],
                "coverage": {
                    **mr.diagnostics,
                    "drops": mr.drops,
                    "n_dropped_total": int(sum(mr.drops.values())),
                },
                "samples": mr.samples,
                "layer_features": mr.layer_rows,
                "analysis": analysis,
            }
            _atomic_write_json(out_path, payload)
            summary["models"][model_id] = {
                "result_path": str(out_path.relative_to(REPO) if out_path.is_relative_to(REPO) else out_path),
                "coverage": payload["coverage"],
                "analysis_brief": {
                    "n_per_layer": len(analysis.get("per_layer", [])),
                    "crossover_candidates": analysis.get("crossover_candidates", []),
                },
            }
            print(f"[depth] wrote {out_path}")
            print(f"[depth] n_layers={mr.diagnostics.get('n_layers')} n_usable_samples={mr.diagnostics.get('n_usable_samples')}")
            spotlight = [
                r for r in mr.layer_rows
                if int(r["sample_idx"]) == int(mr.samples[0]["sample_idx"]) and int(r["layer"]) in SPOTLIGHT_LAYERS
            ] if mr.samples else []
            if not spotlight:
                spotlight = mr.layer_rows[:5]
            print("[depth] sample layer rows:")
            for r in spotlight[:5]:
                print(
                    "  layer={layer:02d} null_ratio={null_ratio_post_rank1:.6f} "
                    "shadow_logvol={shadow_logvol:.6f} p_max={p_max:.6f}".format(**r)
                )
            print(f"[depth] nonfinite/drop counters: {mr.drops}")
        except Exception as exc:  # noqa: BLE001 - keep per-model failures explicit.
            print(f"[depth] MODEL FAILED: {model_id}: {exc}")
            summary["models"][model_id] = {"error": str(exc)}
            _atomic_write_json(
                out_path,
                {
                    "schema": SCHEMA,
                    "model": model_id,
                    "setup": setup,
                    "config": summary["config"],
                    "error": str(exc),
                },
            )

    summary["completed_at_iso"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(out_dir / "depth_audit_logitlens_summary.json", summary)
    print(f"[depth] wrote {out_dir / 'depth_audit_logitlens_summary.json'}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-wise logit-lens shadow-ambiguity depth audit.")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--layers", default="all", help="all, comma list, or ranges like 20-28")
    p.add_argument("--limit", type=int, default=0, help="cap samples; default 0 uses all")
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--k-support", type=int, default=K_SUPPORT_DEFAULT)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
