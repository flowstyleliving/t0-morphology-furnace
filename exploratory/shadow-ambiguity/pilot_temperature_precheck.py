#!/usr/bin/env python3
"""Pilot: label-free temperature-sweep pre-check for research-candidate #10
(shadow-ambiguity).

Cheapest falsifier (no labels, no training): do the readout-morphology
statistics (fisher_eff_rank / spectral_entropy / shadow_logvol /
participation_ratio) carry information INDEPENDENT of token surprise on REAL
model logits, or are they just confidence in disguise? Decision rule from the
candidate: if every headline statistic collapses onto surprise across commits
at the deployment temperature (Spearman |rho| >= 0.9) the candidate dies here,
before any labeled run.

Design:
  * Real commit instants = every interior position of natural text (each
    position's next-token distribution is one commit). One 4-bit MLX model.
  * F_c = W_uᵀ (diag(p) − p pᵀ) W_u is built from the top-K support rows of the
    (dequantized) unembedding via pri_runtime.OutputProjection.get_rows.
  * Spectrum via the K×K dual: F_c = (R W_s)ᵀ(R W_s) with R = sqrt(B),
    B = diag(p_s) − p_s p_sᵀ, so nonzero eig(F_c) = eig(R·Gram·R), Gram = W_s W_sᵀ.
    eff_rank / entropy use only the active (nonzero) spectrum -> the dual is
    exact; shadow_logvol uses the full d-dim spectrum (dual eigenvalues padded
    with d−K structural zeros), faithful to the d×d F_c.
  * Statistics are the reviewed reference impls from test_shadow_ambiguity.py.
  * NUMERICAL HYGIENE: commits with non-finite logits or non-finite W_u rows or
    non-finite spectra are DROPPED and COUNTED (no silent corruption); rho is
    computed over valid commits only.

Run (from repo root, in the t0 .venv):
    .venv/bin/python exploratory/shadow-ambiguity/pilot_temperature_precheck.py
Override model: PILOT_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit .venv/bin/python ...
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so we can import the sibling contract module

from test_shadow_ambiguity import (  # noqa: E402
    fisher_eff_rank,
    fisher_spectral_entropy,
    participation_ratio,
    shadow_logvol_post_rank,
)

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from pri_runtime import OutputProjection  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


MODEL = os.environ.get("PILOT_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")
K_SUPPORT = 512
TS = [0.5, 0.7, 1.0, 1.4, 2.0]
T1_INDEX = TS.index(1.0)
MAX_COMMITS = 400
SEED = 20260607

SNIPPETS = [
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "Water is made of two hydrogen atoms and one oxygen atom.",
    "In machine learning, gradient descent minimizes a loss function by taking steps proportional to the negative of the gradient.",
    "She opened the heavy wooden door and saw, to her complete surprise, that the room was entirely empty.",
    "The meeting was rescheduled to next Thursday because several key stakeholders were traveling abroad.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight.",
    "Once upon a time, in a small village nestled between two mountains, there lived an old clockmaker.",
    "The stock market fell sharply on Monday after the central bank unexpectedly raised interest rates.",
    "To make a basic vinaigrette, whisk together three parts oil and one part vinegar with a pinch of salt.",
    "Despite the rain, the team decided to continue the hike, knowing the summit was only an hour away.",
    "The novel explores themes of memory, loss, and the strange ways that time reshapes our understanding.",
    "A prime number is a natural number greater than one that has no positive divisors other than one and itself.",
    "He paused, uncertain whether to tell her the truth or to keep the secret a little longer.",
    "The committee will review the proposal and announce its decision sometime in the coming weeks.",
    "Quantum entanglement describes a correlation between particles that persists regardless of the distance separating them.",
    "After years of research, the scientists finally understood why the ancient bridge had not collapsed.",
]


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def fc_full_spectrum(W_s: np.ndarray, p_s: np.ndarray, d: int) -> np.ndarray:
    """Full d-dim spectrum of F_c = W_sᵀ (diag(p_s) − p_s p_sᵀ) W_s, via the
    K×K dual (exact nonzero spectrum) padded with d−K structural zeros.
    Assumes finite W_s, p_s (caller guards)."""
    B = np.diag(p_s) - np.outer(p_s, p_s)
    B = 0.5 * (B + B.T)
    wB, QB = np.linalg.eigh(B)
    wB = np.clip(wB, 0.0, None)
    # On the local NumPy/Accelerate stack, np.matmul can emit spurious
    # divide/overflow/invalid warnings here even when inputs and outputs are
    # finite. einsum computes the same products without tripping those flags.
    R = np.einsum("ij,jk->ik", QB * np.sqrt(wB), QB.T, optimize=True)
    gram = np.einsum("ik,jk->ij", W_s, W_s, optimize=True)
    RG = np.einsum("ij,jk->ik", R, gram, optimize=True)
    M = np.einsum("ij,jk->ik", RG, R, optimize=True)
    M = 0.5 * (M + M.T)
    eig = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    pad = max(d - eig.size, 0)
    return np.concatenate([eig, np.zeros(pad)]) if pad else eig


def main() -> int:
    rng = np.random.default_rng(SEED)
    print(f"[pilot] loading {MODEL} ...", flush=True)
    model, tok = load(MODEL)
    op = OutputProjection(model)
    d, V = int(op.hidden_size), int(op.vocab_size)
    print(f"[pilot] d_model={d} vocab={V} K_support={K_SUPPORT} Ts={TS}", flush=True)

    commits = []
    n_drop_logit = 0
    _diag = False
    for text in SNIPPETS:
        ids = tok.encode(text)
        if len(ids) < 4:
            continue
        logits = model(mx.array(ids)[None])
        logits = np.array(logits.astype(mx.float32))[0]   # (seq, V)
        if not _diag:
            fin = np.isfinite(logits)
            safe = np.where(fin, logits, np.nan)
            print(f"[diag] first-snippet logits: finite_frac={fin.mean():.6f} "
                  f"min={np.nanmin(safe):.2f} max={np.nanmax(safe):.2f}", flush=True)
            _diag = True
        for i in range(1, logits.shape[0] - 1):           # interior positions
            li = logits[i].astype(np.float64)
            if not np.isfinite(li).all():
                n_drop_logit += 1
                continue
            commits.append(li)
    print(f"[pilot] dropped {n_drop_logit} positions with non-finite logits", flush=True)
    if len(commits) > MAX_COMMITS:
        sel = rng.choice(len(commits), MAX_COMMITS, replace=False)
        commits = [commits[j] for j in sel]
    N = len(commits)
    print(f"[pilot] {N} candidate commit instants", flush=True)

    surprise = np.zeros((N, len(TS)))
    effrank = np.zeros((N, len(TS)))
    entropy = np.zeros((N, len(TS)))
    logvol1 = np.zeros((N, len(TS)))
    partic = np.zeros((N, len(TS)))
    valid = np.zeros(N, dtype=bool)
    n_drop_ws = 0
    n_drop_spec = 0
    _wsdiag = False

    for n, logit in enumerate(commits):
        topK = np.argpartition(-logit, K_SUPPORT)[:K_SUPPORT]
        W_s = op.get_rows(topK)
        if W_s is None:
            n_drop_ws += 1
            continue
        W_s = W_s.astype(np.float64)
        if not np.isfinite(W_s).all():
            n_drop_ws += 1
            continue
        if not _wsdiag:
            print(f"[diag] first W_s: shape={tuple(W_s.shape)} absmax={np.abs(W_s).max():.4g}", flush=True)
            _wsdiag = True
        row = {k: np.zeros(len(TS)) for k in ("s", "er", "en", "lv", "pr")}
        ok = True
        for ti, T in enumerate(TS):
            p = softmax_np(logit / T)
            spec = fc_full_spectrum(W_s, p[topK], d)
            if not np.isfinite(spec).all():
                ok = False
                break
            row["s"][ti] = -math.log(max(float(p.max()), 1e-300))
            row["er"][ti] = fisher_eff_rank(spec)
            row["en"][ti] = fisher_spectral_entropy(spec)
            row["lv"][ti] = shadow_logvol_post_rank(spec, r=1)
            row["pr"][ti] = participation_ratio(spec)
        if not ok:
            n_drop_spec += 1
            continue
        surprise[n], effrank[n], entropy[n] = row["s"], row["er"], row["en"]
        logvol1[n], partic[n] = row["lv"], row["pr"]
        valid[n] = True
        if (n + 1) % 50 == 0:
            print(f"  ...{n + 1}/{N}", flush=True)

    nv = int(valid.sum())
    print(f"[pilot] valid commits: {nv}/{N} (dropped W_s={n_drop_ws}, spec={n_drop_spec})", flush=True)
    surprise, effrank, entropy = surprise[valid], effrank[valid], entropy[valid]
    logvol1, partic = logvol1[valid], partic[valid]
    N = nv
    if N < 30:
        print("[pilot] too few valid commits to trust rho; aborting.")
        return 1

    stats = {
        "fisher_eff_rank": effrank,
        "spectral_entropy": entropy,
        "shadow_logvol_r1": logvol1,
        "participation_ratio": partic,
    }

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        return float(spearmanr(a, b).statistic)

    s_T1 = surprise[:, T1_INDEX]
    s_grid = surprise.reshape(-1)
    report = {
        "model": MODEL, "d_model": d, "vocab": V, "n_commits": N,
        "n_dropped": {"logit": n_drop_logit, "ws": n_drop_ws, "spec": n_drop_spec},
        "K_support": K_SUPPORT, "Ts": TS,
        "decision_rule": "|rho|>=0.9 across commits at T=1 => confidence-in-disguise",
        "results": {},
    }
    print("\n=== Spearman rho(surprise, stat) ===", flush=True)
    print(f"{'statistic':22s} | acrossCommits@T=1 | grid(commit×T) | withinCommit/acrossT")
    for name, arr in stats.items():
        r_T1 = rho(s_T1, arr[:, T1_INDEX])
        r_grid = rho(s_grid, arr.reshape(-1))
        wc = [
            rho(surprise[n], arr[n])
            for n in range(N)
            if np.std(arr[n]) > 0 and np.std(surprise[n]) > 0
        ]
        r_wc = float(np.nanmean(wc)) if wc else float("nan")
        verdict = "SURVIVES" if abs(r_T1) < 0.9 else "COLLAPSES"
        report["results"][name] = {
            "rho_acrossCommits_T1": r_T1,
            "rho_grid": r_grid,
            "rho_withinCommit_acrossT_mean": r_wc,
            "verdict": verdict,
        }
        print(f"{name:22s} |   {r_T1:+.3f} [{verdict:8s}] |   {r_grid:+.3f}     |   {r_wc:+.3f}")

    survivors = [k for k, v in report["results"].items() if abs(v["rho_acrossCommits_T1"]) < 0.9]
    print("\n=== PRE-CHECK VERDICT ===")
    if survivors:
        print(f"SURVIVES: {survivors} decouple from surprise at T=1 (|rho|<0.9) -> a labeled pilot is justified.")
        report["verdict"] = f"SURVIVES via {survivors}"
    else:
        print("ALL headline statistics COLLAPSE onto surprise at T=1 (|rho|>=0.9) -> confidence-in-disguise; candidate dies at the pre-check.")
        report["verdict"] = "DIES (all collapse)"

    slug = MODEL.split("/")[-1].replace(":", "_")
    outp = os.path.join(HERE, f"pilot_results__{slug}.json")
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[pilot] wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
