#!/usr/bin/env bash
# t0 Step 2.1 — RAUQ-at-commit baseline across all 9 panel models.
#
# Phases:
#   0. RAUQ-wrapper invariance gate (Mistral 7B + Llama 3.2-3B + Llama 3.1-8B,
#      10 samples each). HARD precondition: any non-10/10 byte-identical
#      gen_token_ids result aborts the sweep — an unproven wrapper makes
#      every AUROC untrustworthy.
#   1. 10-model RAUQ scoring → <run>/rauq/<NAME>.rauq.json (+ .log)
#
# Both 1a (commit-only) and 1b (prompt-recurrence) are emitted per model so
# the methodological fork is a reported result. Sequential because OOM-
# serialized on the M4 (each MLX 7B-class model is 8–12 GB). One forward
# pass per sample, no bootstrap → much cheaper than the Step 1 calibrator
# sweep (rough estimate ≈ 1–2 h for all 9).
#
# set -e halts on any failure; nothing is auto-restarted. Refuses to
# overwrite an existing run with .rauq.json artifacts.
#
# Usage: bash scripts/run_t0_step2_rauq.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="$REPO_ROOT/experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl"
OUT="$REPO_ROOT/experiments/t0-baselines/2026-05-16/run-01"
RAUQ_OUT="$OUT/rauq"

# Step 1 nine-model panel (same order: small→large for OOM serialization),
# PLUS Llama-3.1-8B. The 8B is NOT in the Step 1 descriptive panel — it is
# added here as the RAUQ + SinkProbe reproduction target the plan pins
# (Step 2.1 / 2.2 verify). It therefore has no Step 1 "ours" coverage-matrix
# row; the Step 2.3 join must treat it as baseline-only (no head-to-head
# vs our cells on that model).
MODELS=(
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/Llama-3.1-8B-Instruct-4bit"
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
)

if [ -d "$RAUQ_OUT" ] && [ -n "$(find "$RAUQ_OUT" -name '*.rauq.json' 2>/dev/null)" ]; then
  echo "[step2-rauq] FATAL: $RAUQ_OUT already has .rauq.json artifacts; refusing to overwrite. Rename / remove before re-launch."
  exit 1
fi

mkdir -p "$RAUQ_OUT"
LOG="$OUT/step2-rauq-pipeline.log"

{
  echo "[step2-rauq] start: $(date)"
  echo "[step2-rauq] host: $(hostname)  pid: $$"
  echo "[step2-rauq] data=$DATA"
  echo "[step2-rauq] out=$RAUQ_OUT"
  echo ""
  echo "[step2-rauq] === Phase 0: RAUQ-wrapper invariance gate ==="
} | tee -a "$LOG"

for M in \
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  "mlx-community/Llama-3.2-3B-Instruct-4bit" \
  "mlx-community/Llama-3.1-8B-Instruct-4bit"; do
  echo "" | tee -a "$LOG"
  echo "[step2-rauq] invariance gate: $M  $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/rauq_at_commit.py \
    --model "$M" --data "$DATA" --invariance-check 10 --max-new-tokens 4 \
    >> "$LOG" 2>&1
done
echo "[step2-rauq] Phase 0 PASSED (set -e would have aborted on any DIFF)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[step2-rauq] === Phase 1: 10-model RAUQ scoring ===" | tee -a "$LOG"
echo "[step2-rauq] phase 1 start: $(date)" | tee -a "$LOG"
for M in "${MODELS[@]}"; do
  NAME="${M##*/}"
  echo "[step2-rauq] RAUQ: $M  $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/rauq_at_commit.py \
    --model "$M" --data "$DATA" \
    --out "$RAUQ_OUT/${NAME}.rauq.json" \
    --max-new-tokens 4 \
    > "$RAUQ_OUT/${NAME}.log" 2>&1
done
echo "[step2-rauq] phase 1 done: $(date)" | tee -a "$LOG"
echo "[step2-rauq] all phases complete: $(date)" | tee -a "$LOG"
