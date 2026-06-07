#!/usr/bin/env bash
# HISTORICAL PREP SCRIPT, not the sealed reproduction path.
#
# t0 Step 1 pipeline — clean & complete the descriptive panel data + run the
# attention-only calibrator across all 9 panel models on both V-norm and
# multi-step variants. Sequential because OOM-serialized on the M4 (each
# MLX 7B-class model takes 8–12 GB).
#
# Phases:
#   1. invariance probe (Mistral 7B + Llama 3B) — sanity: fp32 cast is observational
#   2. 9-model descriptive panel re-run with fp32 fix → run-03 CSVs
#   3. V-norm calibrator sweep (9 models × --attention-with-v-norms --attention-only)
#   4. multi-step calibrator sweep (9 models × --attention-multistep --attention-only)
#
# Total wall ≈ 5–6 h. set -e halts on any failure; nothing is auto-restarted.
# Refuses to overwrite an existing run-03 with CSV artifacts.
#
# This script is retained for audit context. It depends on prep data and helper
# scripts that are not part of the clean sealed archive. Use
# `scripts/run_t0_sealed_sweep.sh` to reproduce the sealed ACE/T0 run.
#
# Historical usage: bash scripts/run_t0_step1_pipeline.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f scripts/invariance_probe_inter_head.py || ! -f scripts/run_inter_head_panel.sh ]]; then
  echo "[pipeline] HISTORICAL PREP SCRIPT: required prep helpers are not included in this clean archive." >&2
  echo "[pipeline] Use scripts/run_t0_sealed_sweep.sh for sealed reproduction." >&2
  exit 2
fi

DATA="$REPO_ROOT/experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl"
PANEL_OUT="$REPO_ROOT/experiments/inter-head-disagreement/2026-05-15/run-03"
CALIBRATOR_OUT="$REPO_ROOT/experiments/t0-prep-calibrator-sweep/2026-05-16/run-01"

MODELS=(
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
)

# Refuse to overwrite an existing panel run-03 (CSV artifacts present).
if [ -d "$PANEL_OUT" ] && [ -n "$(find "$PANEL_OUT" -name '*.csv' 2>/dev/null)" ]; then
  echo "[pipeline] FATAL: $PANEL_OUT already contains CSV artifacts; refusing to overwrite. Rename / remove before re-launch."
  exit 1
fi

mkdir -p "$PANEL_OUT" "$CALIBRATOR_OUT/v_norms" "$CALIBRATOR_OUT/multistep"
LOG="$CALIBRATOR_OUT/pipeline.log"

{
  echo "[pipeline] start: $(date)"
  echo "[pipeline] host: $(hostname)  pid: $$"
  echo "[pipeline] data=$DATA"
  echo "[pipeline] panel_out=$PANEL_OUT"
  echo "[pipeline] calibrator_out=$CALIBRATOR_OUT"
  echo ""
  echo "[pipeline] === Phase 1: invariance probe (fp32 observational check) ==="
} | tee -a "$LOG"

for M in "mlx-community/Mistral-7B-Instruct-v0.3-4bit" "mlx-community/Llama-3.2-3B-Instruct-4bit"; do
  NAME="${M##*/}"
  echo "" | tee -a "$LOG"
  echo "[pipeline] invariance probe: $M  start: $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/invariance_probe_inter_head.py \
    --model "$M" --data "$DATA" --limit 10 --max-new-tokens 4 \
    >> "$LOG" 2>&1
done

echo "" | tee -a "$LOG"
echo "[pipeline] === Phase 2: 9-model descriptive panel into $PANEL_OUT ===" | tee -a "$LOG"
echo "[pipeline] phase 2 start: $(date)" | tee -a "$LOG"
bash scripts/run_inter_head_panel.sh "$DATA" "$PANEL_OUT" "$PANEL_OUT/run.log"
echo "[pipeline] phase 2 done: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[pipeline] === Phase 3: V-norm calibrator sweep (9 models) ===" | tee -a "$LOG"
echo "[pipeline] phase 3 start: $(date)" | tee -a "$LOG"
for M in "${MODELS[@]}"; do
  NAME="${M##*/}"
  echo "[pipeline] V-norm calibration: $M  $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u pri_calibrator.py \
    --model "$M" --data "$DATA" \
    --out "$CALIBRATOR_OUT/v_norms/${NAME}.profile.json" \
    --task-label "anli_r1_n200_v_norms_t0_prep" \
    --attention-with-v-norms --attention-only \
    --n-bootstrap 200 --max-new-tokens 4 \
    > "$CALIBRATOR_OUT/v_norms/${NAME}.log" 2>&1
done
echo "[pipeline] phase 3 done: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[pipeline] === Phase 4: multi-step calibrator sweep (9 models) ===" | tee -a "$LOG"
echo "[pipeline] phase 4 start: $(date)" | tee -a "$LOG"
for M in "${MODELS[@]}"; do
  NAME="${M##*/}"
  echo "[pipeline] multi-step calibration: $M  $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u pri_calibrator.py \
    --model "$M" --data "$DATA" \
    --out "$CALIBRATOR_OUT/multistep/${NAME}.profile.json" \
    --task-label "anli_r1_n200_multistep_t0_prep" \
    --attention-multistep --attention-only \
    --n-bootstrap 200 --max-new-tokens 5 \
    > "$CALIBRATOR_OUT/multistep/${NAME}.log" 2>&1
done
echo "[pipeline] phase 4 done: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[pipeline] all phases complete: $(date)" | tee -a "$LOG"
