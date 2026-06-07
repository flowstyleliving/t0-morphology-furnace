#!/usr/bin/env bash
# Step 3.2 — TriviaQA calibrator attention panel sweep.
#
# Runs pri_calibrator --attention-with-v-norms --attention-only on all 9
# panel models against the TriviaQA paired-prompt dataset (n=100).
# Mirrors the Phase 3 invocation in run_t0_step1_pipeline.sh so results
# are directly comparable to the ANLI v_norms profile outputs.
#
# Usage:
#   bash scripts/run_triviaqa_calibrator_sweep.sh
#   DATA=<path> OUT=<dir> bash scripts/run_triviaqa_calibrator_sweep.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$REPO_ROOT/experiments/triviaqa-paired/2026-05-25/n100.jsonl}"
OUT="${OUT:-$REPO_ROOT/experiments/triviaqa-paired/2026-05-25/calibrator/v_norms}"
LOG_FILE="${LOG_FILE:-$OUT/sweep.log}"

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

mkdir -p "$OUT"

{
  echo "[triviaqa-sweep] start: $(date)"
  echo "[triviaqa-sweep] host: $(hostname)  pid: $$"
  echo "[triviaqa-sweep] data=$DATA"
  echo "[triviaqa-sweep] out=$OUT"
  echo "[triviaqa-sweep] n_models=${#MODELS[@]}"
} | tee -a "$LOG_FILE"

for M in "${MODELS[@]}"; do
  NAME="${M##*/}"
  PROFILE_OUT="$OUT/${NAME}.profile.json"

  if [[ -f "$PROFILE_OUT" ]]; then
    echo "[triviaqa-sweep] skip (exists): $NAME" | tee -a "$LOG_FILE"
    continue
  fi

  echo "" | tee -a "$LOG_FILE"
  echo "[triviaqa-sweep] === model: $M  $(date) ===" | tee -a "$LOG_FILE"

  if PYTHONUNBUFFERED=1 .venv/bin/python -u pri_calibrator.py \
    --model "$M" \
    --data "$DATA" \
    --out "$PROFILE_OUT" \
    --task-label "triviaqa_paired_n100_v_norms_step3" \
    --attention-with-v-norms --attention-only \
    --n-bootstrap 200 --max-new-tokens 4 \
    > "$OUT/${NAME}.log" 2>&1; then
    echo "[triviaqa-sweep] done: $NAME  $(date)" | tee -a "$LOG_FILE"
  else
    echo "[triviaqa-sweep] FAILED: $NAME  $(date)" | tee -a "$LOG_FILE"
  fi
done

echo "" | tee -a "$LOG_FILE"
echo "[triviaqa-sweep] all done: $(date)" | tee -a "$LOG_FILE"
