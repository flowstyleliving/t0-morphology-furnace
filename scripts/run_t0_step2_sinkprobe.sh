#!/usr/bin/env bash
# t0 Step 2.2 — SinkProbe sink-score baseline across all 10 panel models.
#
# Phases:
#   0. SinkProbe-wrapper invariance gate (Mistral-7B + Llama-3.2-3B +
#      Llama-3.1-8B, 10 samples each). HARD precondition: any non-10/10
#      byte-identical gen_token_ids result aborts the sweep.
#   1. 10-model SinkProbe scoring → <run>/sinkprobe/<NAME>.sinkprobe.json
#
# Attention-mass + ‖V‖-weighted column-sum sink scores, 3 reductions each
# (sink_bos / sink_top1 / sink_topk_sum, k=4), prefix-forward, no head-select.
# Sequential (OOM-serialized on the M4). One forward/sample, no bootstrap.
#
# set -e halts on any failure; nothing auto-restarted. Refuses to overwrite
# an existing run with .sinkprobe.json artifacts.
#
# Usage: bash scripts/run_t0_step2_sinkprobe.sh
set -euo pipefail

# Portable: resolve repo root from this script's location (scripts/..).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="$REPO_ROOT/experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl"
OUT="$REPO_ROOT/experiments/t0-baselines/2026-05-16/run-01"
SINK_OUT="$OUT/sinkprobe"

# Same models + order as run_t0_step2_rauq.sh (Step 2.3 join parity).
# Llama-3.1-8B is RAUQ/SinkProbe baseline-only (not in the Step 1 panel).
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

if [ -d "$SINK_OUT" ] && [ -n "$(find "$SINK_OUT" -name '*.sinkprobe.json' 2>/dev/null)" ]; then
  echo "[step2-sink] FATAL: $SINK_OUT already has .sinkprobe.json artifacts; refusing to overwrite. Rename / remove before re-launch."
  exit 1
fi

mkdir -p "$SINK_OUT"
LOG="$OUT/step2-sinkprobe-pipeline.log"

{
  echo "[step2-sink] start: $(date)"
  echo "[step2-sink] host: $(hostname)  pid: $$"
  echo "[step2-sink] data=$DATA"
  echo "[step2-sink] out=$SINK_OUT"
  echo ""
  echo "[step2-sink] === Phase 0: SinkProbe-wrapper invariance gate ==="
} | tee -a "$LOG"

for M in \
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  "mlx-community/Llama-3.2-3B-Instruct-4bit" \
  "mlx-community/Llama-3.1-8B-Instruct-4bit"; do
  echo "" | tee -a "$LOG"
  echo "[step2-sink] invariance gate: $M  $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/sinkprobe_baseline.py \
    --model "$M" --data "$DATA" --invariance-check 10 --max-new-tokens 4 \
    >> "$LOG" 2>&1
done
echo "[step2-sink] Phase 0 PASSED (set -e would have aborted on any DIFF)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[step2-sink] === Phase 1: 10-model SinkProbe scoring ===" | tee -a "$LOG"
echo "[step2-sink] phase 1 start: $(date)" | tee -a "$LOG"
for M in "${MODELS[@]}"; do
  NAME="${M##*/}"
  echo "[step2-sink] SinkProbe: $M  $(date)" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/sinkprobe_baseline.py \
    --model "$M" --data "$DATA" \
    --out "$SINK_OUT/${NAME}.sinkprobe.json" \
    --max-new-tokens 4 \
    > "$SINK_OUT/${NAME}.log" 2>&1
done
echo "[step2-sink] phase 1 done: $(date)" | tee -a "$LOG"
echo "[step2-sink] all phases complete: $(date)" | tee -a "$LOG"
