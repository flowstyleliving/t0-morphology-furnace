#!/usr/bin/env bash
# PRI t0 SEALED SWEEP — t=0 commit-step attention calibration.
#
# Pre-registration: T0_ACE_PRE_REGISTRATION_PLAN.md  (frozen 2026-05-26)
# Instrument:       t=0 prefill-last-position attention (--t0-commit)
# Panel:            21-cell ATTENTION_PANEL_T0_WITH_V_NORMS (--attention-with-v-norms)
# Bootstrap:        n=1000 (--n-bootstrap 1000)
# Datasets:
#   ANLI:     experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl
#             data_hash d1a3aed5e86af05c4b7bd459bb5938bbcca7ab6c758c855e1bce3f938b62f48e
#   TriviaQA: experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl
#             data_hash f2f870a7e2feb2c711b2a782f6aa6040233c8915bb066e87421e85f4778b3149
#
# Gate (E_A1): >= 7/9 models OOB CI_lo > 0.50 on at least one cell (ANLI)
# Transfer (E_A2): count exact (metric, sign) matches ANLI -> TriviaQA
#
# Usage:
#   bash scripts/run_t0_sealed_sweep.sh            # runs both ANLI + TriviaQA
#   DATASET=anli bash scripts/run_t0_sealed_sweep.sh   # ANLI only
#   DATASET=triviaqa bash scripts/run_t0_sealed_sweep.sh   # TriviaQA only
#
# DO NOT re-specify --t0-commit with a different dataset or seed.
# DO NOT use --n-bootstrap < 1000 (pre-reg requires 1000).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEALED_DATA_DIR="$REPO_ROOT/experiments/t0-sealed/2026-05-26/data"
OUT_BASE="$REPO_ROOT/experiments/t0-sealed/2026-05-26/profiles"
DATASET="${DATASET:-both}"

# ─── Datasets ────────────────────────────────────────────────────────────────
ANLI_DATA="$SEALED_DATA_DIR/anli_R1_seed20260526_n200.jsonl"
TRIVIAQA_DATA="$SEALED_DATA_DIR/triviaqa_paired_seed20260526_n100.jsonl"

ANLI_HASH="d1a3aed5e86af05c4b7bd459bb5938bbcca7ab6c758c855e1bce3f938b62f48e"
TRIVIAQA_HASH="f2f870a7e2feb2c711b2a782f6aa6040233c8915bb066e87421e85f4778b3149"

# ─── Data integrity gate ──────────────────────────────────────────────────────
verify_hash() {
  local file="$1" expected="$2" label="$3"
  if [[ ! -f "$file" ]]; then
    echo "[t0-sealed-sweep] ABORT: $label not found: $file" >&2; exit 1
  fi
  local actual
  actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[t0-sealed-sweep] ABORT: $label hash mismatch" >&2
    echo "[t0-sealed-sweep]   expected: $expected" >&2
    echo "[t0-sealed-sweep]   got:      $actual" >&2
    echo "[t0-sealed-sweep] This sweep is NOT part of the sealed pre-registration." >&2
    exit 1
  fi
  echo "[t0-sealed-sweep] hash OK: $label"
}

if [[ "$DATASET" == "anli" || "$DATASET" == "both" ]]; then
  verify_hash "$ANLI_DATA" "$ANLI_HASH" "ANLI R1 n=200"
fi
if [[ "$DATASET" == "triviaqa" || "$DATASET" == "both" ]]; then
  verify_hash "$TRIVIAQA_DATA" "$TRIVIAQA_HASH" "TriviaQA n=100"
fi

# ─── Panel (9 models, t=0 sealed panel) ───────────────────────────────────────
MODELS=(
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
run_sweep() {
  local task_label="$1"
  local data_path="$2"
  local out_dir="$3"
  local log_file="$out_dir/sweep.log"

  mkdir -p "$out_dir"
  {
    echo "[t0-sealed-sweep] task=$task_label  start: $(date)"
    echo "[t0-sealed-sweep] host: $(hostname)  pid: $$"
    echo "[t0-sealed-sweep] data=$data_path"
    echo "[t0-sealed-sweep] out=$out_dir"
    echo "[t0-sealed-sweep] n_models=${#MODELS[@]}"
    echo "[t0-sealed-sweep] instrument=t0-commit  n_bootstrap=1000"
  } | tee -a "$log_file"

  for M in "${MODELS[@]}"; do
    NAME="${M##*/}"
    PROFILE_OUT="$out_dir/${NAME}.profile.json"

    if [[ -f "$PROFILE_OUT" ]]; then
      echo "[t0-sealed-sweep] skip (exists): $NAME" | tee -a "$log_file"
      continue
    fi

    echo "" | tee -a "$log_file"
    echo "[t0-sealed-sweep] === model: $M  $(date) ===" | tee -a "$log_file"

    if PYTHONUNBUFFERED=1 .venv/bin/python -u pri_calibrator.py \
        --model "$M" \
        --data "$data_path" \
        --out "$PROFILE_OUT" \
        --task-label "$task_label" \
        --t0-commit \
        --attention-with-v-norms \
        --n-bootstrap 1000 \
        > "$out_dir/${NAME}.log" 2>&1; then
      echo "[t0-sealed-sweep] done: $NAME  $(date)" | tee -a "$log_file"
    else
      echo "[t0-sealed-sweep] FAILED: $NAME  $(date)" | tee -a "$log_file"
    fi
  done

  echo "" | tee -a "$log_file"
  echo "[t0-sealed-sweep] $task_label sweep complete: $(date)" | tee -a "$log_file"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

if [[ "$DATASET" == "anli" || "$DATASET" == "both" ]]; then
  run_sweep \
    "t0_sealed_anli_R1_t0_n200" \
    "$ANLI_DATA" \
    "$OUT_BASE/anli"
fi

if [[ "$DATASET" == "triviaqa" || "$DATASET" == "both" ]]; then
  run_sweep \
    "t0_sealed_triviaqa_t0_n100" \
    "$TRIVIAQA_DATA" \
    "$OUT_BASE/triviaqa"
fi

echo "[t0-sealed-sweep] all datasets complete: $(date)"
