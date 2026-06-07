#!/usr/bin/env bash
# Regenerate all ACE/t=0 paper figures + tables from sealed profiles.
# Output: paper/t0/figures/out/*.pdf, *.png, *.tex
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../../.. && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="python"
fi

echo "==> Loader smoke test"
"$PY" load_ace_profiles.py | head -3
echo "==> Fig 1 (ANLI AUROC bars)"
"$PY" fig1_anli_auroc.py
echo "==> Fig 2 (cross-task paired)"
"$PY" fig2_cross_task.py
echo "==> Fig 3 (transfer matrix)"
"$PY" fig3_transfer_matrix.py
echo "==> Table 1 (pre-reg summary, hand-written static)"
cp table1_prereg_summary.tex out/table1_prereg_summary.tex
echo "==> Table 2 (winner cells)"
"$PY" table2_winner_cells.py
echo "==> Table 3 (baselines)"
"$PY" table3_baselines.py
echo ""
echo "Done. Outputs under paper/t0/figures/out/:"
ls -1 out/
