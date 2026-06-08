#!/usr/bin/env bash
# Build all RPV (Readout Pseudo-Volume) workshop figures + table from comprehensive_outputs/.
# Uses the t0-morphology-furnace .venv (matplotlib). Reads JSON only; no model re-traces.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../../../.venv/bin/python"   # t0-morphology-furnace/.venv
"$PY" "$HERE/build_rpv_figs.py"
echo "figures + table written to: $HERE/out"
