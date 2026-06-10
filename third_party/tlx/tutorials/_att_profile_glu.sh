#!/usr/bin/env bash
# ATT profiling wrapper for the v9 fused GLU GEMM.
# Usage: _att_profile_glu.sh BM BN BK NW GM [MxNxK]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTD=/home/stwinata/.local/rtd
OUT=/home/nod/agent_programs/v9/att_glu_results
JSON=/home/nod/agent_programs/v9/att_glu.json

rm -rf "$OUT"; mkdir -p "$OUT"
ROCPROF_ATT_LIBRARY_PATH="$RTD" \
LD_LIBRARY_PATH="$RTD:${LD_LIBRARY_PATH:-}" \
  rocprofv3 --advanced-thread-trace -i "$JSON" -d "$OUT" -- \
    python "$HERE/_prof_run_glu.py" "$@"
echo "=== UI output dirs ==="
find "$OUT" -maxdepth 1 -name "ui_output*" -type d
