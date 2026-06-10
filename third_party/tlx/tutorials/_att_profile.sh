#!/usr/bin/env bash
# ATT profiling wrapper for the v9 TLX GEMM.
# Usage: _att_profile.sh BM BN BK NB NW GM [SIZE]
#
# Env setup notes (gfx950 / ROCm 7.2.4):
#  - rocprof-trace-decoder 0.1.6 (.so) staged in $RTD.
#  - Real libdw.so.1 (elfutils 0.190) staged in $RTD (system lacked it).
#  - torch bundles its own librocprofiler-sdk/register which double-configure
#    under rocprofv3 (error 16); we symlinked the system ones over torch's
#    (originals saved as *.orig in torch/lib).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTD=/home/stwinata/.local/rtd
OUT=/home/nod/agent_programs/v9/att_results
JSON=/home/nod/agent_programs/v9/att.json

rm -rf "$OUT"; mkdir -p "$OUT"
ROCPROF_ATT_LIBRARY_PATH="$RTD" \
LD_LIBRARY_PATH="$RTD:${LD_LIBRARY_PATH:-}" \
  rocprofv3 --advanced-thread-trace -i "$JSON" -d "$OUT" -- \
    python "$HERE/_prof_run.py" "$@"
echo "=== UI output dirs ==="
find "$OUT" -maxdepth 1 -name "ui_output*" -type d
