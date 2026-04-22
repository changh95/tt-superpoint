#!/bin/bash
# SuperPoint TT-NN benchmark driver.
#
# Usage:
#   TT_METAL_DIR=/path/to/tt-metal DEVICE_ID=3 bash run_benchmark.sh
#
# Prints metrics; extract with:
#   grep -E "inference_speed|accuracy|keypoint_f1" run.log
set -u

DEVICE_ID="${DEVICE_ID:-0}"
TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must point at a built tt-metal checkout}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO_ROOT/run.log"

source "$HOME/.tenstorrent-venv/bin/activate"

# PYTHONPATH: this repo (for models.tt / models.reference imports) + tt-metal
# (for ttnn runtime).
export PYTHONPATH="$REPO_ROOT:$TT_METAL_DIR:$TT_METAL_DIR/ttnn"
export TT_METAL_HOME="$TT_METAL_DIR"
export ARCH_NAME="blackhole"
export SP_N_ITER="${SP_N_ITER:-10}"

cd "$REPO_ROOT" || exit 2
rm -f "$LOG"

pytest -s -q \
    --device-id="$DEVICE_ID" \
    models/tests/test_superpoint.py::test_superpoint_benchmark \
    2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
