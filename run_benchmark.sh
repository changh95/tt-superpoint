#!/bin/bash
# SuperPoint TT-NN benchmark driver -- the LEGACY (knob-off) path.
#
# Usage:
#   TT_METAL_DIR=/path/to/tt-metal DEVICE_ID=0 bash run_benchmark.sh
#
# Prints metrics; extract with:
#   grep -E "inference_speed|accuracy|keypoint_f1" run.log
#
# Interpreter: the tt-metal checkout's own python_env (ttnn, torch, transformers, loguru and
# pytest all live there); override with PYTHON=/path/to/python. The `device` fixture and the
# `--device-id` option come from ./conftest.py, so no tt-metal pytest rootdir is needed.
# TT_FUSED=0 on purpose (and the test pins fused=False itself): this script is the knob-off
# regression gate (DEVICE_VALIDATION.md §2(c)); TT_FUSED defaults to the fused path since
# 2026-09-13, which test_superpoint_fused measures.
set -u

DEVICE_ID="${DEVICE_ID:-0}"
TT_METAL_DIR="${TT_METAL_DIR:?TT_METAL_DIR must point at a built tt-metal checkout}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO_ROOT/run.log"
PYTHON="${PYTHON:-$TT_METAL_DIR/python_env/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "run_benchmark.sh: python interpreter not found: $PYTHON (build tt-metal's python_env or set PYTHON=)" >&2
    exit 2
fi

# PYTHONPATH: this repo FIRST (its regular `models` package must win over tt-metal's namespace
# `models`) + tt-metal (for ttnn runtime helpers; ttnn itself is installed in python_env).
export PYTHONPATH="$REPO_ROOT:$TT_METAL_DIR:$TT_METAL_DIR/ttnn"
export TT_METAL_HOME="$TT_METAL_DIR"
export ARCH_NAME="blackhole"
export SP_N_ITER="${SP_N_ITER:-10}"
export TT_FUSED=0
unset TT_FUSED_STAGES

cd "$REPO_ROOT" || exit 2
rm -f "$LOG"

"$PYTHON" -m pytest -s -q \
    --device-id="$DEVICE_ID" \
    models/tests/test_superpoint.py::test_superpoint_benchmark \
    2>&1 | tee "$LOG"

exit "${PIPESTATUS[0]}"
