#!/bin/bash
# check_roots.sh — Verify vLLM and vllm-ascend implementation roots and runtime import
#
# Usage: check_roots.sh <VLLM_SRC> <VLLM_ASCEND_SRC> [WORK_DIR]
#   VLLM_SRC         Path to vLLM source root (e.g. /vllm-workspace/vllm)
#   VLLM_ASCEND_SRC  Path to vllm-ascend source root (e.g. /vllm-workspace/vllm-ascend)
#   WORK_DIR         Working directory for vllm serve (default: /workspace)

set -euo pipefail

VLLM_SRC=${1:?"Usage: check_roots.sh <VLLM_SRC> <VLLM_ASCEND_SRC> [WORK_DIR]"}
VLLM_ASCEND_SRC=${2:?"Usage: check_roots.sh <VLLM_SRC> <VLLM_ASCEND_SRC> [WORK_DIR]"}
WORK_DIR=${3:-/workspace}

FAIL=0

echo "==> [1/4] vLLM source root: $VLLM_SRC"
if [ -d "$VLLM_SRC" ]; then
    git -C "$VLLM_SRC" status -s
else
    echo "    ERROR: directory not found: $VLLM_SRC"
    FAIL=1
fi

echo ""
echo "==> [2/4] vllm-ascend source root: $VLLM_ASCEND_SRC"
if [ -d "$VLLM_ASCEND_SRC" ]; then
    git -C "$VLLM_ASCEND_SRC" status -s
else
    echo "    ERROR: directory not found: $VLLM_ASCEND_SRC"
    FAIL=1
fi

echo ""
echo "==> [3/4] Runtime vllm import path ..."
VLLM_FILE=$(python - 2>&1 <<'PY'
import sys
try:
    import vllm
    print(vllm.__file__)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PY
) || true
if echo "$VLLM_FILE" | grep -q "^ERROR:"; then
    echo "    WARNING: could not import vllm (CANN env may not be sourced yet)"
    echo "    $VLLM_FILE"
else
    echo "    vllm.__file__ = $VLLM_FILE"
fi

echo ""
echo "==> [4/4] Working directory: $WORK_DIR"
if [ -d "$WORK_DIR" ]; then
    echo "    OK: $WORK_DIR exists"
else
    echo "    WARNING: $WORK_DIR does not exist — create it before running vllm serve"
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "==> Root check PASSED"
else
    echo "==> Root check FAILED — resolve the errors above before continuing"
    exit 1
fi
