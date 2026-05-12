#!/bin/bash
# check_npu_env.sh — NPU environment sanity check
#
# Usage: check_npu_env.sh <TP_SIZE> [CANN_SET_ENV_PATH] [ATB_SET_ENV_PATH]
#
#   TP_SIZE           Required tensor-parallel size (e.g. 8 or 16)
#   CANN_SET_ENV_PATH Optional explicit path to CANN toolkit set_env.sh.
#                     If omitted, auto-detected under /usr/local/Ascend and ~/Ascend.
#                     Pass "none" to skip CANN sourcing.
#   ATB_SET_ENV_PATH  Optional path to ATB/NNAL set_env.sh
#                     Default: /usr/local/Ascend/nnal/atb/set_env.sh
#                     Pass "none" to skip ATB sourcing entirely.
#
# Exit codes: 0 = all checks passed, 1 = one or more checks failed

set -eo pipefail

TP_SIZE=${1:-8}
CANN_SET_ENV_ARG=${2:-}
ATB_SET_ENV_PATH=${3:-/usr/local/Ascend/nnal/atb/set_env.sh}

FAIL=0

# ── 1. Source CANN toolkit set_env.sh ────────────────────────────────────────
echo "==> [1/5] Sourcing CANN toolkit set_env.sh ..."
CANN_SET_ENV=""

if [ -n "$CANN_SET_ENV_ARG" ] && [ "$CANN_SET_ENV_ARG" != "none" ]; then
    # User provided explicit path
    if [ -f "$CANN_SET_ENV_ARG" ]; then
        CANN_SET_ENV="$CANN_SET_ENV_ARG"
    else
        echo "    ERROR: provided CANN set_env.sh not found: $CANN_SET_ENV_ARG"
        FAIL=1
    fi
elif [ "$CANN_SET_ENV_ARG" = "none" ]; then
    echo "    Skipped (passed 'none')"
else
    # Auto-detect
    for candidate in \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/latest/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
        "$HOME/Ascend/ascend-toolkit/set_env.sh" \
        "$HOME/Ascend/cann-8.5.1/set_env.sh"; do
        if [ -f "$candidate" ]; then
            CANN_SET_ENV="$candidate"
            break
        fi
    done

    if [ -z "$CANN_SET_ENV" ]; then
        echo "    ERROR: CANN set_env.sh not found. Pass the path as the second argument:"
        echo "      $0 $TP_SIZE /path/to/cann/set_env.sh"
        FAIL=1
    fi
fi

if [ -n "$CANN_SET_ENV" ]; then
    echo "    Found: $CANN_SET_ENV"
    # Use set +u to tolerate unbound variables in the sourced script
    set +u
    # shellcheck disable=SC1090
    source "$CANN_SET_ENV"
    set -u
    echo "    OK: CANN environment sourced"
fi

# ── 2. Source ATB/NNAL set_env.sh ────────────────────────────────────────────
echo "==> [2/5] Sourcing ATB/NNAL set_env.sh ..."
if [ "$ATB_SET_ENV_PATH" = "none" ]; then
    echo "    Skipped (passed 'none')"
elif [ -f "$ATB_SET_ENV_PATH" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$ATB_SET_ENV_PATH"
    set -u
    echo "    OK: ATB/NNAL environment sourced from $ATB_SET_ENV_PATH"
else
    echo "    WARNING: ATB/NNAL set_env.sh not found at $ATB_SET_ENV_PATH"
    echo "    If your environment requires it, re-run with the correct path:"
    echo "      $0 $TP_SIZE <cann-path> /path/to/atb/set_env.sh"
    echo "    Continuing without ATB/NNAL env ..."
fi

# ── 3. NPU device visibility ──────────────────────────────────────────────────
echo "==> [3/5] Checking NPU device visibility (npu-smi info) ..."
if ! npu-smi info; then
    echo "    ERROR: npu-smi info failed"
    FAIL=1
fi

# ── 4. torch_npu importable + NPU tensor creation + TP count ─────────────────
echo "==> [4/5] Checking torch_npu import, tensor creation, and TP count ..."
_NPU_CHECK_OUT=$(python - 2>&1 <<PY
import sys
import torch
try:
    import torch_npu
except ImportError as e:
    print(f"ERROR: cannot import torch_npu: {e}")
    sys.exit(1)

n = torch.npu.device_count()
if n == 0:
    print(f"ERROR: No NPU devices found (device_count={n})")
    sys.exit(1)

try:
    x = torch.tensor([1.0], dtype=torch.bfloat16).npu()
    assert x.device.type == "npu"
except Exception as e:
    print(f"ERROR: NPU tensor creation failed: {e}")
    sys.exit(1)

tp = int("${TP_SIZE}")
if n < tp:
    print(f"ERROR: Need {tp} NPUs for TP={tp}, only {n} available")
    sys.exit(1)

print(f"OK: {n} NPU(s) visible, tensor creation passed, TP={tp} satisfied")
PY
) || true
# Print only the last meaningful line (avoids leaking torch_npu traceback on non-NPU machines)
_LAST=$(echo "$_NPU_CHECK_OUT" | tail -1)
echo "    $_LAST"
if ! echo "$_LAST" | grep -q "^OK:"; then
    FAIL=1
fi

# ── 5. Version info (informational) ──────────────────────────────────────────
echo "==> [5/5] Version info (informational) ..."
python -c "import torch_npu; print('    torch_npu:', torch_npu.__version__)" 2>/dev/null || true
npu-smi info -t board -i 0 2>/dev/null | grep -i "cann\|driver\|firmware" | sed 's/^/    /' || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "==> NPU environment check PASSED"
else
    echo "==> NPU environment check FAILED — resolve the errors above before continuing"
    exit 1
fi
