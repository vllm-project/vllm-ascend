#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  off) ENABLE_MOONEP=false ;;
  slot) ENABLE_MOONEP=true ;;
  *) echo "Usage: MODEL=/path/to/weights VLLM_PYTHON=/path/to/python $0 off|slot [vllm arguments...]" >&2; exit 2 ;;
esac
shift

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
VLLM_PYTHON=${VLLM_PYTHON:-$WORKSPACE/vllm/.venv/bin/python}
: "${MODEL:?Set MODEL to the model weight directory}"
[[ -x "$VLLM_PYTHON" ]] || { echo "Independent validation Python is missing: $VLLM_PYTHON" >&2; exit 2; }

# Verify bindings without importing torch, vllm or native libraries.
"$VLLM_PYTHON" -I -B - "$WORKSPACE" "$ENABLE_MOONEP" <<'PY'
import importlib.util
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
packages = [("vllm", "vllm"), ("vllm_ascend", "vllm-ascend")]
if sys.argv[2] == "true":
    packages.append(("ascend_moonep", "ascend-moonep"))
for module, repo in packages:
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise SystemExit(f"Missing installed package: {module}")
    origin = pathlib.Path(spec.origin).resolve()
    if not origin.is_relative_to(root / repo):
        raise SystemExit(f"{module} resolves outside selected workspace: {origin}")
PY

if [[ "$ENABLE_MOONEP" == true ]]; then
  export MOONEP_TRANSPORT_TYPE=${MOONEP_TRANSPORT_TYPE:-mte}
  export MOONEP_SHMEM_BOOTSTRAP_TYPE=${MOONEP_SHMEM_BOOTSTRAP_TYPE:-ipport}
  export MOONEP_PREFETCH_SIDE_STREAM_OVERLAP=${MOONEP_PREFETCH_SIDE_STREAM_OVERLAP:-dispatch}
  export LD_LIBRARY_PATH="$WORKSPACE/ascend-moonep/ascend_moonep/lib:${LD_LIBRARY_PATH:-}"
fi

PROFILE_ARGS=()
if [[ "${PROFILE:-0}" == 1 ]]; then
  : "${OUT_DIR:?Set an absolute, new OUT_DIR for profiler output}"
  [[ "$OUT_DIR" == /* ]] || { echo "OUT_DIR must be absolute" >&2; exit 2; }
  [[ ! -e "$OUT_DIR/profiling" ]] || { echo "Profiler output already exists: $OUT_DIR/profiling" >&2; exit 2; }
  mkdir -p -- "$OUT_DIR/profiling"
  PROFILE_ARGS=(--profiler-config.profiler=torch
    --profiler-config.torch_profiler_dir="$OUT_DIR/profiling"
    --profiler-config.torch_profiler_with_stack=false)
fi
if [[ -n "${LOG_FILE:-}" ]]; then
  [[ "$LOG_FILE" == /* ]] || { echo "LOG_FILE must be absolute" >&2; exit 2; }
  mkdir -p -- "$(dirname -- "$LOG_FILE")"
  (set -o noclobber; : > "$LOG_FILE") || { echo "Service log already exists: $LOG_FILE" >&2; exit 2; }
  exec > >(tee -a -- "$LOG_FILE") 2>&1
fi

# Avoid importing a sibling source tree accidentally through an empty sys.path entry.
cd -- "$HERE"
exec "$VLLM_PYTHON" -I -B -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --data-parallel-size "${DP_SIZE:-8}" \
  --enable-expert-parallel --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-batched-tokens "${BATCHED_TOKENS:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --no-enable-prefix-caching \
  --additional-config "{\"enable_moonep\": $ENABLE_MOONEP}" "${PROFILE_ARGS[@]}" "$@"
