#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_out="${A5_RESULT_DIR:-${task_dir}/results-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$task_out"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${task_out}/triton-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
"${PYTHON:-python3}" -u "${task_dir}/run_a5.py" --output "$task_out" "$@" 2>&1 | tee "${task_out}/run.log"
