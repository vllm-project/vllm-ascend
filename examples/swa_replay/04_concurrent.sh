#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
swa_positive_integer CONCURRENCY "${CONCURRENCY:-4}"
swa_positive_integer NUM_REQUESTS "${NUM_REQUESTS:-8}"
swa_init concurrent

swa_worker() {
    local index status=0
    for ((index=$1; index<${NUM_REQUESTS:-8}; index+=${CONCURRENCY:-4})); do
        swa_request "$index" || status=1
    done
    return "$status"
}

pids=()
for ((worker=0; worker<${CONCURRENCY:-4} && worker<${NUM_REQUESTS:-8}; worker++)); do
    swa_worker "$worker" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
swa_finish "${NUM_REQUESTS:-8}" || status=1
exit "$status"
