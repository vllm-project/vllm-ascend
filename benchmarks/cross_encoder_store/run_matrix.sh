#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROUNDS=${ROUNDS:-3}

for round in $(seq 1 "$ROUNDS"); do
  if ((round % 2 == 1)); then
    order=(p2p store)
  else
    order=(store p2p)
  fi
  for case_name in "${order[@]}"; do
    CASE="$case_name" ROUND="$round" "${SCRIPT_DIR}/run_2e2pd_case.sh"
  done
done
