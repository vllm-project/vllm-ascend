#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Set BASE_DIR to the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INNER_SCRIPT="$BASE_DIR/../../omni/accelerators/sched/global_proxy/global_proxy.sh"

if [ ! -f "$INNER_SCRIPT" ]; then
  echo "Error: $INNER_SCRIPT not found."
  exit 1
fi

echo "Wrapper: Running $INNER_SCRIPT with arguments: $@"
bash "$INNER_SCRIPT" "$@"
EXIT_CODE=$?
echo "Wrapper: $INNER_SCRIPT exited with code $EXIT_CODE"
exit $EXIT_CODE
