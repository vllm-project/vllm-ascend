#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail
SWA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for script in 01_short.sh 02_long_prefill.sh 03_prefix_reuse.sh 04_concurrent.sh; do
    bash "$SWA_SCRIPT_DIR/$script"
done
