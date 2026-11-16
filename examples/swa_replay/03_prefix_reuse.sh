#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
swa_positive_integer PREFIX_REQUESTS "${PREFIX_REQUESTS:-12}"
swa_init prefix
status=0
for ((index=0; index<${PREFIX_REQUESTS:-12}; index++)); do
    swa_request "$index" || status=1
done
swa_finish "${PREFIX_REQUESTS:-12}" || status=1
exit "$status"
