#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
swa_init short
status=0
swa_request 0 || status=1
swa_finish 1 || status=1
exit "$status"
