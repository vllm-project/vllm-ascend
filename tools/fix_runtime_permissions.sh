#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cann_root=${1:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}

for path in \
    "$repo_root/vllm_ascend/_cann_ops_custom/vendors" \
    "$cann_root/opp/vendors" \
    "$cann_root/tools/msmemscope/lib64"; do
    [[ -d "$path" ]] || continue
    find -H "$path" -type d -exec chmod a+rx,go-w {} +
    find -H "$path" -type f -exec chmod a+rX,go-w {} +
done

if [[ -d "$cann_root/tools/msmemscope" ]]; then
    chmod a+rx,go-w "$cann_root/tools/msmemscope"
fi

ir_env="$cann_root/share/info/ascendnpu-ir/bin/set_env.sh"
if [[ -f "$ir_env" ]]; then
    chmod a+r,go-w "$ir_env"
fi
