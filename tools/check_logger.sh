#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Check that vllm_ascend modules do not use init_logger(__name__).
#
# vllm's logging config registers a handler only for the "vllm" logger
# namespace.  Any logger created via init_logger(__name__) inside a
# vllm_ascend module ends up in the "vllm_ascend.*" namespace, which has
# no handler, so every log call is silently dropped.
#
# The correct pattern is:
#   from vllm.logger import logger
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$REPO_ROOT/vllm_ascend/"

VIOLATIONS=0

for FILE in $(find "$PATCH_DIR" -type f -name "*.py" 2>/dev/null); do
    [[ -f "$FILE" ]] || continue

    # Find lines that call init_logger(__name__)
    while IFS= read -r MATCH; do
        LINENUM=$(echo "$MATCH" | cut -d: -f1)
        LINE=$(echo "$MATCH" | cut -d: -f2-)
        if [[ $VIOLATIONS -eq 0 ]]; then
            echo ""
        fi
        echo "  $FILE:$LINENUM: $LINE"
        VIOLATIONS=$(( VIOLATIONS + 1 ))
    done < <(grep -n 'init_logger[[:space:]]*([[:space:]]*__name__[[:space:]]*)' "$FILE" 2>/dev/null || true)
done

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "Found $VIOLATIONS violation(s): init_logger(__name__) must not be used in vllm_ascend modules."
    echo ""
    echo "vllm's logging handler is registered only for the 'vllm' namespace."
    echo "Loggers created with init_logger(__name__) inside vllm_ascend end up"
    echo "in the 'vllm_ascend.*' namespace, which has no handler — all log"
    echo "messages are silently dropped."
    echo ""
    echo "Fix: replace"
    echo "   from vllm.logger import init_logger"
    echo "   logger = init_logger(__name__)"
    echo "with"
    echo "   from vllm.logger import logger"
    exit 1
fi

exit 0
