#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# GitCode skill environment pre-flight check.
# Checks: token, git, curl, python3, /tmp writable, git author (optional).
# Output: JSON report to stdout. Exit 0 = all required pass, exit 1 = failures.
#
# Usage:
#   bash preflight.sh                    # all checks including git author
#   bash preflight.sh --skip-git-author  # skip git user.name/email check

set -euo pipefail

SKIP_GIT_AUTHOR=false
for arg in "$@"; do
    case "$arg" in
        --skip-git-author) SKIP_GIT_AUTHOR=true ;;
    esac
done

# --- Check helpers ---

check_token() {
    if [ -n "${GITCODE_TOKEN:-}" ]; then
        printf '{"item":"token","status":"pass","detail":"env GITCODE_TOKEN (length=%d)","source":"env"}' "${#GITCODE_TOKEN}"
    else
        printf '{"item":"token","status":"fail","detail":"GITCODE_TOKEN not set"}'
    fi
}

check_binary() {
    local cmd="$1" label="${2:-$1}"
    if command -v "$cmd" >/dev/null 2>&1; then
        local ver
        ver=$("$cmd" --version 2>&1 | head -1 || true)
        printf '{"item":"%s","status":"pass","detail":"%s"}' "$label" "$ver"
    else
        printf '{"item":"%s","status":"fail","detail":"%s not found"}' "$label" "$cmd"
    fi
}

check_tmp() {
    if [ -w /tmp ]; then
        printf '{"item":"tmp","status":"pass","detail":"writable"}'
    else
        printf '{"item":"tmp","status":"fail","detail":"not writable"}'
    fi
}

check_git_author() {
    if [ "$SKIP_GIT_AUTHOR" = true ]; then
        printf '{"item":"git_author","status":"skip","detail":"--skip-git-author"}'
        return
    fi
    local name email
    name=$(git config --global user.name 2>/dev/null || true)
    email=$(git config --global user.email 2>/dev/null || true)
    if [ -n "$name" ] && [ -n "$email" ]; then
        printf '{"item":"git_author","status":"pass","detail":"%s <%s>","source":"global"}' "$name" "$email"
    else
        printf '{"item":"git_author","status":"fail","detail":"global user.name/email not configured"}'
    fi
}

# --- Run checks ---

RESULTS="["
RESULTS+="$(check_token),"
RESULTS+="$(check_binary git),"
RESULTS+="$(check_binary curl),"
RESULTS+="$(check_binary python3 python3),"
RESULTS+="$(check_tmp),"
RESULTS+="$(check_git_author)"
RESULTS+="]"

# Summary
FAIL_COUNT=$(echo "$RESULTS" | python3 -c "import json,sys; print(sum(1 for r in json.load(sys.stdin) if r['status']=='fail'))")
ALL_COUNT=$(echo "$RESULTS" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
PASS_COUNT=$((ALL_COUNT - FAIL_COUNT))

printf '{"results":%s,"summary":{"pass":%d,"fail":%d,"total":%d}}\n' \
    "$RESULTS" "$PASS_COUNT" "$FAIL_COUNT" "$ALL_COUNT"

exit "$FAIL_COUNT"
