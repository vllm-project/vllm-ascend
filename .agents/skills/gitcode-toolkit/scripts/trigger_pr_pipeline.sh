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
# Trigger GitCode PR CI pipeline by posting a "compile" comment.
#
# Usage:
#   bash trigger_pr_pipeline.sh --repo cann/ops-math --pr 123
#   bash trigger_pr_pipeline.sh --repo cann/ops-math --pr 123 --comment "compile"
#
# Requires: GITCODE_TOKEN environment variable.

set -euo pipefail

# --- Parse args ---
REPO=""
PR_NUMBER=""
COMMENT="compile"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --pr) PR_NUMBER="$2"; shift 2 ;;
        --comment) COMMENT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done

if [ -z "$REPO" ] || [ -z "$PR_NUMBER" ]; then
    echo "Usage: bash trigger_pr_pipeline.sh --repo <owner/repo> --pr <N> [--comment <text>]"
    exit 2
fi

if [ -z "${GITCODE_TOKEN:-}" ]; then
    echo '{"status":"fail","detail":"GITCODE_TOKEN not set"}'
    exit 1
fi

OWNER="${REPO%/*}"
REPO_NAME="${REPO#*/}"

API_URL="https://api.gitcode.com/api/v5/repos/${OWNER}/${REPO_NAME}/pulls/${PR_NUMBER}/comments?access_token=${GITCODE_TOKEN}"

# Escape comment for JSON
COMMENT_ESCAPED=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$COMMENT")

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json' \
    -d "{\"body\": ${COMMENT_ESCAPED}}" \
    --connect-timeout 30 --max-time 60 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "201" ]; then
    COMMENT_ID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id','?'))" 2>/dev/null || echo "?")
    echo "{\"status\":\"ok\",\"http_code\":201,\"comment_id\":${COMMENT_ID},\"pr\":\"${OWNER}/${REPO_NAME}#${PR_NUMBER}\",\"comment\":\"${COMMENT}\"}"
else
    echo "{\"status\":\"fail\",\"http_code\":${HTTP_CODE},\"detail\":$(echo "$BODY" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))")}"
    exit 1
fi
