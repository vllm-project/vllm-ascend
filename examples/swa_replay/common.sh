#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SWA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

swa_positive_integer() {
    if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer: %s\n' "$1" "$2" >&2
        return 2
    fi
}

swa_init() {
    SWA_KIND="$1"
    SWA_PYTHON="${PYTHON_BIN:-python3}"
    command -v "$SWA_PYTHON" >/dev/null
    command -v curl >/dev/null
    SWA_BASE_URL="${BASE_URL:-http://127.0.0.1:8989}"
    SWA_MODEL="${MODEL:-dsv41}"
    SWA_REPEATS="${LONG_REPEATS:-256}"
    SWA_MAX_TOKENS="${MAX_TOKENS:-32}"
    swa_positive_integer LONG_REPEATS "$SWA_REPEATS"
    swa_positive_integer MAX_TOKENS "$SWA_MAX_TOKENS"
    SWA_RUN_DIR="${OUTPUT_DIR:-/tmp/swa-curl-results}/${SWA_KIND}-$(date +%Y%m%d-%H%M%S)-$$"
    mkdir -p "$SWA_RUN_DIR"
    SWA_STARTED_NS="$("$SWA_PYTHON" -c 'import time; print(time.time_ns())')"
    SWA_LOG_START=0
    if [[ -n "${SERVICE_LOG:-}" && -r "$SERVICE_LOG" ]]; then
        SWA_LOG_START="$(wc -l < "$SERVICE_LOG")"
    fi
    printf 'Case=%s model=%s endpoint=%s\nResults: %s\n' "$SWA_KIND" "$SWA_MODEL" "$SWA_BASE_URL" "$SWA_RUN_DIR"
}

swa_request() {
    local index="$1" request_dir curl_exit=0
    request_dir="$(printf '%s/request-%03d' "$SWA_RUN_DIR" "$index")"
    mkdir -p "$request_dir" || return
    "$SWA_PYTHON" "$SWA_SCRIPT_DIR/request_tools.py" build \
        --kind "$SWA_KIND" --index "$index" --model "$SWA_MODEL" \
        --repeats "$SWA_REPEATS" --max-tokens "$SWA_MAX_TOKENS" \
        --output "$request_dir/request.json" || return
    curl --silent --show-error --connect-timeout "${CURL_CONNECT_TIMEOUT:-10}" \
        --max-time "${CURL_MAX_TIME:-600}" \
        --header 'Content-Type: application/json' \
        --data-binary "@$request_dir/request.json" \
        --output "$request_dir/response.body" \
        --write-out '{"http_code":"%{http_code}","time_total_s":%{time_total},"response_bytes":%{size_download}}\n' \
        "${SWA_BASE_URL%/}/v1/chat/completions" \
        > "$request_dir/curl.json" 2> "$request_dir/curl.stderr" || curl_exit=$?
    "$SWA_PYTHON" "$SWA_SCRIPT_DIR/request_tools.py" capture \
        --directory "$request_dir" --index "$index" --kind "$SWA_KIND" \
        --curl-exit "$curl_exit"
}

swa_finish() {
    local status=0
    "$SWA_PYTHON" "$SWA_SCRIPT_DIR/request_tools.py" report \
        --directory "$SWA_RUN_DIR" --expected "$1" --kind "$SWA_KIND" \
        --started-ns "$SWA_STARTED_NS" --base-url "$SWA_BASE_URL" --model "$SWA_MODEL" || status=$?
    if [[ -n "${SERVICE_LOG:-}" && -r "$SERVICE_LOG" ]]; then
        printf '\nNew replay messages since this script started (a worker logs activation only once):\n'
        awk -v start="$SWA_LOG_START" \
            'NR > start && /SWA decoder bounded replay active|SWA bounded replay is active|Decoder bounded replay requires synchronized/ {print NR ":" $0}' \
            "$SERVICE_LOG"
    fi
    return "$status"
}
