#!/usr/bin/env bash
# auto_resume_wrapper.sh - Auto-resume wrapper for SOTA Tuning Loop

set -uo pipefail

WORK_DIR=""
INTERRUPTED=0

handle_sigint() {
    INTERRUPTED=1
    echo "Received SIGINT, stopping wrapper without relaunch."
    exit 130
}

trap handle_sigint INT

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [ -z "$WORK_DIR" ] || [ "$#" -eq 0 ]; then
    echo "Usage: $0 --work-dir <dir> [--] <full_claude_command...>"
    echo "Example: $0 --work-dir ./tuning_run/my_run -- claude -p \"...\" --max-turns 0"
    exit 1
fi

CLAUDE_CMD_ARRAY=("$@")

MAX_RETRIES=20
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "=== Tuning Session Attempt $((RETRY_COUNT+1))/$MAX_RETRIES ==="

    if [ -f "$WORK_DIR/final_report.md" ]; then
        echo "✅ final_report.md found in $WORK_DIR. Tuning loop completed successfully!"
        exit 0
    fi

    echo "Tuning not yet complete (final_report.md not found)."

    if [ -f "$WORK_DIR/ledger.md" ]; then
        echo "Found existing ledger.md in $WORK_DIR. Resuming run..."
    else
        echo "No existing ledger found. Starting fresh run..."
    fi

    echo "Launching Claude CLI..."
    "${CLAUDE_CMD_ARRAY[@]}"
    EXIT_CODE=$?
    echo "Claude process exited with code $EXIT_CODE"

    if [ "$EXIT_CODE" -eq 130 ] || [ "$INTERRUPTED" -eq 1 ]; then
        echo "Claude session ended by SIGINT. Wrapper exits without retry."
        exit 130
    fi

    if [ -f "$WORK_DIR/final_report.md" ]; then
        echo "✅ Tuning loop finished during this attempt."
        exit 0
    fi

    echo "⚠️ Claude process terminated before completion."
    echo "Waiting 5 seconds before resuming..."
    sleep 5

    RETRY_COUNT=$((RETRY_COUNT+1))
done

echo "❌ Exceeded maximum retries ($MAX_RETRIES). Exiting."
exit 1
