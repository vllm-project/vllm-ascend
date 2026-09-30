#!/bin/bash
# ============================================================
# aop_classify.sh - Check if failure is environmental
#
# Args: $1 = failed_test (from capture: "yaml", "pytest", "yaml,pytest")
#
# Only scans the log files that actually failed.
# Writes failure_type to $GITHUB_OUTPUT.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES="$SCRIPT_DIR/rules-env.txt"
LOG_DIR="/tmp/test-logs"
FAILED_TEST="${1:-yaml,pytest}"

check_log() {
  local log_file="$1"
  local label="$2"

  if [ ! -f "$log_file" ] || [ ! -s "$log_file" ]; then
    echo "  [$label] log empty/missing -> skipped"
    return 0
  fi

  local count
  count=$(grep -vE '^[[:space:]]*(#|$)' "$RULES" | grep -ciEf - "$log_file" 2>/dev/null || echo 0)

  echo "  [$label] env patterns matched: ${count}"

  if [ "$count" -gt 0 ]; then
    echo "  [$label] --- matches ---"
    grep -vE '^[[:space:]]*(#|$)' "$RULES" | grep -niEf - "$log_file" | head -10
    return 1
  fi
  return 0
}

echo "=== Failure Classification ==="
echo "[DEBUG] rules file : ${RULES}"
echo "[DEBUG] rules exists: $(test -f "$RULES" && echo yes || echo no)"
echo "[DEBUG] rules lines : $(grep -c . "$RULES" 2>/dev/null || echo 0)"
echo "[DEBUG] rules content:"
cat -n "$RULES" 2>/dev/null || echo "(file not found)"

ENV_FOUND=0
if [[ "$FAILED_TEST" == *pytest* ]]; then
  check_log "${LOG_DIR}/pytest-driven.log" "pytest-driven" || ENV_FOUND=1
fi
if [[ "$FAILED_TEST" == *yaml* ]]; then
  check_log "${LOG_DIR}/yaml-test.log"   "yaml-test"   || ENV_FOUND=1
fi

if [ "$ENV_FOUND" -eq 1 ]; then
  echo "=== Result: env_failure ==="
  echo "failure_type=env_failure" >> "$GITHUB_OUTPUT"
else
  echo "=== Result: not_env_failure ==="
  echo "failure_type=not_env_failure" >> "$GITHUB_OUTPUT"
fi
