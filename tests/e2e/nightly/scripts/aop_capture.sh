#!/bin/bash
# ============================================================
# aop_capture.sh - Capture test results from log files
#
# Called from _e2e_nightly_single_node.yaml [AOP] steps.
# Writes outputs to $GITHUB_OUTPUT.
#
# Usage: aop_capture.sh <yaml_outcome> <pytest_outcome>
# ============================================================
set -euo pipefail

YAML_OUTCOME="$1"
PYTEST_OUTCOME="$2"
LOG_DIR="/tmp/test-logs"

echo "============================================"
echo " Test Result Summary"
echo "   YAML-driven : ${YAML_OUTCOME:-skipped}"
echo "   Pytest-driven: ${PYTEST_OUTCOME:-skipped}"
echo "============================================"

parse_log() {
  local log_file="$1"
  local prefix="$2"

  if [ ! -f "$log_file" ]; then
    return 0
  fi

  echo ""
  echo "--- ${prefix} tail (last 40 lines) ---"
  tail -n 40 "$log_file"
  echo "--- end ---"

  local summary
  summary=$(grep -E '=+.*(passed|failed|error).*=+' "$log_file" | tail -1 || true)
  echo "${prefix}_summary=${summary}" >> "$GITHUB_OUTPUT"
  echo "${prefix}_summary: ${summary}"

  local failures
  failures=$(grep -c 'FAILED' "$log_file" || true)
  echo "${prefix}_failures=${failures}" >> "$GITHUB_OUTPUT"
}

parse_log "${LOG_DIR}/pytest-driven.log" "pytest"
parse_log "${LOG_DIR}/yaml-test.log"   "yaml"

# Final verdict + which test failed
if [ "$YAML_OUTCOME" = "failure" ] || [ "$PYTEST_OUTCOME" = "failure" ]; then
  echo "result=failure" >> "$GITHUB_OUTPUT"
  FAILED=""
  [ "$YAML_OUTCOME" = "failure" ]   && FAILED="${FAILED}yaml,"
  [ "$PYTEST_OUTCOME" = "failure" ] && FAILED="${FAILED}pytest,"
  echo "failed_test=${FAILED%,}" >> "$GITHUB_OUTPUT"
elif [ "$YAML_OUTCOME" = "success" ] || [ "$PYTEST_OUTCOME" = "success" ]; then
  echo "result=success" >> "$GITHUB_OUTPUT"
else
  echo "result=skipped" >> "$GITHUB_OUTPUT"
fi
