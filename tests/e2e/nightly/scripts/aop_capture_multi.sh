#!/bin/bash
# ============================================================
# aop_capture_multi.sh - Capture multi-node test result from pod logs
#
# Args: $1 = stream_logs outcome  (from steps.stream_logs.outcome)
#       $2 = log_prefix           (e.g. /tmp/vllm-xxx)
#
# Merges all pod logs into /tmp/test-logs/multi-node.log,
# then calls aop_capture.sh to parse.
# Writes outputs to $GITHUB_OUTPUT.
# ============================================================
set -euo pipefail

STREAM_OUTCOME="${1:-skipped}"
LOG_PREFIX="${2:-/tmp}"

mkdir -p /tmp/test-logs
rm -f /tmp/test-logs/multi-node.log

# Merge all pod logs into one file
for f in "${LOG_PREFIX}"_logs.txt; do
  if [ -f "$f" ]; then
    echo "===== $(basename "$f") =====" >> /tmp/test-logs/multi-node.log
    cat "$f" >> /tmp/test-logs/multi-node.log
  fi
done

# If no pod logs found, try wildcard
if [ ! -s /tmp/test-logs/multi-node.log ]; then
  for f in /tmp/*_logs.txt; do
    if [ -f "$f" ]; then
      echo "===== $(basename "$f") =====" >> /tmp/test-logs/multi-node.log
      cat "$f" >> /tmp/test-logs/multi-node.log
    fi
  done
fi

echo "============================================"
echo " Multi-node Test Result"
echo "   Stream logs : ${STREAM_OUTCOME}"
echo "============================================"

if [ -s /tmp/test-logs/multi-node.log ]; then
  echo "--- multi-node log tail (last 40 lines) ---"
  tail -n 40 /tmp/test-logs/multi-node.log
  echo "--- end ---"

  SUMMARY=$(grep -E '=+.*(passed|failed|error).*=+' /tmp/test-logs/multi-node.log | tail -1 || true)
  echo "pytest_summary=${SUMMARY}" >> "$GITHUB_OUTPUT"
  echo "pytest_summary: ${SUMMARY}"

  FAILURES=$(grep -c 'FAILED' /tmp/test-logs/multi-node.log || true)
  echo "pytest_failures=${FAILURES}" >> "$GITHUB_OUTPUT"
fi

# Final verdict based on stream_logs outcome
if [ "$STREAM_OUTCOME" = "failure" ]; then
  echo "result=failure" >> "$GITHUB_OUTPUT"
elif [ "$STREAM_OUTCOME" = "success" ]; then
  echo "result=success" >> "$GITHUB_OUTPUT"
else
  echo "result=skipped" >> "$GITHUB_OUTPUT"
fi
