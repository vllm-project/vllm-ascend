#!/bin/bash
# ============================================================
# aop_skip.sh - Log skip reason and show failure details
#
# Args: failure_type last_status last_date age_days
#       pytest_summary yaml_summary
# ============================================================
set -euo pipefail

FT="${1:-unknown}"
LAST_STATUS="${2:-?}"
LAST_DATE="${3:-?}"
AGE="${4:-?}"
PYTEST_SUMMARY="${5:-}"
YAML_SUMMARY="${6:-}"

case "$FT" in
  env_failure) REASON="environment issue" ;;
  *)           REASON="last run > 3 days ago" ;;
esac

echo "================================================"
echo " SKIP - no further action"
echo "   Failure type : ${FT}"
echo "   Last status  : ${LAST_STATUS}"
echo "   Last date    : ${LAST_DATE}"
echo "   Age (days)   : ${AGE}"
echo "   Reason       : ${REASON}"
echo "   PyTest       : ${PYTEST_SUMMARY:-N/A}"
echo "   YAML         : ${YAML_SUMMARY:-N/A}"
echo "================================================"

echo "::group::Failed test details"
for f in /tmp/test-logs/pytest-driven.log /tmp/test-logs/yaml-test.log; do
  if [ -f "$f" ]; then
    grep -A 10 'FAILED' "$f" || true
  fi
done
echo "::endgroup::"
