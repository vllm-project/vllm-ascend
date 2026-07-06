#!/bin/bash
# ============================================================
# aop_process.sh - Handle a recent real failure + auto bisect
#
# Args:
#   $1  failure_type
#   $2  commit_age_days
#   $3  runner
#   $4  tests
#   $5  config_file_path
#   $6  pytest_summary
#   $7  yaml_summary
#   $8  scene           (single_node | multi_node)
#   $9  bad_commit      (commit SHA, default HEAD)
#   $10 num_nodes       (multi_node only)
#   $11 coord_dir       (multi_node only)
#   $12 case_name       (optional)
# ============================================================
set -euo pipefail

FT="${1:-unknown}"
AGE="${2:-?}"
RUNNER="${3:-?}"
TESTS="${4:-}"
CONFIG="${5:-}"
PYTEST_SUMMARY="${6:-}"
YAML_SUMMARY="${7:-}"
SCENE="${8:-single_node}"
BAD_COMMIT="${9:-HEAD}"
NUM_NODES="${10:-}"
COORD_DIR="${11:-}"
NAME="${12:-}"

echo "================================================"
echo " PROCESS - needs attention"
echo "   Failure type : ${FT}"
echo "   Commit age   : ${AGE} days"
echo "   Runner       : ${RUNNER}"
echo "   Tests        : ${TESTS:-N/A}"
echo "   Config       : ${CONFIG:-N/A}"
echo "   Scene        : ${SCENE}"
echo "   Bad commit   : ${BAD_COMMIT}"
echo "   PyTest       : ${PYTEST_SUMMARY:-N/A}"
echo "   YAML         : ${YAML_SUMMARY:-N/A}"
echo "================================================"

echo "::group::Failed test details"
for f in /tmp/test-logs/pytest-driven.log /tmp/test-logs/yaml-test.log /tmp/test-logs/multi-node.log; do
  if [ -f "$f" ]; then
    grep -A 10 'FAILED' "$f" || true
  fi
done
echo "::endgroup::"

# =====================================================
# Auto bisect
# =====================================================

# Extract case_name if not provided (single_node requires it)
if [ -z "$NAME" ] && [ "$SCENE" = "single_node" ]; then
  if [ -n "$TESTS" ]; then
    # py-driven: tests/e2e/.../test_xxx.py  →  test_xxx
    NAME=$(basename "$TESTS" .py)
  elif [ -n "$CONFIG" ]; then
    # YAML-driven: Qwen3-32B-Int8.yaml  →  Qwen3-32B-Int8
    NAME=$(basename "$CONFIG" .yaml)
  fi

  if [ -z "$NAME" ]; then
    echo "WARNING: could not extract case_name, bisect may fail"
  else
    echo "Extracted name: ${NAME}"
  fi
fi

GOOD_TABLE="${GOOD_TABLE:-}"

BISECT_CMD=(
  python -m tests.e2e.nightly.bisect.auto_bisect
  --scene "${SCENE}"
  --bad-commit "${BAD_COMMIT}"
  --good-table "${GOOD_TABLE}"
)

[ -n "$CONFIG" ]    && BISECT_CMD+=(--config-yaml "$CONFIG")
[ -n "$NAME" ] && BISECT_CMD+=(--name "$NAME")
[ -n "$NUM_NODES" ] && BISECT_CMD+=(--num-nodes "$NUM_NODES")
[ -n "$COORD_DIR" ] && BISECT_CMD+=(--coord-dir "$COORD_DIR")

echo ""
echo "=== Running auto bisect ==="
echo "${BISECT_CMD[@]}"
"${BISECT_CMD[@]}"
