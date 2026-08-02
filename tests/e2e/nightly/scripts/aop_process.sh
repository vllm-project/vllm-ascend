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
#   $13 soc             (optional)
#   $14-$23             optional bisect controls
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
SOC="${13:-}"
BISECT_GOOD_COMMIT="${14:-}"
BISECT_FAIL_CONFIRM_RETRIES="${15:-}"
BISECT_TRIAL_TIMEOUT="${16:-}"
BISECT_BARRIER_TIMEOUT="${17:-}"
BISECT_NO_VERIFY_GOOD="${18:-}"
BISECT_NO_VERIFY_BAD="${19:-}"
BISECT_FORCE_INITIAL_BUILD="${20:-}"
BISECT_CONFIG_BASE_PATH="${21:-}"

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
for f in /tmp/test-logs/pytest-driven.log /tmp/test-logs/yaml-test.log /tmp/test-logs/multi-node.log /tmp/test-logs/model-accuracy-*.log; do
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
if [ -n "$GOOD_TABLE" ]; then
  ENV_TABLE="${ENV_TABLE:-$(dirname "$GOOD_TABLE")/env_table.csv}"
else
  ENV_TABLE="${ENV_TABLE:-}"
fi
RUN_LINK="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-vllm-project/vllm-ascend}/actions/runs/${GITHUB_RUN_ID:-unknown}"

if [ -n "$GOOD_TABLE" ] && [ -n "$NAME" ]; then
  echo "Recording current failure runtime environment: ${ENV_TABLE}"
  python tests/e2e/nightly/scripts/update_good_table.py \
    --cache-csv "$GOOD_TABLE" \
    --env-table "$ENV_TABLE" \
    --status failure \
    --test-name "$NAME" \
    --test-path "${CONFIG:-$TESTS}" \
    --config-base-path "${CONFIG_BASE_PATH:-$BISECT_CONFIG_BASE_PATH}" \
    --scene "$SCENE" \
    --run-link "$RUN_LINK" || true
fi

BISECT_CMD=(
  python -m tools.bisect.auto_bisect
  --scene "${SCENE}"
  --bad-commit "${BAD_COMMIT}"
  --good-table "${GOOD_TABLE}"
)

[ -n "$CONFIG" ]    && BISECT_CMD+=(--config-yaml "$CONFIG")
[ -n "$NAME" ] && BISECT_CMD+=(--name "$NAME")
[ -n "$SOC" ] && BISECT_CMD+=(--soc "$SOC")
[ -n "$ENV_TABLE" ] && BISECT_CMD+=(--env-table "$ENV_TABLE")
[ -n "$NUM_NODES" ] && BISECT_CMD+=(--num-nodes "$NUM_NODES")
[ -n "$COORD_DIR" ] && BISECT_CMD+=(--coord-dir "$COORD_DIR")
[ -n "$BISECT_GOOD_COMMIT" ] && BISECT_CMD+=(--good-commit "$BISECT_GOOD_COMMIT")
[ -n "$BISECT_FAIL_CONFIRM_RETRIES" ] && BISECT_CMD+=(--fail-confirm-retries "$BISECT_FAIL_CONFIRM_RETRIES")
[ -n "$BISECT_TRIAL_TIMEOUT" ] && BISECT_CMD+=(--trial-timeout-s "$BISECT_TRIAL_TIMEOUT")
[ -n "$BISECT_BARRIER_TIMEOUT" ] && BISECT_CMD+=(--barrier-timeout-s "$BISECT_BARRIER_TIMEOUT")
[ "$BISECT_NO_VERIFY_GOOD" = "true" ] && BISECT_CMD+=(--no-verify-good)
[ "$BISECT_NO_VERIFY_BAD" = "true" ] && BISECT_CMD+=(--no-verify-bad)
[ "$BISECT_FORCE_INITIAL_BUILD" = "true" ] && BISECT_CMD+=(--force-initial-build)
# AOP always uses the conservative strategy so bisect jumps cannot reuse a
# stale native build. This is an internal correctness policy, not a user knob.
BISECT_CMD+=(--native-check since-build)
[ -n "$BISECT_CONFIG_BASE_PATH" ] && BISECT_CMD+=(--config-base-path "$BISECT_CONFIG_BASE_PATH")

echo ""
echo "=== Running auto bisect ==="
echo "${BISECT_CMD[@]}"
"${BISECT_CMD[@]}"
