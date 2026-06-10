#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

enable_coverage=false
npu_type=""
num_npus=""
mode=""

usage() {
  cat <<EOF
Usage: $0 [OPTIONS] --npu-type <TYPE> --num-npus <N>

Run full test suite locally (equivalent to the CI schedule/dispatch path).

Options:
  --enable-coverage    Run tests with coverage collection
  --npu-type TYPE      NPU type: a2, a3, 310p, cpu (default: auto-detect)
  --num-npus N         Number of NPUs: 1, 2, 4, or 0 for cpu (default: auto-detect)
  -h, --help           Show this help message

Examples:
  # Auto-detect hardware and run all tests
  $0

  # Run with coverage on A2 single card
  $0 --enable-coverage --npu-type a2 --num-npus 1

  # Run CPU tests only
  $0 --npu-type cpu --num-npus 0
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --enable-coverage)
      enable_coverage=true
      shift
      ;;
    --npu-type)
      npu_type="$2"
      shift 2
      ;;
    --num-npus)
      num_npus="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

detect_npu() {
  if ! command -v npu-smi &>/dev/null; then
    echo "cpu"
    return
  fi
  local chip_info
  chip_info=$(npu-smi info -t board 2>/dev/null | head -20 || true)
  if echo "$chip_info" | grep -qi "910B\|910b"; then
    echo "a2"
  elif echo "$chip_info" | grep -qi "910A\|910a"; then
    echo "a3"
  elif echo "$chip_info" | grep -qi "310P\|310p"; then
    echo "310p"
  else
    echo "cpu"
  fi
}

count_npus() {
  if [ "${npu_type}" = "cpu" ]; then
    echo "0"
    return
  fi
  if command -v npu-smi &>/dev/null; then
    npu-smi info -l 2>/dev/null | grep -c "NPU" || echo "0"
  else
    echo "0"
  fi
}

if [ -z "$npu_type" ]; then
  npu_type=$(detect_npu)
  echo "Auto-detected NPU type: ${npu_type}"
fi

if [ -z "$num_npus" ]; then
  num_npus=$(count_npus)
  echo "Auto-detected NPU count: ${num_npus}"
fi

if [ "${npu_type}" = "cpu" ]; then
  mode="without-device"
  num_npus=0
else
  mode="with-device"
fi

echo "============================================"
echo " Running full test suite locally"
echo " NPU type:  ${npu_type}"
echo " NPU count: ${num_npus}"
echo " Mode:      ${mode}"
echo " Coverage:  ${enable_coverage}"
echo "============================================"

cd "${PROJECT_ROOT}"

echo ""
echo ">>> Step 1: Selecting all tests..."
select_output=$(python3 "${SCRIPT_DIR}/select_tests.py" \
  --changed-files vllm_ascend/dummy.py \
  --run-all-modules 2>&1)

has_tests=$(echo "$select_output" | grep "^has_tests=" | head -1 | cut -d= -f2)
if [ "${has_tests}" != "true" ]; then
  echo "No tests found. Exiting."
  exit 0
fi

test_groups_json=$(echo "$select_output" | grep "^test_groups=" | head -1 | cut -d= -f2-)

echo ""
echo ">>> Step 2: Extracting tests for ${npu_type} x${num_npus}card..."

tests_for_group=$(python3 -c "
import json, sys

groups = json.loads('''${test_groups_json}''')
npu_type = '${npu_type}'
num_npus = ${num_npus:-0}

matched = []
for g in groups:
    if g['npu_type'] == npu_type and g['num_npus'] == num_npus:
        matched.extend(g['tests'].split())

if not matched:
    available = set()
    for g in groups:
        available.add(f\"{g['npu_type']}x{g['num_npus']}\")
    print(f'No tests found for {npu_type}x{num_npus}.', file=sys.stderr)
    print(f'Available groups: {sorted(available)}', file=sys.stderr)
    sys.exit(1)

for t in matched:
    print(t)
")

if [ $? -ne 0 ]; then
  echo "Failed to extract tests. Showing all available groups:"
  echo "$select_output"
  exit 1
fi

test_count=$(echo "$tests_for_group" | wc -l)
echo "Found ${test_count} test(s) for ${npu_type} x${num_npus}card"

echo ""
echo ">>> Step 3: Running tests..."

coverage_flag=""
if [ "${enable_coverage}" = "true" ]; then
  coverage_flag="--enable-coverage"
fi

${SCRIPT_DIR}/run_selected_tests.sh \
  ${coverage_flag} \
  "${npu_type}" \
  "${num_npus}" \
  "${mode}" \
  ${tests_for_group}
