#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

enable_coverage=false
include_cpu=false
skip_ut=false
npu_type=""
num_npus=""
mode=""

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Run full test suite locally (equivalent to the CI schedule/dispatch path).

Options:
  --enable-coverage    Run tests with coverage collection
  --include-cpu        Also include cpu-ut tests (npu_type=cpu) alongside NPU tests
  --skip-ut            Skip tests/ut tests, only run e2e tests
  --npu-type TYPE      NPU type: a2, a3, 310p, cpu (default: auto-detect)
  --num-npus N         Number of NPUs: 1, 2, 4, or 0 for cpu (default: auto-detect)
  -h, --help           Show this help message

Examples:
  # Auto-detect hardware and run all tests
  $0

  # Run with coverage, including cpu-ut
  $0 --enable-coverage --include-cpu

  # Run CPU tests only
  $0 --npu-type cpu --num-npus 0

  # Run with coverage on A2 single card, including cpu-ut
  $0 --enable-coverage --npu-type a2 --num-npus 1 --include-cpu
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --enable-coverage)
      enable_coverage=true
      shift
      ;;
    --include-cpu)
      include_cpu=true
      shift
      ;;
    --skip-ut)
      skip_ut=true
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
    return
  fi
  if echo "$chip_info" | grep -qi "910A\|910a"; then
    echo "a3"
    return
  fi
  if echo "$chip_info" | grep -qi "310P\|310p"; then
    echo "310p"
    return
  fi
  chip_info=$(npu-smi info 2>/dev/null | head -30 || true)
  if echo "$chip_info" | grep -qi "910B\|910b"; then
    echo "a2"
    return
  fi
  if echo "$chip_info" | grep -qi "910A\|910a"; then
    echo "a3"
    return
  fi
  if echo "$chip_info" | grep -qi "Ascend910"; then
    echo "a3"
    return
  fi
  if echo "$chip_info" | grep -qi "310P\|310p"; then
    echo "310p"
    return
  fi
  echo "cpu"
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
echo " NPU type:     ${npu_type}"
echo " NPU count:    ${num_npus}"
echo " Mode:         ${mode}"
echo " Coverage:     ${enable_coverage}"
echo " Include CPU:  ${include_cpu}"
echo " Skip UT:      ${skip_ut}"
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

extract_tests() {
  local filter_npu_type="$1"
  local filter_num_npus="$2"

  python3 -c "
import json, sys

groups = json.loads('''${test_groups_json}''')
skip_ut = ${skip_ut}

matched = []
for g in groups:
    if g['npu_type'] == '${filter_npu_type}' and g['num_npus'] == ${filter_num_npus}:
        matched.extend(g['tests'].split())

if not matched:
    sys.exit(0)

for t in matched:
    if skip_ut and t.startswith('tests/ut'):
        continue
    print(t)
"
}

coverage_flag=""
if [ "${enable_coverage}" = "true" ]; then
  coverage_flag="--enable-coverage"
fi

if [ "${include_cpu}" = "true" ] && [ "${npu_type}" != "cpu" ]; then
  echo ""
  echo ">>> Step 2a: Extracting NPU tests (${npu_type} x${num_npus}card)..."
  npu_tests=$(extract_tests "${npu_type}" "${num_npus}")
  npu_test_count=$(echo "$npu_tests" | grep -c . || true)

  echo ""
  echo ">>> Step 2b: Extracting CPU tests..."
  cpu_tests=$(extract_tests "cpu" "0")
  cpu_test_count=$(echo "$cpu_tests" | grep -c . || true)

  echo "Found ${npu_test_count} NPU test(s), ${cpu_test_count} CPU test(s)"

  if [ "${npu_test_count}" -gt 0 ]; then
    echo ""
    echo ">>> Step 3a: Running NPU tests (${npu_type} x${num_npus}card)..."
    ${SCRIPT_DIR}/run_selected_tests.sh \
      ${coverage_flag} \
      "${npu_type}" \
      "${num_npus}" \
      "with-device" \
      ${npu_tests}
  fi

  if [ "${cpu_test_count}" -gt 0 ]; then
    echo ""
    echo ">>> Step 3b: Running CPU tests..."
    ${SCRIPT_DIR}/run_selected_tests.sh \
      ${coverage_flag} \
      "cpu" \
      "0" \
      "without-device" \
      ${cpu_tests}
  fi
else
  echo ""
  echo ">>> Step 2: Extracting tests for ${npu_type} x${num_npus}card..."
  tests_for_group=$(extract_tests "${npu_type}" "${num_npus}")
  test_count=$(echo "$tests_for_group" | grep -c . || true)

  if [ "${test_count}" -eq 0 ]; then
    echo "No tests found for ${npu_type} x${num_npus}card."
    available=$(python3 -c "
import json
groups = json.loads('''${test_groups_json}''')
avail = set()
for g in groups:
    avail.add(f\"{g['npu_type']}x{g['num_npus']}\")
for a in sorted(avail):
    print(a)
")
    echo "Available groups:"
    echo "$available"
    exit 1
  fi

  echo "Found ${test_count} test(s) for ${npu_type} x${num_npus}card"

  echo ""
  echo ">>> Step 3: Running tests..."
  ${SCRIPT_DIR}/run_selected_tests.sh \
    ${coverage_flag} \
    "${npu_type}" \
    "${num_npus}" \
    "${mode}" \
    ${tests_for_group}
fi
