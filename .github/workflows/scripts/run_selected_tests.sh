#!/usr/bin/env bash
set -euo pipefail

enable_coverage=false
if [ "${ENABLE_COVERAGE:-}" = "true" ]; then
  enable_coverage=true
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --enable-coverage)
      enable_coverage=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 [--enable-coverage] <npu_type> <num_npus> <with-device|without-device> [--timing] <test> [test ...]"
  exit 1
fi

npu_type="$1"
num_npus="$2"
mode="$3"
shift 3

record_timing=false
if [ "$1" = "--timing" ]; then
  record_timing=true
  shift
fi

targets=("$@")

if [ "${mode}" != "with-device" ] && [ "${mode}" != "without-device" ]; then
  echo "Invalid mode: ${mode}"
  exit 1
fi

test_results=()
failed_logs=()
timing_entries=()
test_index=0
overall_status=0
total_collected=0
total_passed=0
total_failed=0
total_errors=0
total_skipped=0
pytest_log_dir="${RUNNER_TEMP:-/tmp}/selected-tests-${npu_type}-${num_npus}card"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

mkdir -p "${pytest_log_dir}"

setup_coverage() {
  local target="$1"
  local test_basename="${target%.py}"
  test_basename="${test_basename//\//__}"
  test_basename="${test_basename//::/--}"
  local covdata_dir="${project_root}/tests/outputs/${test_basename}/covdata"
  mkdir -p "${covdata_dir}"
  export COVERAGE_FILE="${covdata_dir}/coverage"
  echo -e "  \033[33mCOVERAGE_FILE:\033[0m ${COVERAGE_FILE}"
}

setup_vllm_cache_root() {
  if [ "${CI:-}" != "true" ]; then
    return
  fi
  export VLLM_CACHE_ROOT
  VLLM_CACHE_ROOT="$(mktemp -d "${RUNNER_TEMP:-/tmp}/vllm-cache-${npu_type}-${num_npus}card.XXXXXX")"
  echo "Using vLLM cache root: ${VLLM_CACHE_ROOT}"
}

record_junit_results() {
  local target="$1"
  local junit_file="$2"
  if [ ! -s "${junit_file}" ]; then
    echo "::warning::JUnit report not found for ${target}; case count is unavailable."
    return
  fi

  local stats
  if ! stats="$(
    python - "${junit_file}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
if root.tag == "testsuite":
    suites = [root]
else:
    suites = list(root.findall("testsuite"))

def total(attribute):
    return sum(int(suite.attrib.get(attribute, 0)) for suite in suites)

tests = total("tests")
failures = total("failures")
errors = total("errors")
skipped = total("skipped")
passed = max(tests - failures - errors - skipped, 0)
print(f"{tests}\t{passed}\t{failures}\t{errors}\t{skipped}")
PY
  )"; then
    echo "::warning::Could not parse JUnit report for ${target}; case count is unavailable."
    return
  fi

  local collected passed failed errors skipped
  IFS=$'\t' read -r collected passed failed errors skipped <<< "${stats}"
  total_collected=$((total_collected + collected))
  total_passed=$((total_passed + passed))
  total_failed=$((total_failed + failed))
  total_errors=$((total_errors + errors))
  total_skipped=$((total_skipped + skipped))
  echo "Case count: collected=${collected}, passed=${passed}, failed=${failed}, errors=${errors}, skipped=${skipped}"
}

print_test_info() {
  echo -e "\033[1;34m=== TEST INFO ===\033[0m"
  echo -e "  \033[33mDevice:\033[0m ${npu_type}"
  if [ "${npu_type}" != "cpu" ]; then
    echo -e "  \033[33mNPU count:\033[0m ${num_npus}"
  fi
  echo -e "  \033[33mCoverage:\033[0m ${enable_coverage}"
  echo -e "  \033[33mTargets:\033[0m"
  for target in "${targets[@]}"; do
    echo -e "    \033[32m-\033[0m ${target}"
  done
  echo -e "\033[1;34m====================\033[0m"
}

print_summary() {
  echo -e "\033[1;34m=== TEST SUMMARY ===\033[0m"
  for result in "${test_results[@]}"; do
    IFS='|' read -r target status log_file <<< "${result}"
    echo -e "  ${status}: ${target}"
    echo -e "    log: ${log_file}"
  done
  if [ "${#failed_logs[@]}" -gt 0 ]; then
    echo -e "\033[1;31m=== FAILED TEST LOGS ===\033[0m"
    for failed in "${failed_logs[@]}"; do
      IFS='|' read -r target log_file <<< "${failed}"
      echo "::group::${target} failure log"
      cat "${log_file}"
      echo "::endgroup::"
    done
  fi
  echo -e "\033[1;34m=== PYTEST CASE COUNT ===\033[0m"
  echo "  collected: ${total_collected}"
  echo "  passed: ${total_passed}"
  echo "  failed: ${total_failed}"
  echo "  errors: ${total_errors}"
  echo "  skipped: ${total_skipped}"
}

run_pytest_target() {
  local target="$1"
  test_index=$((test_index + 1))
  local log_name="${target}"
  log_name="${log_name#tests/}"
  log_name="${log_name%.py}"
  log_name="${log_name//[^a-zA-Z0-9_.-]/_}"
  local log_file="${pytest_log_dir}/${test_index}-${log_name}.log"
  local junit_file="${pytest_log_dir}/${test_index}-${log_name}.xml"
  rm -f "${junit_file}"
  echo "::group::${target}"
  echo -e "\033[1;34m=== Running target: ${target} ===\033[0m"
  local start_time=0
  if [ "${record_timing}" = true ]; then
    start_time=$(date +%s%N)
  fi
  if [ "${enable_coverage}" = "true" ]; then
    setup_coverage "${target}"
    set +e
    python -m coverage run --rcfile="${project_root}/tests/coveragerc" -m pytest \
      -sv --color=yes --junitxml="${junit_file}" "${target}" 2>&1 | tee "${log_file}"
  else
    set +e
    pytest -sv --color=yes --junitxml="${junit_file}" "${target}" 2>&1 | tee "${log_file}"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  record_junit_results "${target}" "${junit_file}"
  # When a target fails, mark its covdata dir so the downstream coverage
  # assembler treats it as unusable and backfills from the OBS history
  # instead of shipping the failed run's partial coverage.
  if [ "${status}" -ne 0 ] && [ "${enable_coverage}" = "true" ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi
  if [ "${record_timing}" = true ]; then
    local elapsed_ns=$(( $(date +%s%N) - start_time ))
    local elapsed=$(( elapsed_ns / 1000000000 )).$(( (elapsed_ns % 1000000000) / 100000000 ))
    timing_entries+=("{\"name\":\"${target}\",\"passed\":$([ ${status} -eq 0 ] && echo true || echo false),\"elapsed\":${elapsed}}")
  fi
  echo "::endgroup::"
  if [ "${status}" -eq 0 ]; then
    test_results+=("${target}|PASSED|${log_file}")
  else
    test_results+=("${target}|FAILED|${log_file}")
    failed_logs+=("${target}|${log_file}")
    if [ "${record_timing}" != true ]; then
      print_summary
      exit "${status}"
    fi
  fi
}

run_pytest_batch() {
  local target="$1"
  shift
  local batch_targets=("$@")
  test_index=$((test_index + 1))
  local log_file="${pytest_log_dir}/${test_index}-cpu-ut.log"
  local junit_file="${pytest_log_dir}/${test_index}-cpu-ut.xml"
  rm -f "${junit_file}"

  echo "::group::${target}"
  echo -e "\033[1;34m=== Running target: ${target} ===\033[0m"
  local start_time=0
  if [ "${record_timing}" = true ]; then
    start_time=$(date +%s%N)
  fi
  if [ "${enable_coverage}" = "true" ]; then
    echo "DEBUG: Go to the [Coverage Branch] page."
    setup_coverage "cpu-ut"
    set +e
    python -m coverage run --rcfile="${project_root}/tests/coveragerc" -m pytest \
      -sv --color=yes --junitxml="${junit_file}" "${batch_targets[@]}" 2>&1 | tee "${log_file}"
  else
    set +e
    pytest -sv --color=yes --junitxml="${junit_file}" "${batch_targets[@]}" 2>&1 | tee "${log_file}"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  record_junit_results "${target}" "${junit_file}"
  if [ "${status}" -ne 0 ] && [ "${enable_coverage}" = "true" ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi
  if [ "${record_timing}" = true ]; then
    local elapsed_ns=$(( $(date +%s%N) - start_time ))
    local elapsed=$(( elapsed_ns / 1000000000 )).$(( (elapsed_ns % 1000000000) / 100000000 ))
    timing_entries+=("{\"name\":\"${target}\",\"passed\":$([ ${status} -eq 0 ] && echo true || echo false),\"elapsed\":${elapsed}}")
  fi
  echo "::endgroup::"
  if [ "${status}" -eq 0 ]; then
    test_results+=("${target}|PASSED|${log_file}")
  else
    test_results+=("${target}|FAILED|${log_file}")
    failed_logs+=("${target}|${log_file}")
    if [ "${record_timing}" != true ]; then
      print_summary
      exit "${status}"
    fi
  fi
}

print_timing_json() {
  if [ "${#timing_entries[@]}" -eq 0 ]; then
    return
  fi
  local json="["
  local i=0
  for entry in "${timing_entries[@]}"; do
    if [ "${i}" -gt 0 ]; then
      json+=","
    fi
    json+="${entry}"
    i=$((i + 1))
  done
  json+="]"
  echo "${json}" > "${pytest_log_dir}/test_timing_data.json"
  echo -e "\033[1;34m=== Timing data written to ${pytest_log_dir}/test_timing_data.json ===\033[0m"
}

print_test_info
setup_vllm_cache_root

if [ "${npu_type}" = "cpu" ]; then
  run_pytest_batch "cpu-ut (${#targets[@]} targets)" "${targets[@]}"
elif [ "${mode}" = "with-device" ]; then
  aclgraph_capture_replay="tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py"
  run_aclgraph_capture_replay=0
  for target in "${targets[@]}"; do
    if [ "${target}" = "${aclgraph_capture_replay}" ]; then
      run_aclgraph_capture_replay=1
      continue
    fi
    run_pytest_target "${target}"
  done
  if [ "${run_aclgraph_capture_replay}" = "1" ]; then
    pip uninstall -y triton-ascend triton
    run_pytest_target "${aclgraph_capture_replay}"
  fi
else
  for target in "${targets[@]}"; do
    run_pytest_target "${target}"
  done
fi

print_timing_json
print_summary
exit "${overall_status}"
