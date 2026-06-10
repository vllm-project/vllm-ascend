#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <npu_type> <num_npus> <with-device|without-device> <test> [test ...]"
  exit 1
fi

npu_type="$1"
num_npus="$2"
mode="$3"
shift 3
targets=("$@")

if [ "${mode}" != "with-device" ] && [ "${mode}" != "without-device" ]; then
  echo "Invalid mode: ${mode}"
  exit 1
fi

test_results=()
failed_logs=()
test_index=0
overall_status=0
pytest_log_dir="${RUNNER_TEMP:-/tmp}/selected-tests-${npu_type}-${num_npus}card"

mkdir -p "${pytest_log_dir}"

print_test_info() {
  echo -e "\033[1;34m=== TEST INFO ===\033[0m"
  echo -e "  \033[33mDevice:\033[0m ${npu_type}"
  if [ "${npu_type}" != "cpu" ]; then
    echo -e "  \033[33mNPU count:\033[0m ${num_npus}"
  fi
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
}

run_pytest_target() {
  local target="$1"
  test_index=$((test_index + 1))
  local log_name="${target}"
  log_name="${log_name#tests/}"
  log_name="${log_name%.py}"
  log_name="${log_name//[^a-zA-Z0-9_.-]/_}"
  local log_file="${pytest_log_dir}/${test_index}-${log_name}.log"
  echo "::group::${target}"
  echo -e "\033[1;34m=== Running target: ${target} ===\033[0m"
  set +e
  pytest -sv --color=yes "${target}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "::endgroup::"
  if [ "${status}" -eq 0 ]; then
    test_results+=("${target}|PASSED|${log_file}")
  else
    test_results+=("${target}|FAILED|${log_file}")
    failed_logs+=("${target}|${log_file}")
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
  fi
}

run_pytest_batch() {
  local target="$1"
  shift
  local batch_targets=("$@")
  test_index=$((test_index + 1))
  local log_file="${pytest_log_dir}/${test_index}-cpu-ut.log"

  echo "::group::${target}"
  echo -e "\033[1;34m=== Running target: ${target} ===\033[0m"
  set +e
  pytest -sv --color=yes "${batch_targets[@]}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "::endgroup::"
  if [ "${status}" -eq 0 ]; then
    test_results+=("${target}|PASSED|${log_file}")
  else
    test_results+=("${target}|FAILED|${log_file}")
    failed_logs+=("${target}|${log_file}")
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
  fi
}

print_test_info

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

print_summary
exit "${overall_status}"
