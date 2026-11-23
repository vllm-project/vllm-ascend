#!/usr/bin/env bash
set -euo pipefail

enable_coverage=false
if [ "${ENABLE_COVERAGE:-}" = "true" ]; then
  enable_coverage=true
fi

# Set to true temporarily to run all selected targets before reporting failure.
continue_on_error=false

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

  # torch.utils.cpp_extension uses a file baton in its build directory.
  # A cancelled self-hosted job can leave that baton behind and make every
  # later NPU job sharing the default cache wait forever before ninja starts.
  export TORCH_EXTENSIONS_DIR
  TORCH_EXTENSIONS_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/torch-extensions-${npu_type}-${num_npus}card.XXXXXX")"
  echo "Using Torch extensions directory: ${TORCH_EXTENSIONS_DIR}"
}

terminate_process_group() {
  local process_group="$1"
  if ! kill -0 -- "-${process_group}" 2>/dev/null; then
    return
  fi

  echo "Cleaning up processes left by pytest (process group ${process_group})"
  kill -TERM -- "-${process_group}" 2>/dev/null || true
  for _ in {1..10}; do
    if ! kill -0 -- "-${process_group}" 2>/dev/null; then
      return
    fi
    sleep 0.5
  done
  kill -KILL -- "-${process_group}" 2>/dev/null || true
}

run_logged_command() {
  local log_file="$1"
  shift
  local command_pid
  local process_group=""
  local tail_pid

  : > "${log_file}"
  if command -v setsid >/dev/null 2>&1; then
    # Keep each pytest target in its own process group. EngineCore and HCCL
    # workers can outlive pytest after a forced shutdown; without this cleanup
    # they retain NPU memory, port 16666, and the stdout pipe used by tee.
    setsid "$@" > "${log_file}" 2>&1 &
    command_pid=$!
    process_group="${command_pid}"
  else
    echo "Warning: setsid is unavailable; descendant cleanup is disabled"
    "$@" > "${log_file}" 2>&1 &
    command_pid=$!
  fi

  # Redirecting pytest to a regular file prevents leaked descendants from
  # keeping a tee pipeline open forever. Stream that file while pytest runs so
  # the Actions log still has live output.
  tail --pid="${command_pid}" -n +1 -f "${log_file}" &
  tail_pid=$!
  wait "${command_pid}"
  local status=$?

  if [ -n "${process_group}" ]; then
    terminate_process_group "${process_group}"
  fi
  wait "${tail_pid}" || true
  return "${status}"
}

print_test_info() {
  echo -e "\033[1;34m=== TEST INFO ===\033[0m"
  echo -e "  \033[33mDevice:\033[0m ${npu_type}"
  if [ "${npu_type}" != "cpu" ]; then
    echo -e "  \033[33mNPU count:\033[0m ${num_npus}"
  fi
  echo -e "  \033[33mCoverage:\033[0m ${enable_coverage}"
  echo -e "  \033[33mContinue after failure:\033[0m ${continue_on_error}"
  echo -e "  \033[33mTargets:\033[0m"
  for target in "${targets[@]}"; do
    echo -e "    \033[32m-\033[0m ${target}"
  done
  echo -e "\033[1;34m====================\033[0m"
}

print_summary() {
  local result target status log_file failed
  echo -e "\033[1;34m=== TEST SUMMARY ===\033[0m"
  for result in "${test_results[@]}"; do
    IFS='|' read -r target status log_file <<< "${result}"
    echo -e "  ${status}: ${target}"
    echo -e "    log: ${log_file}"
  done
  if [ "${npu_type}" != "cpu" ] && [ "${#failed_logs[@]}" -gt 0 ]; then
    echo -e "\033[1;31m=== FAILED TEST LOGS ===\033[0m"
    for failed in "${failed_logs[@]}"; do
      IFS='|' read -r target log_file <<< "${failed}"
      echo "::group::${target} failure log"
      cat "${log_file}"
      echo "::endgroup::"
    done
  fi
}

pytest_report_all_skipped() {
  python - "$1" <<'PY'
import sys
from xml.etree import ElementTree

try:
    cases = ElementTree.parse(sys.argv[1]).findall(".//testcase")
except (OSError, ElementTree.ParseError):
    sys.exit(1)

skips = [case.find("skipped") for case in cases]
# pytest also uses <skipped> for xfail outcomes in JUnit XML.
all_skipped = skips and all(skip is not None and skip.get("type") != "pytest.xfail" for skip in skips)
sys.exit(0 if all_skipped else 1)
PY
}

run_pytest() {
  local target="$1"
  shift
  local log_name="cpu-ut"
  local coverage_target="cpu-ut"
  if [ "${npu_type}" != "cpu" ]; then
    log_name="${target#tests/}"
    log_name="${log_name%.py}"
    log_name="${log_name//[^a-zA-Z0-9_.-]/_}"
    coverage_target="${target}"
    echo "::group::${target}"
  fi
  test_index=$((test_index + 1))
  local log_file="${pytest_log_dir}/${test_index}-${log_name}.log"
  local report_file
  report_file=$(mktemp "${pytest_log_dir}/pytest-report.XXXXXX")
  local pytest_args=(-sv --color=yes --junitxml="${report_file}" "$@")

  echo -e "\033[1;34m=== Running target: ${target} ===\033[0m"
  local start_time=0
  if [ "${record_timing}" = true ]; then
    start_time=$(date +%s%N)
  fi
  local status=0
  if [ "${enable_coverage}" = "true" ]; then
    setup_coverage "${coverage_target}"
    run_logged_command "${log_file}" python -m coverage run --rcfile="${project_root}/tests/coveragerc" -m pytest "${pytest_args[@]}" || status=$?
  elif [ "${npu_type}" = "cpu" ]; then
    run_logged_command "${log_file}" python -m slipcover --source="${project_root}/vllm_ascend" --omit="*/tests/*,*/ops/triton/*" --json --out "${project_root}/tests/coverage.json" -m pytest -n 8 "${pytest_args[@]}" || status=$?
  else
    run_logged_command "${log_file}" pytest "${pytest_args[@]}" || status=$?
  fi
  local result="FAILED"
  # A module-level skip can leave pytest with zero collected items (exit 5).
  # Require a recorded skip so an empty target still fails.
  if { [ "${status}" -eq 0 ] || [ "${status}" -eq 5 ]; } && pytest_report_all_skipped "${report_file}"; then
    result="SKIPPED"
  elif [ "${status}" -eq 0 ]; then
    result="PASSED"
  fi
  rm -f -- "${report_file}"

  # Tests that failed or were entirely skipped have no usable coverage.
  if [ "${result}" != "PASSED" ] && [ "${enable_coverage}" = "true" ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi
  if [ "${record_timing}" = true ]; then
    local elapsed_ns=$(( $(date +%s%N) - start_time ))
    local elapsed=$(( elapsed_ns / 1000000000 )).$(( (elapsed_ns % 1000000000) / 100000000 ))
    timing_entries+=("{\"name\":\"${target}\",\"passed\":$([ "${result}" = "PASSED" ] && echo true || echo false),\"elapsed\":${elapsed}}")
  fi
  if [ "${npu_type}" != "cpu" ]; then
    echo "::endgroup::"
  fi
  test_results+=("${target}|${result}|${log_file}")
  if [ "${result}" = "FAILED" ]; then
    failed_logs+=("${target}|${log_file}")
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
    if [ "${record_timing}" != true ] && [ "${continue_on_error}" != true ]; then
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
  run_pytest "cpu-ut (${#targets[@]} targets)" "${targets[@]}"
elif [ "${mode}" = "with-device" ]; then
  aclgraph_capture_replay="tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py"
  run_aclgraph_capture_replay=0
  for target in "${targets[@]}"; do
    if [ "${target}" = "${aclgraph_capture_replay}" ]; then
      run_aclgraph_capture_replay=1
      continue
    fi
    run_pytest "${target}" "${target}"
  done
  if [ "${run_aclgraph_capture_replay}" = "1" ]; then
    pip uninstall -y triton-ascend triton
    run_pytest "${aclgraph_capture_replay}" "${aclgraph_capture_replay}"
  fi
else
  for target in "${targets[@]}"; do
    run_pytest "${target}" "${target}"
  done
fi

print_timing_json
print_summary
exit "${overall_status}"
