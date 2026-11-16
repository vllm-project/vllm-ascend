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

run_prefixed_logged_command() {
  local log_file="$1"
  local line_prefix="$2"
  shift 2
  local command_pid
  local process_group=""
  local tail_pid

  : > "${log_file}"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" > "${log_file}" 2>&1 &
    command_pid=$!
    process_group="${command_pid}"
  else
    echo "Warning: setsid is unavailable; descendant cleanup is disabled"
    "$@" > "${log_file}" 2>&1 &
    command_pid=$!
  fi

  # Prefix each line so the two card logs stay readable when they stream together.
  tail --pid="${command_pid}" -n +1 -f "${log_file}" \
    | awk -v prefix="${line_prefix}" '{ print prefix $0; fflush() }' &
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
  local start_time=0
  if [ "${record_timing}" = true ]; then
    start_time=$(date +%s%N)
  fi
  if [ "${enable_coverage}" = "true" ]; then
    setup_coverage "${target}"
    set +e
    run_logged_command "${log_file}" python -m coverage run --rcfile="${project_root}/tests/coveragerc" -m pytest -sv --color=yes "${target}"
  else
    set +e
    if [ "${npu_type}" = "cpu" ]; then
      run_logged_command "${log_file}" python -m slipcover --source="${project_root}/vllm_ascend" --omit="*/tests/*,*/ops/triton/*" --json --out "${project_root}/tests/coverage.json" -m pytest -n 8 -sv --color=yes "${target}"
    else
      run_logged_command "${log_file}" pytest -sv --color=yes "${target}"
    fi
  fi
  local status=$?
  set -e
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
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
    if [ "${record_timing}" != true ] && [ "${continue_on_error}" != true ]; then
      print_summary
      exit "${status}"
    fi
  fi
}

run_pytest_batch() {
  local target="$1"
  local coverage_key="$2"
  shift 2
  local batch_targets=("$@")
  test_index=$((test_index + 1))
  local log_name="${target}"
  log_name="${log_name//[^a-zA-Z0-9_.-]/_}"
  local log_file="${pytest_log_dir}/${test_index}-${log_name}.log"
  echo "::group::${target}"

  echo -e "\033[1;34m=== Running target: ${target} ===\033[0m"
  local start_time=0
  if [ "${record_timing}" = true ]; then
    start_time=$(date +%s%N)
  fi
  if [ "${enable_coverage}" = "true" ]; then
    echo "DEBUG: Go to the [Coverage Branch] page."
    setup_coverage "${coverage_key}"
    set +e
    run_logged_command "${log_file}" python -m coverage run --rcfile="${project_root}/tests/coveragerc" -m pytest -sv --color=yes "${batch_targets[@]}"
  else
    set +e
    if [ "${npu_type}" = "cpu" ]; then
      run_logged_command "${log_file}" python -m slipcover --source="${project_root}/vllm_ascend" --omit="*/tests/*,*/ops/triton/*" --json --out "${project_root}/tests/coverage.json" -m pytest -n 8 -sv --color=yes "${batch_targets[@]}"
    else
      run_logged_command "${log_file}" pytest -sv --color=yes "${batch_targets[@]}"
    fi
  fi
  local status=$?
  set -e
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

# Operator cases are single-card tests. On a multi-card runner, split the files
# across the visible NPUs and run one pytest process per card.
ops_test_prefix="tests/e2e/pull_request/two_card/ops/"

run_ops_device_worker() {
  local device="$1"
  local log_file="$2"
  local status_file="$3"
  local elapsed_file="$4"
  local cache_root="$5"
  local ext_dir="$6"
  shift 6
  local batch_targets=("$@")
  local line_prefix="npu${device}: "
  local start_time=0
  if [ "${record_timing}" = true ]; then
    start_time=$(date +%s%N)
  fi

  # set -e is disabled for this worker. The status file is the result the parent reads.
  set +e
  if [ "${enable_coverage}" = "true" ]; then
    ASCEND_RT_VISIBLE_DEVICES="${device}" \
      VLLM_CACHE_ROOT="${cache_root}" \
      TORCH_EXTENSIONS_DIR="${ext_dir}" \
      MASTER_PORT="$((29500 + device))" \
      run_prefixed_logged_command "${log_file}" "${line_prefix}" \
        python -m coverage run --rcfile="${project_root}/tests/coveragerc" -m pytest -sv --color=yes "${batch_targets[@]}"
  else
    ASCEND_RT_VISIBLE_DEVICES="${device}" \
      VLLM_CACHE_ROOT="${cache_root}" \
      TORCH_EXTENSIONS_DIR="${ext_dir}" \
      MASTER_PORT="$((29500 + device))" \
      run_prefixed_logged_command "${log_file}" "${line_prefix}" \
        pytest -sv --color=yes "${batch_targets[@]}"
  fi
  local status=$?
  echo "${status}" > "${status_file}"
  if [ "${record_timing}" = true ]; then
    local elapsed_ns=$(( $(date +%s%N) - start_time ))
    local elapsed=$(( elapsed_ns / 1000000000 )).$(( (elapsed_ns % 1000000000) / 100000000 ))
    echo "${elapsed}" > "${elapsed_file}"
  fi
  set -e
  return 0
}

run_ops_across_devices() {
  local ops_targets=("$@")
  local device_count="${num_npus}"
  local list_dir
  list_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/ops-split.XXXXXX")"
  local index=0
  local target
  local device
  for target in "${ops_targets[@]}"; do
    device=$((index % device_count))
    printf '%s\n' "${target}" >> "${list_dir}/npu${device}.txt"
    index=$((index + 1))
  done

  echo -e "\033[1;34m=== Splitting ${#ops_targets[@]} operator files across ${device_count} NPUs ===\033[0m"
  for ((device=0; device<device_count; device++)); do
    local list_file="${list_dir}/npu${device}.txt"
    if [ ! -s "${list_file}" ]; then
      continue
    fi
    echo "  npu${device}:"
    while IFS= read -r target; do
      echo "    - ${target}"
    done < "${list_file}"
  done

  if [ "${enable_coverage}" = "true" ]; then
    setup_coverage "tests/e2e/pull_request/two_card/ops"
  fi

  local worker_pids=()
  local status_files=()
  local elapsed_files=()
  local log_files=()
  local labels=()
  local device_targets=()
  for ((device=0; device<device_count; device++)); do
    local list_file="${list_dir}/npu${device}.txt"
    if [ ! -s "${list_file}" ]; then
      continue
    fi
    device_targets=()
    mapfile -t device_targets < "${list_file}"
    test_index=$((test_index + 1))
    local log_file="${pytest_log_dir}/${test_index}-ops-npu${device}.log"
    local status_file="${pytest_log_dir}/${test_index}-ops-npu${device}.status"
    local elapsed_file="${pytest_log_dir}/${test_index}-ops-npu${device}.elapsed"
    local cache_root
    local ext_dir
    cache_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/vllm-cache-${npu_type}-npu${device}.XXXXXX")"
    ext_dir="$(mktemp -d "${RUNNER_TEMP:-/tmp}/torch-extensions-${npu_type}-npu${device}.XXXXXX")"
    local label="ops npu${device} (${#device_targets[@]} targets)"
    echo "npu${device}: cache ${cache_root}"
    run_ops_device_worker "${device}" "${log_file}" "${status_file}" "${elapsed_file}" \
      "${cache_root}" "${ext_dir}" "${device_targets[@]}" &
    worker_pids+=("$!")
    status_files+=("${status_file}")
    elapsed_files+=("${elapsed_file}")
    log_files+=("${log_file}")
    labels+=("${label}")
  done

  local pid
  for pid in "${worker_pids[@]}"; do
    wait "${pid}" || true
  done

  local ops_failed=0
  local i
  for i in "${!labels[@]}"; do
    local status="1"
    if [ -f "${status_files[$i]}" ]; then
      status="$(<"${status_files[$i]}")"
    fi
    if [ "${status}" = "0" ]; then
      test_results+=("${labels[$i]}|PASSED|${log_files[$i]}")
    else
      ops_failed=1
      test_results+=("${labels[$i]}|FAILED|${log_files[$i]}")
      failed_logs+=("${labels[$i]}|${log_files[$i]}")
      if [ "${overall_status}" -eq 0 ]; then
        overall_status="${status}"
      fi
    fi
    if [ "${record_timing}" = true ] && [ -f "${elapsed_files[$i]}" ]; then
      local elapsed
      elapsed="$(<"${elapsed_files[$i]}")"
      local passed=false
      if [ "${status}" = "0" ]; then
        passed=true
      fi
      timing_entries+=("{\"name\":\"${labels[$i]}\",\"passed\":${passed},\"elapsed\":${elapsed}}")
    fi
  done

  if [ "${enable_coverage}" = "true" ] && [ "${ops_failed}" -ne 0 ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi
  if [ "${ops_failed}" -ne 0 ] && [ "${record_timing}" != true ] && [ "${continue_on_error}" != true ]; then
    print_summary
    exit "${overall_status}"
  fi
}

run_packed_ops_and_regular() {
  local ops_targets=()
  local regular_targets=()
  local target
  for target in "$@"; do
    case "${target}" in
      "${ops_test_prefix}"*)
        ops_targets+=("${target}")
        ;;
      *)
        regular_targets+=("${target}")
        ;;
    esac
  done
  if [ "${#ops_targets[@]}" -gt 0 ]; then
    if [ "${mode}" = "with-device" ] && [ "${num_npus}" -ge 2 ]; then
      run_ops_across_devices "${ops_targets[@]}"
    else
      run_pytest_batch "ops (${#ops_targets[@]} targets)" "tests/e2e/pull_request/two_card/ops" "${ops_targets[@]}"
    fi
  fi
  if [ "${#regular_targets[@]}" -gt 0 ]; then
    for target in "${regular_targets[@]}"; do
      run_pytest_target "${target}"
    done
  fi
}

if [ "${npu_type}" = "cpu" ]; then
  run_pytest_batch "cpu-ut (${#targets[@]} targets)" "cpu-ut" "${targets[@]}"
elif [ "${mode}" = "with-device" ]; then
  aclgraph_capture_replay="tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py"
  run_aclgraph_capture_replay=0
  device_targets=()
  for target in "${targets[@]}"; do
    if [ "${target}" = "${aclgraph_capture_replay}" ]; then
      run_aclgraph_capture_replay=1
      continue
    fi
    device_targets+=("${target}")
  done
  if [ "${#device_targets[@]}" -gt 0 ]; then
    run_packed_ops_and_regular "${device_targets[@]}"
  fi
  if [ "${run_aclgraph_capture_replay}" = "1" ]; then
    pip uninstall -y triton-ascend triton
    run_pytest_target "${aclgraph_capture_replay}"
  fi
else
  run_packed_ops_and_regular "${targets[@]}"
fi

print_timing_json
print_summary
exit "${overall_status}"
