#!/bin/bash
set -euo pipefail

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

INTERNAL_DP_TEST_PATH="tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py"
EXTERNAL_DP_TEST_PATH="tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py"

if [ -z "${MULTI_NODE_TEST_PATH:-}" ]; then
    if [[ "${CONFIG_BASE_PATH:-}" == *"external_dp/config"* || "${CONFIG_YAML_PATH:-}" == *"external_dp/config"* ]]; then
        MULTI_NODE_TEST_PATH="$EXTERNAL_DP_TEST_PATH"
    else
        MULTI_NODE_TEST_PATH="$INTERNAL_DP_TEST_PATH"
    fi
fi

# Configuration
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# cann and atb environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh

set +eu
source /usr/local/Ascend/nnal/atb/set_env.sh
set -eu

# Home path for aisbench
export BENCHMARK_HOME=${WORKSPACE}/vllm-ascend/benchmark

# Logging configurations
export VLLM_LOGGING_LEVEL="INFO"
# Reduce glog verbosity for mooncake
export GLOG_minloglevel=1
# Set transformers to offline mode to avoid downloading models during tests
export HF_HUB_OFFLINE="1"
# Default is 600s
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_failure() {
    echo -e "${RED}${FAIL_TAG:-test_failed} ✗ ERROR: $1${NC}"
    exit 1
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages and exit
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

show_vllm_info() {
    cd "$WORKSPACE"
    echo "Installed vLLM-related Python packages:"
    pip list | grep vllm || echo "No vllm packages found."

    echo ""
    echo "============================"
    echo "vLLM Git information"
    echo "============================"
    cd vllm
    if [ -d .git ]; then
    echo "Branch:      $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit hash: $(git rev-parse HEAD)"
    echo "Author:      $(git log -1 --pretty=format:'%an <%ae>')"
    echo "Date:        $(git log -1 --pretty=format:'%ad' --date=iso)"
    echo "Message:     $(git log -1 --pretty=format:'%s')"
    echo "Tags:        $(git tag --points-at HEAD || echo 'None')"
    echo "Remote:      $(git remote -v | head -n1)"
    echo ""
    else
    echo "No .git directory found in vllm"
    fi
    cd ..

    echo ""
    echo "============================"
    echo "vLLM-Ascend Git information"
    echo "============================"
    cd vllm-ascend
    if [ -d .git ]; then
    echo "Branch:      $(git rev-parse --abbrev-ref HEAD)"
    echo "Commit hash: $(git rev-parse HEAD)"
    echo "Author:      $(git log -1 --pretty=format:'%an <%ae>')"
    echo "Date:        $(git log -1 --pretty=format:'%ad' --date=iso)"
    echo "Message:     $(git log -1 --pretty=format:'%s')"
    echo "Tags:        $(git tag --points-at HEAD || echo 'None')"
    echo "Remote:      $(git remote -v | head -n1)"
    echo ""
    else
    echo "No .git directory found in vllm-ascend"
    fi
    cd ..
}

check_npu_info() {
    echo "====> Check NPU info"
    npu-smi info
    cat "/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info"
}

check_and_config() {
    echo "====> Configure mirrors and git proxy"
    git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi"
}

install_extra_components() {
    echo "====> Installing extra components for DeepSeek-v3.2-exp-bf16"

    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-sfa-linux.aarch64.run; then
        echo "Failed to download CANN-custom_ops-sfa-linux.aarch64.run"
        return 1
    fi
    chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
    ./CANN-custom_ops-sfa-linux.aarch64.run --quiet

    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl; then
        echo "Failed to download custom_ops wheel"
        return 1
    fi
    pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl

    export ASCEND_CUSTOM_OPP_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    rm -f CANN-custom_ops-sfa-linux.aarch64.run \
          custom_ops-1.0-cp311-cp311-linux_aarch64.whl
    echo "====> Extra components installation completed"
}

checkout_src() {
    echo "====> Checkout source code"
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    pip uninstall -y vllm-ascend || true
    cp -r "$WORKSPACE/vllm-ascend/benchmark" /tmp/aisbench-backup || true
    rm -rf "$WORKSPACE/vllm-ascend"

    if [ ! -d "$WORKSPACE/vllm-ascend" ]; then
        echo "Cloning vllm-ascend from $VLLM_ASCEND_REMOTE_URL"
        git clone --depth 1 --recurse-submodules "$VLLM_ASCEND_REMOTE_URL" "$WORKSPACE/vllm-ascend"
        cd "$WORKSPACE/vllm-ascend"
        PR_REF=$(git ls-remote origin 'refs/pull/*/head' | grep "^${VLLM_ASCEND_REF}" | awk '{print $2}' | head -1)
        if [ -n "$PR_REF" ]; then
            git fetch --depth 1 origin "$PR_REF"
            git checkout FETCH_HEAD
        else
            git fetch origin '+refs/pull/*/head:refs/remotes/pull/*' 2>/dev/null || true
            git checkout "$VLLM_ASCEND_REF"
        fi
        git submodule update --init --recursive
    fi
}

install_vllm_ascend() {
    echo "====> Install vllm-ascend"
    pip install -r "$WORKSPACE/vllm-ascend/requirements-dev.txt"
    pip install -e "$WORKSPACE/vllm-ascend"
}

install_aisbench() {
    echo "====> Install AISBench benchmark"

    BENCH_DIR="$WORKSPACE/vllm-ascend/benchmark"

    cp -r /tmp/aisbench-backup "$BENCH_DIR"

    cd "$BENCH_DIR"
    pip install -e . \
        -r requirements/api.txt \
        -r requirements/extra.txt

    python3 -m pip cache purge || echo "WARNING: pip cache purge failed, but proceeding..."

}

show_triton_ascend_info() {
    echo "====> Check triton ascend info"
    clang -v
    which bishengir-compile
    pip show triton-ascend
}

kill_npu_processes() {
  pgrep python3 | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  sleep 4
}

run_tests_with_log() {
    set +e
    kill_npu_processes
    mkdir -p "${LOG_PREFIX}"
    echo "====> Run pytest entry: $MULTI_NODE_TEST_PATH"
    local log_file="${LOG_PREFIX}/node_${LWS_WORKER_INDEX:-?}_pytest.log"
    pytest -sv --show-capture=no "$MULTI_NODE_TEST_PATH" 2>&1 | tee "$log_file"
    ret=$?
    echo "pytest exit code: ret=${ret}"
    set -e
    if [ "${LWS_WORKER_INDEX:-}" = "0" ]; then
        if [ $ret -eq 0 ]; then
            print_success "All tests passed!"
            touch "${LOG_PREFIX}/aop_done" 2>/dev/null
        else
            echo "Leader: waiting 10s for worker logs..."
            sleep 10
            if [ "${AOP_MULTI_ENABLED:-}" = "true" ]; then
                set +e; aop_pipeline; set -e
            fi
            local done_file="${LOG_PREFIX}/aop_done"
            touch "$done_file"
            echo "Leader: notifying workers (${done_file})"
            echo -e "${RED}${FAIL_TAG:-test_failed} ✗ ERROR: Some tests failed${NC}"
            exit 1
        fi
    elif [ "${AOP_MULTI_ENABLED:-}" = "true" ]; then
        if [ $ret -eq 0 ]; then
            echo "Worker: test passed, waiting for leader..."
            local wait_timeout=30
            while [ $wait_timeout -gt 0 ] && [ ! -f "${LOG_PREFIX}/aop_done" ]; do
                sleep 1
                wait_timeout=$((wait_timeout - 1))
            done
        fi
        if [ ! -f "${LOG_PREFIX}/aop_done" ]; then
            local coord="${COORD_DIR:-/root/.cache/nightly_bisect/coord}"
            local release="${LOG_PREFIX}/aop_done"
            mkdir -p "$coord"
            touch "${coord}/worker_ready_${LWS_WORKER_INDEX}"
            echo "Worker: signalling ready at ${coord}/worker_ready_${LWS_WORKER_INDEX}"
            echo "Worker: joining bisect as worker node (index ${LWS_WORKER_INDEX})..."
            cd "$WORKSPACE/vllm-ascend"
            python -m tests.e2e.nightly.bisect.auto_bisect \
                --scene multi_node \
                --config-yaml "${CONFIG_YAML_PATH}" \
                --bad-commit HEAD \
                --coord-dir "${coord}" \
                --release-file "${release}"
            while [ ! -f "$release" ]; do sleep 5; done
            echo "Worker: release signal received, exiting"
            exit 1
        else
            echo "Worker: leader finished successfully, exiting"
        fi
    fi
}

# Run AOP decision pipeline on failure: classify → check age → bisect-or-exit
# Same logic as _e2e_nightly_multi_node.yaml AOP hooks.
aop_pipeline() {
    local rules="$WORKSPACE/vllm-ascend/tests/e2e/nightly/scripts/rules-env.txt"
    local table="${GOOD_TABLE:-}"
    # Strip branch prefix from BENCHMARK_JOB_NAME (e.g. "main-Qwen3.5-27B-w8a8-A2" → "Qwen3.5-27B-w8a8-A2")
    local case_name="${BENCHMARK_JOB_NAME#*-}"
    if [ -z "$case_name" ] || [ "$case_name" = "$BENCHMARK_JOB_NAME" ]; then
        case_name="${CONFIG_YAML_PATH%.yaml}"
    fi

    echo "============================================"
    echo "  AOP Pipeline (Pod) - START"
    echo "  Config      : ${CONFIG_YAML_PATH}"
    echo "  Case name   : ${case_name}"
    echo "  Rules file  : ${rules}"
    echo "  Table file  : ${table}"
    echo "  Log prefix  : ${LOG_PREFIX}"
    echo "  BENCHMARK_JOB_NAME: ${BENCHMARK_JOB_NAME:-}"
    echo "============================================"

    # ---- Step 1: Classify ----
    echo ""
    echo "--- [1/3] Classify: scanning pod logs for env patterns ---"
    echo "  Rules content:"
    if [ -f "$rules" ]; then
        grep -vE '^[[:space:]]*(#|$)' "$rules" | sed 's/^/    > /'
    else
        echo "    (rules file not found)"
    fi

    echo ""
    echo "  Pod logs found:"
    local found_any=0
    for f in "${LOG_PREFIX}/node_"*"_pytest.log"; do
        if [ -f "$f" ]; then
            echo "    - ${f} ($(wc -l < "$f") lines)"
            found_any=1
        fi
    done
    [ "$found_any" -eq 0 ] && echo "    (no pod logs found)"

    local env_count=0
    if [ -f "$rules" ]; then
        for f in "${LOG_PREFIX}/node_"*"_pytest.log"; do
            if [ -f "$f" ]; then
                local n
                n=$(grep -vE '^[[:space:]]*(#|$)' "$rules" | grep -ciEf - "$f" 2>/dev/null || echo 0)
                n=${n%%[!0-9]*}
                echo "    Scan ${f}: ${n} matches"
                env_count=$((env_count + n))
                if [ "$n" -gt 0 ]; then
                    echo "    Matched lines:"
                    grep -vE '^[[:space:]]*(#|$)' "$rules" | grep -niEf - "$f" | head -5 | sed 's/^/      /'
                fi
            fi
        done
    fi
    echo "  Classify result: env_count=${env_count}"

    if [ "$found_any" -eq 0 ]; then
        echo "  Decision: no pod logs → SKIP"
        echo "=== AOP Pipeline (Pod) - END (no logs) ==="
        return 1
    fi

    if [ "$env_count" -gt 0 ]; then
        echo "  Decision: env_failure → SKIP"
        echo "=== AOP Pipeline (Pod) - END (env skip) ==="
        return 1
    fi

    # ---- Step 2: Check age ----
    echo ""
    echo "--- [2/3] Check commit age ---"
    echo "  Looking up: ${case_name}"
    local skip_age=0
    if [ ! -f "$table" ]; then
        echo "  Table file not found: ${table}"
        echo "  Decision: no table → SKIP"
        echo "=== AOP Pipeline (Pod) - END (age skip) ==="
        return 1
    fi

    # Only consider success rows
    local success_rows
    success_rows=$(grep "^${case_name}," "$table" | grep -F ',success,' || true)
    if [ -z "$success_rows" ]; then
        echo "  No success row found for '${case_name}'"
        echo "  Decision: no success entry → SKIP"
        echo "=== AOP Pipeline (Pod) - END (age skip) ==="
        return 1
    fi

    # Pick most recent success row
    local best_date=""
    while IFS= read -r row; do
        local d
        d=$(echo "$row" | awk -F',' '{print $NF}' | xargs)
        [ -z "$d" ] && continue
        if [ -z "$best_date" ] || [[ "$d" > "$best_date" ]]; then
            best_date="$d"
        fi
    done <<< "$success_rows"

    if [ -z "$best_date" ]; then
        echo "  No valid date in success rows"
        echo "  Decision: no date → SKIP"
        echo "=== AOP Pipeline (Pod) - END (age skip) ==="
        return 1
    fi

    echo "  Matched row: $(grep -m1 "$best_date" <<< "$success_rows")"
    local last_ts now_ts age_days
    last_ts=$(date -d "$best_date" +%s 2>/dev/null || echo 0)
    if [ "$last_ts" = "0" ] || [ -z "$last_ts" ]; then
        echo "  Date parse failed: ${best_date}"
        echo "  Decision: invalid date → SKIP"
        echo "=== AOP Pipeline (Pod) - END (age skip) ==="
        return 1
    fi
    now_ts=$(date +%s)
    age_days=$(( (now_ts - last_ts) / 86400 ))
    echo "  Last success: ${best_date} (${age_days} days ago, threshold: 3 days)"

    if [ "$age_days" -gt 3 ]; then
        echo "  Decision: old commit (> 3 days) → SKIP"
        echo "=== AOP Pipeline (Pod) - END (age skip) ==="
        return 1
    fi

    # ---- Step 3: Bisect ----
    echo ""
    echo "--- [3/3] Run bisect ---"
    echo "  Scene       : multi_node"
    echo "  Config      : ${CONFIG_YAML_PATH}"
    echo "  Bad commit  : HEAD"
    echo "  Name        : ${case_name}"
    local coord="${COORD_DIR:-/root/.cache/nightly_bisect/coord}"
    echo "  Coord dir   : ${coord}"

    # Wait for all workers to signal ready
    echo "  Waiting for workers..."
    for i in $(seq 1 30); do
        local ready_count=0
        for f in "${coord}"/worker_ready_*; do
            [ -e "$f" ] && ready_count=$((ready_count + 1))
        done
        echo "    [${i}/30] ready workers: ${ready_count}"
        if [ "$ready_count" -ge 1 ]; then break; fi
        sleep 2
    done

    cd "$WORKSPACE/vllm-ascend"
    local bisect_rc=0
    python -m tests.e2e.nightly.bisect.auto_bisect \
        --scene multi_node \
        --config-yaml "${CONFIG_YAML_PATH}" \
        --bad-commit HEAD \
        --good-table "${table}" \
        --name "${case_name}" \
        --coord-dir "${coord}" || bisect_rc=$?
    echo "  bisect completed (exit code: ${bisect_rc})"
    echo "=== AOP Pipeline (Pod) - END ==="
    return 1
}

clear_logs() {
    print_section "Clearing logs from previous runs"
    rm -fr "$HOME/ascend/log" || true
}

backup_ascend_logs() {
    if [ -n "${LOG_PREFIX:-}" ]; then
        local dest="${LOG_PREFIX}/node_${LWS_WORKER_INDEX:-unknown}_plogs"
        mkdir -p "$dest"
        cp -r /root/ascend/log/. "$dest/" 2>/dev/null || true
        echo "Ascend logs backed up to $dest"
    fi
}

main() {
    trap backup_ascend_logs EXIT
    check_npu_info
    clear_logs
    check_and_config
    if [[ "$IS_PR_TEST" == "true" ]]; then
        checkout_src
        install_vllm_ascend
        install_aisbench
    fi
    show_vllm_info
    show_triton_ascend_info
    if [[ "$CONFIG_YAML_PATH" == *"DeepSeek-V3_2-Exp-bf16.yaml" ]]; then
        install_extra_components
    fi
    cd "$WORKSPACE/vllm-ascend"
    run_tests_with_log
}

main "$@"
