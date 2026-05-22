#!/bin/bash
set -euo pipefail

export WORKSPACE=${WORKSPACE:-/vllm-workspace}
export IS_PR_TEST=${IS_PR_TEST:-false}

GREEN="\033[0;32m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m"

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh

set +eu
source /usr/local/Ascend/nnal/atb/set_env.sh
set -eu

export BENCHMARK_HOME=${WORKSPACE}/vllm-ascend/benchmark
export VLLM_LOGGING_LEVEL="INFO"
export GLOG_minloglevel=1
export HF_HUB_OFFLINE="1"
export VLLM_ENGINE_READY_TIMEOUT_S=1800

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_failure() {
    echo -e "${RED}${FAIL_TAG:-test_failed} ERROR: $1${NC}"
    exit 1
}

print_success() {
    echo -e "${GREEN}$1${NC}"
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

checkout_src() {
    echo "====> Checkout source code"
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    pip uninstall -y vllm-ascend || true
    cp -r "$WORKSPACE/vllm-ascend/benchmark" /tmp/aisbench-backup || true
    rm -rf "$WORKSPACE/vllm-ascend"

    echo "Cloning vllm-ascend from $VLLM_ASCEND_REMOTE_URL"
    git clone --depth 1 "$VLLM_ASCEND_REMOTE_URL" "$WORKSPACE/vllm-ascend"
    cd "$WORKSPACE/vllm-ascend"
    PR_REF=$(git ls-remote origin 'refs/pull/*/head' | grep "^${VLLM_ASCEND_REF}" | awk '{print $2}' | head -1)
    if [ -n "$PR_REF" ]; then
        git fetch --depth 1 origin "$PR_REF"
        git checkout FETCH_HEAD
    else
        git fetch origin '+refs/pull/*/head:refs/remotes/pull/*' 2>/dev/null || true
        git checkout "$VLLM_ASCEND_REF"
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

show_vllm_info() {
    cd "$WORKSPACE"
    echo "Installed vLLM-related Python packages:"
    pip list | grep vllm || echo "No vllm packages found."
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
    pytest -sv --show-capture=no tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py
    ret=$?
    set -e
    if [ "$LWS_WORKER_INDEX" -eq 0 ]; then
        if [ $ret -eq 0 ]; then
            print_success "All external DP tests passed!"
        else
            print_failure "External DP tests failed. Please download logs from all nodes for details."
        fi
    fi
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
    cd "$WORKSPACE/vllm-ascend"
    run_tests_with_log
}

main "$@"
