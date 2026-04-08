#!/bin/bash
set -euo pipefail

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Configuration
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# cann and atb environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh

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
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
}

checkout_src() {
    echo "====> Checkout source code"
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    pip uninstall -y vllm vllm-ascend || true
    rm -rf "$WORKSPACE/vllm" "$WORKSPACE/vllm-ascend"

    echo "Cloning vllm-ascend from $VLLM_ASCEND_REMOTE_URL ref: $VLLM_ASCEND_REF"
    git init "$WORKSPACE/vllm-ascend"
    cd "$WORKSPACE/vllm-ascend"
    git remote add origin "$VLLM_ASCEND_REMOTE_URL"
    if ! git fetch --depth 1 origin "$VLLM_ASCEND_REF"; then
        # VLLM_ASCEND_REF might be a PR head commit hash, look up the PR ref
        PR_REF=$(git ls-remote origin 'refs/pull/*/head' | grep "^${VLLM_ASCEND_REF}" | awk '{print $2}' | head -1)
        if [ -n "$PR_REF" ]; then
            git fetch --depth 1 origin "$PR_REF"
        else
            print_error "Failed to fetch vllm-ascend ref: $VLLM_ASCEND_REF"
        fi
    fi
    git checkout FETCH_HEAD

    if [ ! -d "$WORKSPACE/vllm" ]; then
        echo "Cloning vllm version/ref: $VLLM_VERSION"
        git init "$WORKSPACE/vllm"
        git -C "$WORKSPACE/vllm" fetch --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_VERSION"
        git -C "$WORKSPACE/vllm" checkout FETCH_HEAD
    fi
}

install_vllm() {
    echo "====> Install vllm and vllm-ascend"
    VLLM_TARGET_DEVICE=empty pip install -e "$WORKSPACE/vllm"
    pip install -r "$WORKSPACE/vllm-ascend/requirements-dev.txt"
    pip install -e "$WORKSPACE/vllm-ascend"
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
    pytest -sv --show-capture=no tests/e2e/nightly/multi_node/scripts/test_multi_node.py
    ret=$?
    set -e
    if [ "$LWS_WORKER_INDEX" -eq 0 ]; then
        if [ $ret -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_failure "Some tests failed, please check the error stack above for details. \
If this is insufficient to pinpoint the error, please download and review the logs of all other nodes from the job's summary."
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
        install_vllm
    fi
    show_vllm_info
    show_triton_ascend_info
    cd "$WORKSPACE/vllm-ascend"
    run_tests_with_log
}

main "$@"
