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
# Home path for aisbench
export BENCHMARK_HOME=${WORKSPACE}/vllm-ascend/benchmark

# Logging configurations
export VLLM_LOGGING_LEVEL="INFO"
# Reduce glog verbosity for mooncake
export GLOG_minloglevel=1
# Set transformers to offline mode to avoid downloading models during tests
export TRANSFORMERS_OFFLINE="1"

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
    # Fix me(Potabk): Currently, there have some issues with accessing GitHub via https://gh-proxy.test.osinfra.cn in certain regions.
    # We should switch to a more stable proxy for now until the network proxy is stable enough.
    git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
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

install_triton_ascend() {
    echo "====> Installing triton_ascend"

    BISHENG_NAME="Ascend-BiSheng-toolkit_aarch64_20260105.run"
    BISHENG_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${BISHENG_NAME}"

    if ! wget -q -O "${BISHENG_NAME}" "${BISHENG_URL}"; then
        echo "Failed to download ${BISHENG_NAME}"
        return 1
    fi
    chmod +x "${BISHENG_NAME}"

    if ! "./${BISHENG_NAME}" --install; then
        rm -f "${BISHENG_NAME}"
        echo "Failed to install ${BISHENG_NAME}"
        return 1
    fi
    rm -f "${BISHENG_NAME}"

    export PATH=/usr/local/Ascend/tools/bishengir/bin:$PATH
    which bishengir-compile
    python3 -m pip install -i https://test.pypi.org/simple/ triton-ascend==3.2.0.dev20260105
    echo "====> Triton ascend installation completed"
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
            print_failure "Some tests failed, please check the error stack above for details.\
            If this is insufficient to pinpoint the error, please download and review the logs of all other nodes from the job's summary."
        fi
    fi
}

main() {
    check_npu_info
    check_and_config
    show_vllm_info
    install_triton_ascend
    if [[ "$CONFIG_YAML_PATH" == *"DeepSeek-V3_2-Exp-bf16.yaml" ]]; then
        install_extra_components
    fi
    cd "$WORKSPACE/vllm-ascend"
    run_tests_with_log
}

main "$@"
