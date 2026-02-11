#!/bin/bash
# 开启 -x 可以打印执行的每一行指令，定位报错行号的神器
set -euxo pipefail 

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# 错误捕获：如果脚本意外退出，打印退出码和行号
trap 'echo -e "${RED}[DEBUG] Script exited with status $? at line $LINENO${NC}"' ERR

echo "[DEBUG] Starting script execution at $(date)"
echo "[DEBUG] Current user: $(whoami)"
echo "[DEBUG] WORKSPACE is set to: ${WORKSPACE:-UNDEFINED}"

# Configuration
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# cann and atb environment setup
set +ue
echo "[INFO] Sourcing ascend-toolkit set_env.sh..."
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "[DEBUG] ascend-toolkit sourced successfully."
else
    echo "[ERROR] ascend-toolkit set_env.sh NOT FOUND!"
fi

# 检查 CANN 路径是否存在
CANN_IR_ENV="/usr/local/Ascend/cann-8.5.0/share/info/ascendnpu-ir/bin/set_env.sh"
echo "[INFO] Checking $CANN_IR_ENV"
if [ -f "$CANN_IR_ENV" ]; then
    source "$CANN_IR_ENV"
else
    echo "[ERROR] CANN IR env file missing: $CANN_IR_ENV"
fi

echo "[INFO] Sourcing atb set_env.sh..."
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
else
    echo "[ERROR] ATB set_env.sh NOT FOUND!"
fi
set -ue

# Home path for aisbench
export BENCHMARK_HOME=${WORKSPACE}/vllm-ascend/benchmark
echo "[DEBUG] BENCHMARK_HOME set to: $BENCHMARK_HOME"

# Logging configurations
export VLLM_LOGGING_LEVEL="INFO"
export GLOG_minloglevel=1
export HF_HUB_OFFLINE="1"

show_vllm_info() {
    echo -e "\n${BLUE}=== SHOW VLLM INFO ===${NC}"
    echo "[DEBUG] Current Directory: $(pwd)"
    pip list | grep vllm || echo "[WARN] No vllm packages found in pip list."

    for repo in "vllm" "vllm-ascend"; do
        echo "Checking $repo git info..."
        if [ -d "$WORKSPACE/$repo/.git" ]; then
            cd "$WORKSPACE/$repo"
            echo "[DEBUG] Git status for $repo:"
            git log -1 --pretty=format:"%h - %an, %ar : %s"
            echo ""
        else
            echo "[WARN] $WORKSPACE/$repo/.git does not exist!"
        fi
    done
}

check_npu_info() {
    echo -e "\n${BLUE}=== NPU INFO ===${NC}"
    npu-smi info || echo "[ERROR] npu-smi failed!"
    INFO_FILE="/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info"
    if [ -f "$INFO_FILE" ]; then
        cat "$INFO_FILE"
    else
        echo "[ERROR] Toolkit info file missing at $INFO_FILE"
    fi
}

upgrade_vllm_ascend_scr() {
    echo -e "\n${BLUE}=== UPGRADE VLLM-ASCEND (PR 5853) ===${NC}"
    TARGET_DIR="$WORKSPACE/vllm-ascend"
    
    if [ ! -d "$TARGET_DIR" ]; then
        echo "[ERROR] Target directory $TARGET_DIR does not exist. Cannot upgrade."
        exit 1
    fi

    cd "$TARGET_DIR"
    echo "[DEBUG] Current Dir: $(pwd)"
    
    echo "[DEBUG] Fetching PR 5853..."
    git fetch origin pull/5853/head:pr-5853 || { echo "[ERROR] Git fetch failed"; exit 1; }
    
    echo "[DEBUG] Checking out pr-5853..."
    git checkout pr-5853 || { echo "[ERROR] Git checkout failed"; exit 1; }
    
    echo "[DEBUG] Final Git Commit Head:"
    git rev-parse HEAD
}

install_extra_components() {
    echo -e "\n${BLUE}=== INSTALL EXTRA COMPONENTS ===${NC}"
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    echo "[DEBUG] Downloading components to temporary dir: $TEMP_DIR"
    
    # 使用 -v 显示下载进度/详细错误
    wget -v https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-sfa-linux.aarch64.run
    chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
    ./CANN-custom_ops-sfa-linux.aarch64.run --quiet
    
    wget -v https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl
    pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
    
    rm -rf "$TEMP_DIR"
}

run_tests_with_log() {
    echo -e "\n${BLUE}=== RUNNING TESTS ===${NC}"
    set +e
    pgrep python3 | xargs -r kill -9
    pgrep VLLM | xargs -r kill -9
    sleep 2
    
    echo "[DEBUG] Starting PyTest..."
    # 确保在正确的目录下运行 pytest
    cd "$WORKSPACE/vllm-ascend"
    pytest -sv --show-capture=no tests/e2e/nightly/multi_node/scripts/test_multi_node.py
    ret=$?
    set -e
    
    if [ $ret -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
    else
        echo -e "${RED}✗ Pytest failed with exit code $ret${NC}"
        exit $ret
    fi
}

main() {
    echo "[DEBUG] Main sequence initiated."
    check_npu_info
    
    # 强制进入工作目录前检查
    if [ -z "${WORKSPACE:-}" ]; then
        echo "[ERROR] WORKSPACE variable is not set!"
        exit 1
    fi

    if [[ "$CONFIG_YAML_PATH" == *"DeepSeek-V3_2-Exp-bf16.yaml" ]]; then
        install_extra_components
    fi

    upgrade_vllm_ascend_scr
    show_vllm_info
    
    run_tests_with_log
}

main "$@"
