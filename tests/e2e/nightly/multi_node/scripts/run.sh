#!/bin/bash
set -euo pipefail

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Configuration
LOG_DIR="/root/.cache/tests/logs"
OVERWRITE_LOGS=true
SRC_DIR="$WORKSPACE/source_code"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_failure() {
    echo -e "${RED}${FAIL_TAG} ✗ ERROR: $1${NC}"
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

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        print_error "$1"
    fi
}

if [ $(id -u) -ne 0 ]; then
	print_error "Require root permission, try sudo ./dependencies.sh"
fi


check_npu_info() {
    echo "====> Check NPU info"
    npu-smi info
    cat "/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/ascend_toolkit_install.info"
}

check_and_config() {
    echo "====> Configure mirrors and git proxy"
    git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/"
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
}

checkout_src() {
    echo "====> Checkout source code"
    mkdir -p "$SRC_DIR"

    # vllm-ascend
    if [ ! -d "$SRC_DIR/vllm-ascend" ]; then
        git clone --depth 1 -b $VLLM_ASCEND_VERSION $VLLM_ASCEND_REMOTE_URL "$SRC_DIR/vllm-ascend"
    fi

    # vllm
    if [ ! -d "$SRC_DIR/vllm" ]; then
        git clone -b $VLLM_VERSION https://github.com/vllm-project/vllm.git "$SRC_DIR/vllm"
    fi
}

install_sys_dependencies() {
    echo "====> Install system dependencies"
    apt-get update -y

    DEP_LIST=()
    while IFS= read -r line; do
        [[ -n "$line" && ! "$line" =~ ^# ]] && DEP_LIST+=("$line")
    done < "$SRC_DIR/vllm-ascend/packages.txt"

    apt-get install -y "${DEP_LIST[@]}" gcc g++ cmake libnuma-dev iproute2
}

install_vllm() {
    echo "====> Install vllm and vllm-ascend"
    VLLM_TARGET_DEVICE=empty pip install -e "$SRC_DIR/vllm"
    pip install -e "$SRC_DIR/vllm-ascend"
    pip install modelscope
    # Install for pytest
    pip install -r "$SRC_DIR/vllm-ascend/requirements-dev.txt"
}

install_ais_bench() {
    local AIS_BENCH="$SRC_DIR/vllm-ascend/benchmark"
    git clone https://gitee.com/aisbench/benchmark.git $AIS_BENCH
    cd $AIS_BENCH
    git checkout v3.0-20250930-master
    pip3 install -e ./
    pip3 install -r requirements/api.txt
    pip3 install -r requirements/extra.txt
    cd -
}

install_go() {
    # Check if Go is already installed
    if command -v go &> /dev/null; then
        GO_VERSION=$(go version | awk '{print $3}')
        if [[ "$GO_VERSION" == "go$GOVER" ]]; then
            echo -e "${YELLOW}Go $GOVER is already installed. Skipping...${NC}"
        else
            echo -e "${YELLOW}Found Go $GO_VERSION. Will install Go $GOVER...${NC}"
            download_go
        fi
    else
        download_go
    fi

    # Add Go to PATH if not already there
    if ! grep -q "export PATH=\$PATH:/usr/local/go/bin" ~/.bashrc; then
        echo -e "${YELLOW}Adding Go to your PATH in ~/.bashrc${NC}"
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        echo -e "${YELLOW}Please run 'source ~/.bashrc' or start a new terminal to use Go${NC}"
    fi
    export PATH=$PATH:/usr/local/go/bin
}

install_extra_components() {
    echo "====> Installing extra components for DeepSeek-R1-W8A8"
    
    mkdir -p /vllm-workspace/CANN
    
    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-sfa-linux.aarch64.run; then
        echo "Failed to download CANN-custom_ops-sfa-linux.aarch64.run"
        return 1
    fi
    chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
    ./CANN-custom_ops-sfa-linux.aarch64.run --quiet
    
    export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
    
    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl; then
        echo "Failed to download custom_ops wheel"
        return 1
    fi
    pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
    
    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-mlapo-linux.aarch64.run; then
        echo "Failed to download CANN-custom_ops-mlapo-linux.aarch64.run"
        return 1
    fi
    chmod +x ./CANN-custom_ops-mlapo-linux.aarch64.run 
    ./CANN-custom_ops-mlapo-linux.aarch64.run --quiet --install-path=/vllm-workspace/CANN
    
    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/torch_npu-2.7.1%2Bgitb7c90d0-cp311-cp311-linux_aarch64.whl; then
        echo "Failed to download torch_npu wheel"
        return 1
    fi
    pip install torch_npu-2.7.1+gitb7c90d0-cp311-cp311-linux_aarch64.whl
    
    if ! wget -q https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/libopsproto_rt2.0.so; then
        echo "Failed to download libopsproto_rt2.0.so"
        return 1
    fi
    cp libopsproto_rt2.0.so /usr/local/Ascend/ascend-toolkit/8.2.RC1/opp/built-in/op_proto/lib/linux/aarch64/libopsproto_rt2.0.so
    
    source /vllm-workspace/CANN/vendors/customize/bin/set_env.bash
    export LD_PRELOAD=/vllm-workspace/CANN/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so:${LD_PRELOAD}
    
    cat >> ~/.bashrc << 'EOF'

# Extra components for DeepSeek-R1-W8A8
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
source /vllm-workspace/CANN/vendors/customize/bin/set_env.bash
export LD_PRELOAD=/vllm-workspace/CANN/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so:${LD_PRELOAD}
EOF
    
    rm -f CANN-custom_ops-sfa-linux.aarch64.run \
          custom_ops-1.0-cp311-cp311-linux_aarch64.whl \
          CANN-custom_ops-mlapo-linux.aarch64.run \
          torch_npu-2.7.1+gitb7c90d0-cp311-cp311-linux_aarch64.whl \
          libopsproto_rt2.0.so
    
    echo "====> Extra components installation completed"
}

kill_npu_processes() {
  pgrep python3 | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  sleep 4
}

run_tests_with_log() {
    set +e
    kill_npu_processes
    BASENAME=$(basename "$CONFIG_YAML_PATH" .yaml)
    # each worker should have log file
    LOG_FILE="${RESULT_FILE_PATH}/${BASENAME}_worker_${LWS_WORKER_INDEX}.log"
    mkdir -p ${RESULT_FILE_PATH}
    pytest -sv tests/e2e/nightly/multi_node/test_multi_node.py 2>&1 | tee $LOG_FILE
    ret=${PIPESTATUS[0]}
    set -e
    if [ "$LWS_WORKER_INDEX" -eq 0 ]; then
        if [ $ret -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_failure "Some tests failed!"
            mv LOG_FILE error_${LOG_FILE}
        fi
    fi
}

main() {
    check_npu_info
    check_and_config
    checkout_src
    install_sys_dependencies
    install_vllm
    if [[ "$CONFIG_YAML_PATH" == *"DeepSeek-R1-W8A8.yaml" ]]; then
        install_extra_components
    fi
    install_ais_bench
    cd "$WORKSPACE/source_code"
    . $SRC_DIR/vllm-ascend/tests/e2e/nightly/multi_node/scripts/build_mooncake.sh
    cd "$WORKSPACE/source_code/vllm-ascend"
    run_tests_with_log
}

main "$@"
