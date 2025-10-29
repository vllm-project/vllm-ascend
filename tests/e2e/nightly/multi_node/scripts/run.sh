#!/bin/bash
set -euo pipefail

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Configuration
GOVER=1.23.8
LOG_DIR="/root/.cache/tests/logs"
OVERWRITE_LOGS=true
SRC_DIR="$WORKSPACE/source_code"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
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

download_go() {
    ARCH=$(uname -m)
    GOVER=1.23.8
    if [ "$ARCH" = "aarch64" ]; then
        ARCH="arm64"
    elif [ "$ARCH" = "x86_64" ]; then
        ARCH="amd64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    # Download Go
    echo "Downloading Go $GOVER..."
    wget -q --show-progress https://golang.google.cn/dl/go$GOVER.linux-$ARCH.tar.gz
    check_success "Failed to download Go $GOVER"

    # Install Go
    echo "Installing Go $GOVER..."
    tar -C /usr/local -xzf go$GOVER.linux-$ARCH.tar.gz
    check_success "Failed to install Go $GOVER"

    # Clean up downloaded file
    rm -f go$GOVER.linux-$ARCH.tar.gz
    check_success "Failed to clean up Go installation file"

    print_success "Go $GOVER installed successfully"
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

kill_npu_processes() {
  pgrep python3 | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  sleep 4
}

run_tests() {
    set +e
    kill_npu_processes
    pytest -sv tests/e2e/nightly/multi_node/test_multi_node.py
    ret=$?
    if [ "$LWS_WORKER_INDEX" -eq 0 ]; then
        if [ $ret -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_error "Some tests failed!"
            kubectl delete pod $CONTROLLER_NAME -n vllm-project
        fi
    fi
    set -e
}

install_kubectl() {
    arch=$(uname -m)
    KUBECTL=/root/.cache/.kube/kubectl
    if echo "$arch" | grep -qiE "arm|aarch64"; then
        echo "Detected ARM architecture: $arch"
        KUBECTL="$KUBECTL"_arm
    fi
    install -o root -g root -m 0755 $KUBECTL /usr/local/bin/kubectl
    echo "$SECRET" | base64 -d > /tmp/kubeconfig
    export KUBECONFIG=/tmp/kubeconfig
}

main() {
    check_npu_info
    check_and_config
    checkout_src
    install_sys_dependencies
    install_kubectl
    install_vllm
    install_ais_bench
    # to speed up mooncake build process, install Go here
    install_go
    cd "$WORKSPACE/source_code"
    . $SRC_DIR/vllm-ascend/tests/e2e/nightly/multi_node/scripts/build_mooncake.sh \
    pooling_async_memecpy_v1 9d96b2e1dd76cc601d76b1b4c5f6e04605cd81d3
    cd "$WORKSPACE/source_code/vllm-ascend"
    run_tests
}

main "$@"
