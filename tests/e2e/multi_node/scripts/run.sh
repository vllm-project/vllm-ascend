#!/bin/bash
set -euo pipefail

export SRC_DIR="$WORKSPACE/source_code"

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
        git clone --depth 1 -b $VLLM_ASCEND_VERSION https://github.com/vllm-project/vllm-ascend.git "$SRC_DIR/vllm-ascend"
    fi

    # vllm
    if [ ! -d "$SRC_DIR/vllm" ]; then
        git clone -b $VLLM_VERSION https://github.com/vllm-project/vllm.git "$SRC_DIR/vllm"
    fi

    #mooncake
    if [ ! -d "$SRC_DIR/Mooncake" ]; then
        git clone https://github.com/kvcache-ai/Mooncake.git "$SRC_DIR/Mooncake"
        cd "$SRC_DIR/Mooncake"
        git checkout 06cc217504a6f1b0cdaa26b096b985651b262748
        cd -
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

install_mooncake() {
    echo "====> Install mooncake"
    apt-get update
    apt install -y --allow-change-held-packages python3 python-is-python3
    apt-get install -y --no-install-recommends mpich libmpich-dev
    cd $SRC_DIR/Mooncake
    sed -i '/option(USE_ASCEND_DIRECT)/s/OFF/ON/' mooncake-common/common.cmake
    bash dependencies.sh --yes
    mkdir build
    cd -
    cd $SRC_DIR/Mooncake/build
    cmake ..
    make -j
    make install
    cd -
}

run_tests() {
    echo "====> Run tests"
    cd "$SRC_DIR/vllm-ascend"
    pytest -sv tests/e2e/multi_node/test_multi_dp.py
}

main() {
    check_npu_info
    check_and_config
    checkout_src
    install_sys_dependencies
    install_vllm
    #install_mooncake
    run_tests
}

main "$@"
