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
        git clone -b pooling_async_memecpy_v1 https://github.com/AscendTransport/Mooncake "$SRC_DIR/Mooncake"
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
    apt-get update -y
    apt-get install -y --no-install-recommends mpich libmpich-dev
    cd $SRC_DIR/Mooncake
    bash dependencies.sh --yes
    apt purge mpich libmpich-dev -y
    apt purge openmpi-bin -y
    apt purge openmpi-bin libopenmpi-dev -y
    apt install mpich libmpich-dev -y
    export CPATH=/usr/lib/aarch64-linux-gnu/mpich/include/:$CPATH
    export CPATH=/usr/lib/aarch64-linux-gnu/openmpi/lib:$CPATH

    mkdir build
    cd -
    cd $SRC_DIR/Mooncake/build
    cmake ..
    make -j
    make install
    cp mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
    cp mooncake-transfer-engine/src/libtransfer_engine.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
    cd -
}

kill_npu_processes() {
  pgrep python3 | xargs -r kill -9
  pgrep VLLM | xargs -r kill -9

  sleep 4
}

run_tests() {
    echo "====> Run tests"
    shopt -s nullglob

    pytest -sv tests/e2e/nightly/multi_node/test_multi_node.py
    kill_npu_processes

    shopt -u nullglob
}

main() {
    check_npu_info
    check_and_config
    checkout_src
    install_sys_dependencies
    install_vllm
    install_mooncake
    run_tests
}

main "$@"
