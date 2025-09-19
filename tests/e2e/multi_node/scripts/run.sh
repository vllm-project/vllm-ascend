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
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
}

checkout_src() {
    echo "====> Checkout source code"
    mkdir -p "$SRC_DIR"

    # vllm-ascend
    if [ ! -d "$SRC_DIR/vllm-ascend" ]; then
        git clone --depth 1 https://github.com/vllm-project/vllm-ascend.git "$SRC_DIR/vllm-ascend"
    fi

    # vllm
    if [ ! -d "$SRC_DIR/vllm" ]; then
        git clone -b v0.10.2 https://github.com/vllm-project/vllm.git "$SRC_DIR/vllm"
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

run_tests() {
    echo "====> Run tests"
    cd "$SRC_DIR/vllm-ascend"
    pytest -v tests/e2e/multi_node/multi_node_dp.py
}

main() {
    check_npu_info
    check_and_config
    checkout_src
    install_sys_dependencies
    install_vllm
    run_tests
}

main "$@"
