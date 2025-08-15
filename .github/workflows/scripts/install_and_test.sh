#!/bin/bash

set -euo pipefail
export WORKSPACE="/home/workspace"

check_npu_info() {
    npu-smi info
    cat /usr/local/Ascend/ascend-toolkit/latest/"$(uname -i)"-linux/ascend_toolkit_install.info
}

check_and_config() {
    # config mirror
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

}

install_sys_dependencies() {
    echo "====> Install system dependencies"
    cd $WORKSPACE
    # install sys dependencies
    apt-get update -y
    apt-get -y install `cat /root/workspace/packages.txt`
    apt-get -y install gcc g++ cmake libnuma-dev iproute2
    # kimi-k2 dependency
    pip install blobfile
}

install_vllm() {
    # install vllm
    cd $WORKSPACE/vllm-empty
    VLLM_TARGET_DEVICE=empty pip install -e .

    # install vllm-ascend
    cd $WORKSPACE
    pip install -e .
}

wait_for_server() {
    echo "====> Waiting for server to start"
}

main() {
    NODE_TYPE=$1
    if [ -n "${2:-}" ]; then
        export MASTER_ADDR="$2"
    fi
    check_npu_info
    check_and_config
    install_sys_dependencies
    install_vllm
    echo "====> Installation completed successfully"
    echo "====> Starting multi node tests"
    # test data parallel on mp backend
    . $WORKSPACE/examples/online_serving/multi_node_dp.sh "$NODE_TYPE"

    # test pipline parallel on ray backend
    sleep 1000
}

main "$@"

