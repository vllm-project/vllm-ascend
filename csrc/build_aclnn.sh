#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

case "$SOC_VERSION" in
    "ASCEND310P3")
        # currently, no custom aclnn ops support ASCEND310P3
        exit 0
        ;;
    "ASCEND910B1")
        CUSTOM_OPS="grouped_matmul_swiglu_quant"
        SOC_ARG="ascend910b"
        ;;
    "ASCEND910C")
        CUSTOM_OPS="grouped_matmul_swiglu_quant"
        SOC_ARG="ascend910_93"
        ;;
    *)
        echo "Unsupported SOC_VERSION: $SOC_VERSION"
        exit 1
        ;;
esac

# build custom ops
cd csrc
rm -rf build output
echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
bash build.sh -n $CUSTOM_OPS -c $SOC_ARG

# install custom ops
./output/CANN-custom_ops*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom
source $ROOT_DIR/vllm_ascend/_cann_ops_custom/vendors/customize/bin/set_env.bash
