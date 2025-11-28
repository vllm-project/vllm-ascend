#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    # ASCEND310P series
    # currently, no custom aclnn ops for ASCEND310 series
    # CUSTOM_OPS=""
    # SOC_ARG="ascend310p"
    exit 0
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list"
    SOC_ARG="ascend910_93"
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi

# build custom ops
cd csrc
rm -rf build output
echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
bash build.sh -n $CUSTOM_OPS -c $SOC_ARG

# install custom ops to vllm_ascend/_cann_ops_custom
./output/CANN-custom_ops*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom
source $ROOT_DIR/vllm_ascend/_cann_ops_custom/vendors/customize/bin/set_env.bash
