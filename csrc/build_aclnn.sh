#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

if [[ "$SOC_VERSION" =~ ^[Aa][Ss][Cc][Ee][Nn][Dd]310 ]]; then
    # ASCEND310P series
    # currently, no custom aclnn ops for ASCEND310 series
    exit 0
elif [[ "$SOC_VERSION" =~ ^[Aa][Ss][Cc][Ee][Nn][Dd]910[Bb][0-9]$ ]]; then
    # ASCEND910B (A2) series
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^[Aa][Ss][Cc][Ee][Nn][Dd]910_93[0-9]{2}$ ]]; then
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
