#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

git config --global --add safe.directory "$ROOT_DIR"

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    # ASCEND310P series
    # currently, no custom aclnn ops for ASCEND310 series
    # CUSTOM_OPS=""
    # SOC_ARG="ascend310p"
    exit 0
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer;sparse_flash_attention;dispatch_ffn_combine"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer;sparse_flash_attention;dispatch_ffn_combine"
    SOC_ARG="ascend910_93"
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi

git submodule init
git submodule update


# For the compatibility of CANN8.5 and CANN8.3: copy and modify moe_distribute_base.h
file_path=$(find /usr/local/Ascend/ascend-toolkit -name "moe_distribute_base.h" 2>/dev/null | head -n1)
if [ -z "$file_path" ]; then
    echo "cannot find moe_distribute_base.h file in CANN env"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET_DIR="$SCRIPT_DIR/dispatch_ffn_combine/op_kernel/utils/"
TARGET_FILE="$TARGET_DIR/$(basename "$file_path")"

echo "*************************************"
echo $file_path
echo "$TARGET_DIR"
cp "$file_path" "$TARGET_DIR"

sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"


# build custom ops
cd csrc
rm -rf build output
echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
bash build.sh -n $CUSTOM_OPS -c $SOC_ARG

# install custom ops to vllm_ascend/_cann_ops_custom
./output/CANN-custom_ops*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom
