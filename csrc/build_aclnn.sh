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
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer;sparse_flash_attention"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    # depdendency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "depdendency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    # depdendency: cann-toolkit file moe_distribute_base.h
    HCCL_STRUCT_FILE_PATH=$(find -L "${ASCEND_TOOLKIT_HOME}" -name "moe_distribute_base.h" 2>/dev/null | head -n1)
    if [ -z "$HCCL_STRUCT_FILE_PATH" ]; then
        echo "cannot find moe_distribute_base.h file in CANN env"
        exit 1
    fi
    # for dispatch_gmm_combine_decode
    yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/csrc/dispatch_gmm_combine_decode/op_kernel"
    # for dispatch_ffn_combine
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    TARGET_DIR="$SCRIPT_DIR/dispatch_ffn_combine/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"

    echo "*************************************"
    echo $HCCL_STRUCT_FILE_PATH
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"
    CUSTOM_OPS="grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer;sparse_flash_attention;dispatch_ffn_combine;dispatch_gmm_combine_decode;"
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
