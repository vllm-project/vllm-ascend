#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    # ASCEND310P series
    CUSTOM_OPS="causal_conv1d_v310"
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}


    CUSTOM_OPS="moe_grouped_matmul;grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer_vllm;sparse_flash_attention;matmul_allreduce_add_rmsnorm;moe_init_routing_custom;moe_gating_top_k;add_rms_norm_bias;apply_top_k_top_p_custom;transpose_kv_cache_by_block;copy_and_expand_eagle_inputs;causal_conv1d;lightning_indexer_quant;"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
    # for dispatch_gmm_combine_decode
    yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/csrc/utils/inc/kernel"

    CUSTOM_OPS_ARRAY=(
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer_vllm"
        "sparse_flash_attention"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_bf16"
        "dispatch_gmm_combine_decode"
        "moe_combine_normal"
        "moe_dispatch_normal"
        "dispatch_layout"
        "notify_dispatch"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "moe_grouped_matmul"
        "lightning_indexer_quant"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
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
bash build.sh -n "$CUSTOM_OPS" -c "$SOC_ARG"

# install custom ops to vllm_ascend/_cann_ops_custom
./output/CANN-custom_ops*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom
