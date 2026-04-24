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
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}

    CUSTOM_OPS_ARRAY=(
        "grouped_matmul_swiglu_quant_clamp"
        "sparse_flash_attention"
        "lightning_indexer"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"


        "add_rms_norm_bias"

        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"

        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
        "lightning_indexer_quant_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"

        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        
        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"

    )


    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
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
    # for dispatch_gmm_combine_decode
    yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/csrc/utils/inc/kernel"
    # for dispatch_ffn_combine
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    TARGET_DIR="$SCRIPT_DIR/mc2/dispatch_ffn_combine/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"

    echo "*************************************"
    echo $HCCL_STRUCT_FILE_PATH
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"

    # for dispatch_normal and combine_normal
    TARGET_DIR="$SCRIPT_DIR/mc2/moe_dispatch_normal/op_kernel/utils/"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"

    TARGET_DIR="$SCRIPT_DIR/mc2/moe_combine_normal/op_kernel/utils/"
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    
    # CUSTOM_OPS_ARRAY=(
    #     "dispatch_ffn_combine"
    #     "dispatch_gmm_combine_decode"
    #     "moe_combine_normal"
    #     "moe_dispatch_normal"
    #     "dispatch_layout"
    #     "notify_dispatch"
    # )
    CUSTOM_OPS_ARRAY=(
        "grouped_matmul_swiglu_quant_clamp"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "notify_dispatch"
        "dispatch_ffn_combine"
        "dispatch_gmm_combine_decode"
        "moe_combine_normal"
        "moe_dispatch_normal"
        "dispatch_layout"

        "sparse_flash_attention"
        "lightning_indexer"


        "add_rms_norm_bias"

        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"

        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"

        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"

        "rms_norm_dynamic_quant"
        "inplace_partial_rotary_mul"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend910_95 ]]; then
    # ASCEND910B (A2) series
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
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH=${ABSOLUTE_CATLASS_PATH}:${CPATH}

    CUSTOM_OPS_ARRAY=(
        "moe_gating_top_k_hash"
        
        "indexer_compress_epilog"
        "inplace_partial_rotary_mul"
        "kv_compress_epilog"
        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
        "kv_quant_sparse_attn_sharedkv"
        "kv_quant_sparse_attn_sharedkv_metadata"

        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
    )


    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_95"
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi


# # build custom ops
# cd csrc
# rm -rf build output build_out
# echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
# bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# # install custom ops to vllm_ascend/_cann_ops_custom
# ./build/cann-ops-transformer*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom


(
  set -euo pipefail

  cd csrc
  rm -rf -- build output build_out

  : "${ROOT_DIR:?ROOT_DIR is not set}"
  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  echo "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"

  install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"

  mkdir -p -- "$install_dir"

  # 删除 install_dir 下除 .gitkeep 外的所有内容（包含隐藏文件/目录）
  find "$install_dir" -mindepth 1 \
    ! -name '.gitkeep' \
    -exec rm -rf -- {} +

  shopt -s nullglob
  runs=(./build/cann-ops-transformer*.run)
  shopt -u nullglob

  (( ${#runs[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#runs[@]}" >&2; exit 1; }

  chmod +x -- "${runs[0]}" || true
  "${runs[0]}" --install-path="${install_dir}"
)