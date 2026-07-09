#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2

# ============================================================
# CANN 环境检测 -- 兼容 CANN 9.0.0 / 9.1.0+
# 使用 ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH / 自动检测 / 动态回退
# ============================================================
if [[ -n "${ASCEND_TOOLKIT_HOME}" && -d "${ASCEND_TOOLKIT_HOME}" ]]; then
    CANN_ROOT="${ASCEND_TOOLKIT_HOME}"
elif [[ -n "${ASCEND_HOME_PATH}" && -d "${ASCEND_HOME_PATH}" ]]; then
    CANN_ROOT="${ASCEND_HOME_PATH}"
else
    CANN_ROOT=$(find /usr/local/Ascend -maxdepth 1 -type d -name 'cann-*' 2>/dev/null | sort -V | tail -n1)
    if [[ -z "${CANN_ROOT}" ]]; then
        echo "ERROR: CANN root not found. Checked ASCEND_TOOLKIT_HOME, ASCEND_HOME_PATH, and /usr/local/Ascend/cann-*" >&2
        exit 1
    fi
    echo "[INFO] Auto-detected CANN root: ${CANN_ROOT}"
fi
echo "[INFO] Using CANN root: ${CANN_ROOT}"

# ============================================================
# aicpu_engine_struct.h 兼容性处理
# CANN 9.1.0 移除了该头文件，但 aicpu_ext_info_handle.h 仍引用它。
# 创建 stub 头文件（使用 /tmp 路径，避免被后续 rm -rf build 删除）
# ============================================================
STUB_DIR="/tmp/vllm_ascend_ai_cpu_stub"
mkdir -p "${STUB_DIR}"

AICPU_ENGINE_STRUCT_FILE=$(find -L "${CANN_ROOT}" -name "aicpu_engine_struct.h" 2>/dev/null | head -n1)
if [[ -n "${AICPU_ENGINE_STRUCT_FILE}" ]]; then
    AICPU_ENGINE_DIR=$(dirname "${AICPU_ENGINE_STRUCT_FILE}")
    export CPATH="${AICPU_ENGINE_DIR}:${CPATH}"
    echo "[INFO] Found aicpu_engine_struct.h at ${AICPU_ENGINE_STRUCT_FILE}"
else
    STUB_FILE="${STUB_DIR}/aicpu_engine_struct.h"
    if [[ ! -f "${STUB_FILE}" ]]; then
        cat > "${STUB_FILE}" << 'STUB_EOF'
#ifndef AICPU_ENGINE_STRUCT_H
#define AICPU_ENGINE_STRUCT_H
/* stub for CANN 9.1.0+ - aicpu_engine_struct.h removed from installation */
typedef struct aicpu_engine_struct {
    int reserved;
} aicpu_engine_struct_t;
#endif
STUB_EOF
    fi
    export CPATH="${STUB_DIR}:${CPATH}"
    echo "[WARN] aicpu_engine_struct.h not found in CANN, using stub at ${STUB_FILE}"
fi

export ASCEND_TOOLKIT_PATH="${CANN_ROOT}"
echo "[INFO] Auto set ASCEND_TOOLKIT_PATH=${ASCEND_TOOLKIT_PATH}"

# ---- 原有 SOC 版本分支逻辑开始 ----

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    exit 0
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
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
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "depdendency catlass is missing, try to fetch it..."
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
    fi
        # HCCL moe_distribute_base.h 兼容（CANN 9.0.0/9.1.0+ 通用）
    # 使用本地文件（HcclOpResParamCustom + remoteRes），不复制 CANN 版本
    LOCAL_HCCL_SRC="${ROOT_DIR}/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/moe_distribute_base.h"
    if [[ -f "${LOCAL_HCCL_SRC}" ]]; then
        mkdir -p "${ROOT_DIR}/csrc/utils/inc/kernel"
        mkdir -p "${ROOT_DIR}/csrc/mc2/dispatch_ffn_combine/op_kernel/utils"
        mkdir -p "${ROOT_DIR}/csrc/mc2/dispatch_gmm_combine_decode/op_kernel"
        mkdir -p "${ROOT_DIR}/csrc/mc2/moe_combine_normal/op_kernel/utils"
        mkdir -p "${ROOT_DIR}/csrc/mc2/moe_dispatch_normal/op_kernel/utils"
        mkdir -p "${ROOT_DIR}/csrc/mc2/notify_dispatch/op_kernel/kernel"
        for tgt in "${ROOT_DIR}/csrc/utils/inc/kernel" \
                  "${ROOT_DIR}/csrc/mc2/dispatch_ffn_combine/op_kernel/utils" \
                  "${ROOT_DIR}/csrc/mc2/dispatch_gmm_combine_decode/op_kernel" \
                  "${ROOT_DIR}/csrc/mc2/moe_combine_normal/op_kernel/utils" \
                  "${ROOT_DIR}/csrc/mc2/moe_dispatch_normal/op_kernel/utils" \
                  "${ROOT_DIR}/csrc/mc2/notify_dispatch/op_kernel/kernel"; do
            cp "${LOCAL_HCCL_SRC}" "${tgt}/"
        done
    else
        echo "[WARN] Local moe_distribute_base.h not found, HCCL ops may fail"
    fi


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
    exit 0
fi

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
  find "$install_dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf -- {} +
  shopt -s nullglob
  runs=(./build/cann-ops-transformer*.run)
  shopt -u nullglob
  (( ${#runs[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#runs[@]}" >&2; exit 1; }
  chmod +x -- "${runs[0]}" || true
  "${runs[0]}" --install-path="${install_dir}"
)
