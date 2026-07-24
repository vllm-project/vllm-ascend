#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2
: "${ROOT_DIR:?ROOT_DIR is not set}"

log() {
    echo "[build_aclnn] $*"
}

setup_catlass_dependency() {
    local catlass_path="${ROOT_DIR}/csrc/third_party/catlass/include"
    local catlass_commit
    local absolute_catlass_path

    git config --global --add safe.directory "$ROOT_DIR"
    catlass_commit=$(git config -f "${ROOT_DIR}/.gitmodules" --get submodule.csrc/third_party/catlass.commit)
    if [[ ! -d "${catlass_path}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        git submodule sync
        if ! git submodule update --init --recursive; then
            log "fetch failed"
            exit 1
        fi
        cd "${ROOT_DIR}/csrc/third_party/catlass" || exit 1
        git fetch origin
        git checkout "${catlass_commit}" || exit 1
        cd - || exit 1
    fi
    absolute_catlass_path=$(cd "${catlass_path}" && pwd)
    export CPATH="${absolute_catlass_path}${CPATH:+:${CPATH}}"
    log "catlass include=${absolute_catlass_path}"
}

resolve_op_dir() {
    local op_name=$1
    local candidate_dir
    for candidate_dir in \
        "${ROOT_DIR}/csrc/moe/${op_name}" \
        "${ROOT_DIR}/csrc/gmm/${op_name}" \
        "${ROOT_DIR}/csrc/attention/${op_name}" \
        "${ROOT_DIR}/csrc/mc2/${op_name}" \
        "${ROOT_DIR}/csrc/ffn/${op_name}" \
        "${ROOT_DIR}/csrc/posembedding/${op_name}"; do
        if [[ -d "${candidate_dir}" ]]; then
            echo "${candidate_dir}"
            return 0
        fi
    done
    find "${ROOT_DIR}/csrc" -maxdepth 3 -type d -name "${op_name}" -print -quit 2>/dev/null
}

log_selected_ops() {
    local op_name
    local op_path
    local kernel_cpp_file_count

    log "resolved SOC_ARG=${SOC_ARG}"
    log "resolved CUSTOM_OPS=${CUSTOM_OPS}"
    log "custom op count=${#CUSTOM_OPS_ARRAY[@]}"
    for op_name in "${CUSTOM_OPS_ARRAY[@]}"; do
        op_path=$(resolve_op_dir "${op_name}")
        if [[ -z "${op_path}" ]]; then
            log "op ${op_name}: dir=<missing>"
            continue
        fi
        kernel_cpp_file_count=0
        if [[ -d "${op_path}/op_kernel" ]]; then
            kernel_cpp_file_count=$(find "${op_path}/op_kernel" -maxdepth 1 -name '*.cpp' | wc -l | tr -d ' ')
        fi
        log "op ${op_name}: dir=${op_path} cmake=$([[ -f "${op_path}/CMakeLists.txt" ]] && echo yes || echo no) op_host_cmake=$([[ -f "${op_path}/op_host/CMakeLists.txt" ]] && echo yes || echo no) op_kernel_cpp_count=${kernel_cpp_file_count}"
    done
}

log "start: ROOT_DIR=${ROOT_DIR:-<unset>} SOC_VERSION=${SOC_VERSION:-<unset>} cwd=$(pwd)"
log "env: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    log "matched SOC branch: ascend310"
    # ASCEND310P series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "causal_conv1d_v310"
        "recurrent_gated_delta_rule_v310"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    log "matched SOC branch: ascend910b"
    # ASCEND910B (A2) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_v2"
        "moe_grouped_matmul"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "lightning_indexer_quant"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "recurrent_gated_delta_rule"
        "ngram_spec_decode"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
        "store_kv_block_metadata"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_v2"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_w4_a8"
        "dispatch_ffn_combine_bf16"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "moe_grouped_matmul"
        "lightning_indexer_quant"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "recurrent_gated_delta_rule"
        "ngram_spec_decode"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
        "store_kv_block_metadata"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend950 ]]; then
    log "matched SOC branch: ascend950"
    # ASCEND950 (A5) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "moe_gating_top_k_hash"
        "indexer_compress_epilog"
        "inplace_partial_rotary_mul"
        "kv_compress_epilog"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "kv_quant_sparse_attn_sharedkv"
        "kv_quant_sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "hc_pre"
        "swiglu_group_quant"
        "load_index_kv_cache"
        "indexer_compress_epilog_v2"
        "causal_conv1d"
        "recurrent_gated_delta_rule"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
        "store_kv_block_metadata"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend950"
else
    # others
    # currently, no custom aclnn ops for other series
    log "no custom ACLNN ops configured for SOC_VERSION=${SOC_VERSION}; skip build_aclnn"
    exit 0
fi

log_selected_ops


# # build custom ops
# cd csrc
# rm -rf build output build_out
# echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
# bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# # install custom ops to vllm_ascend/_cann_ops_custom
# ./build/cann-ops-transformer*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom


(
  set -euo pipefail

  : "${ROOT_DIR:?ROOT_DIR is not set}"

  log "subshell cwd before cd=$(pwd)"
  cd "${ROOT_DIR}/csrc"
  log "subshell cwd after cd=$(pwd)"
  log "preserving csrc/build and cleaning output dirs"
  rm -rf -- output build_out

  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  log "build command: bash build.sh --pkg --ops=\"${CUSTOM_OPS}\" --soc=\"${SOC_ARG}\""
  log "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
  log "build.sh finished"

  custom_ops_install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"
  log "custom_ops_install_dir=${custom_ops_install_dir}"

  mkdir -p -- "$custom_ops_install_dir"

  # Remove all top-level entries under custom_ops_install_dir except .gitkeep, including hidden files and directories.
  find "$custom_ops_install_dir" -mindepth 1 -maxdepth 1 \
    ! -name '.gitkeep' \
    -exec rm -rf -- {} +

  shopt -s nullglob
  installer_candidates=(./build/cann-ops-transformer*.run)
  shopt -u nullglob

  log "installer candidate count=${#installer_candidates[@]}"
  for installer_file in "${installer_candidates[@]}"; do
    log "installer candidate: $(ls -lh "${installer_file}")"
  done

  (( ${#installer_candidates[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#installer_candidates[@]}" >&2; exit 1; }

  chmod +x -- "${installer_candidates[0]}" || true
  log "running installer: ${installer_candidates[0]}"
  "${installer_candidates[0]}" --install-path="${custom_ops_install_dir}"
  # CANN leaves generated vendor script dirs owner-read-only; keep repo-local
  # editable-build artifacts removable by the non-root user who built them.
  if [[ -d "${custom_ops_install_dir}/vendors/custom_transformer/scripts" ]]; then
    chmod u+w "${custom_ops_install_dir}/vendors/custom_transformer/scripts"
  fi
  log "installer finished"
  log "installed files under ${custom_ops_install_dir} (maxdepth=4, first 120 entries):"
  { find "${custom_ops_install_dir}" -mindepth 1 -maxdepth 4 -print | sort | head -n 120 | sed 's#^#[build_aclnn] install: #'; } || true

  # install batch_invariant run package and whl package
  if [[ "${VLLM_BATCH_INVARIANT:-0}" == "1" ]]; then
    log "VLLM_BATCH_INVARIANT=1, installing batch_invariant run package and whl package..."

    # call separate installation script
    batch_invariant_script="${ROOT_DIR}/csrc/build_batch_invariant_ops.sh"
    if [[ -f "${batch_invariant_script}" ]]; then
      log "Calling batch_invariant_ops build script: ${batch_invariant_script}"
      bash "${batch_invariant_script}" "${SOC_ARG}"
    else
      log "Warning: batch_invariant_ops build script not found at ${batch_invariant_script}"
    fi
  else
    log "VLLM_BATCH_INVARIANT is not set to 1, skipping batch_invariant ops build"
  fi
)
