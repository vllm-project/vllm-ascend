#!/bin/bash

ROOT_DIR=$1
SOC_VERSION=$2
: "${ROOT_DIR:?ROOT_DIR is not set}"

log() {
    echo "[build_aclnn] $*"
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
    CUSTOM_OPS_ARRAY=(
        "causal_conv1d_v310"
        "recurrent_gated_delta_rule_v310"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    log "matched SOC branch: ascend910b"
    # ASCEND910B (A2) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    CATLASS_COMMIT=$(git config -f "${ROOT_DIR}/.gitmodules" --get submodule.csrc/third_party/catlass.commit)
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        git submodule sync
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
        cd "${ROOT_DIR}/csrc/third_party/catlass" || exit 1
        git fetch origin
        git checkout "${CATLASS_COMMIT}" || exit 1
        cd - || exit 1
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH="${ABSOLUTE_CATLASS_PATH}${CPATH:+:${CPATH}}"
    log "catlass include=${ABSOLUTE_CATLASS_PATH}"

    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_v2"
        "moe_grouped_matmul"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "matmul_allreduce_add_rmsnorm"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "lightning_indexer_quant"
        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
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
        "hamming_dist_top_k"
        "reshape_and_cache_bnsd"
        "recurrent_gated_delta_rule"
        "fused_gdn_gating"
        "ngram_spec_decode"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    CATLASS_COMMIT=$(git config -f "${ROOT_DIR}/.gitmodules" --get submodule.csrc/third_party/catlass.commit)
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        git submodule sync
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
        cd "${ROOT_DIR}/csrc/third_party/catlass" || exit 1
        git fetch origin
        git checkout "${CATLASS_COMMIT}" || exit 1
        cd - || exit 1
    fi
    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_v2"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_w4_a8"
        "dispatch_ffn_combine_bf16"
        "dispatch_gmm_combine_decode"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "moe_grouped_matmul"
        "lightning_indexer_quant"
        "compressor"
        "quant_lightning_indexer"
        "quant_lightning_indexer_metadata"
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
        "hamming_dist_top_k"
        "reshape_and_cache_bnsd"
        "recurrent_gated_delta_rule"
        "fused_gdn_gating"
        "ngram_spec_decode"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend950 ]]; then
    log "matched SOC branch: ascend950"
    # ASCEND950 (A5) series
    # dependency: catlass
    git config --global --add safe.directory "$ROOT_DIR"
    CATLASS_PATH=${ROOT_DIR}/csrc/third_party/catlass/include
    CATLASS_COMMIT=$(git config -f "${ROOT_DIR}/.gitmodules" --get submodule.csrc/third_party/catlass.commit)
    if [[ ! -d "${CATLASS_PATH}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        git submodule sync
        if ! git submodule update --init --recursive; then
            echo "fetch failed"
            exit 1
        fi
        cd "${ROOT_DIR}/csrc/third_party/catlass" || exit 1
        git fetch origin
        git checkout "${CATLASS_COMMIT}" || exit 1
        cd - || exit 1
    fi
    ABSOLUTE_CATLASS_PATH=$(cd "${CATLASS_PATH}" && pwd)
    export CPATH="${ABSOLUTE_CATLASS_PATH}${CPATH:+:${CPATH}}"
    log "catlass include=${ABSOLUTE_CATLASS_PATH}"

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
        "hc_pre"
        "swiglu_group_quant"
        "load_index_kv_cache"
        "indexer_compress_epilog_v2"
        "causal_conv1d"
        "recurrent_gated_delta_rule"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
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

  # TRACE_PREPROCESSOR_HOOK_START
  # -----------------------------------------------------------------
  # Run trace_preprocessor.py on op_kernel source directories BEFORE
  # compilation to replace TRACE_POINT("label","B/E") with unique
  # integer point_ids.  Sources are backed up to .preproc_bak first,
  # then restored after build (no git dependency).
  # Set TRACE_DISABLE=1 to skip this step.
  # -----------------------------------------------------------------
  _TRACE_PREPROCESSOR="${ROOT_DIR}/csrc/scripts/trace/trace_preprocessor.py"
  _TRACE_PROCESSED_DIRS=()
  if [[ -z "${TRACE_DISABLE:-}" ]] && [[ -f "${_TRACE_PREPROCESSOR}" ]]; then
      _TRACE_PYTHON=""
      for _py in python3 python; do
          if command -v "${_py}" &>/dev/null; then
              _TRACE_PYTHON="${_py}"
              break
          fi
      done
      if [[ -z "${_TRACE_PYTHON}" ]]; then
          log "trace_preprocessor: WARNING — no python found, skip preprocessing (kernel will write point_id=0)"
      else
          _TRACE_BUILD_OUT="${ROOT_DIR}/csrc/trace_preprocess_out"
          mkdir -p "${_TRACE_BUILD_OUT}"
          for _op_kernel_dir in \
              "${ROOT_DIR}/csrc/mc2/dispatch_ffn_combine/op_kernel" \
              "${ROOT_DIR}/csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel"; do
              if [[ -d "${_op_kernel_dir}" ]]; then
                  _op_name=$(basename "$(dirname "${_op_kernel_dir}")")
                  _out_subdir="${_TRACE_BUILD_OUT}/${_op_name}"
                  mkdir -p "${_out_subdir}"
                  _bak_dir="${_op_kernel_dir}.preproc_bak"
                  log "trace_preprocessor: backing up ${_op_kernel_dir} -> ${_bak_dir}"
                  rm -rf "${_bak_dir}"
                  cp -r "${_op_kernel_dir}" "${_bak_dir}"
                  log "trace_preprocessor: processing ${_op_kernel_dir} -> ${_out_subdir}"
                  if ${_TRACE_PYTHON} "${_TRACE_PREPROCESSOR}" \
                      "${_op_kernel_dir}" "${_out_subdir}" --modify; then
                      log "trace_preprocessor: OK, point_map.json -> ${_out_subdir}/point_map.json"
                      # Verify: count remaining TRACE_POINT calls (should be 0 after --modify).
                      # Use TRACE_POINT\s*\(\s*" to match calls like TRACE_POINT("label","B")
                      # but NOT the macro definition #define TRACE_POINT(label, event) 0
                      # grep returns 1 on no-match; must tolerate with set -o pipefail
                      _remaining=$(grep -r 'TRACE_POINT\s*(\s*"' "${_op_kernel_dir}" -l 2>/dev/null | wc -l) || _remaining=0
                      if [[ "${_remaining}" -gt 0 ]]; then
                          log "trace_preprocessor: ERROR — ${_remaining} files still contain TRACE_POINT after --modify!"
                          log "trace_preprocessor: restoring backup and skipping this dir"
                          rm -rf "${_op_kernel_dir}"
                          mv "${_bak_dir}" "${_op_kernel_dir}"
                      else
                          log "trace_preprocessor: verified — 0 TRACE_POINT strings remaining in ${_op_kernel_dir}"
                          _TRACE_PROCESSED_DIRS+=("${_op_kernel_dir}")
                      fi
                  else
                      log "trace_preprocessor: WARNING — preprocessing failed, restoring backup"
                      rm -rf "${_op_kernel_dir}"
                      mv "${_bak_dir}" "${_op_kernel_dir}"
                  fi
              fi
          done
      fi
  fi
  # TRACE_PREPROCESSOR_HOOK_END

  bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
  log "build.sh finished"

  # TRACE_PREPROCESSOR_HOOK_CLEANUP
  # Restore op_kernel sources from .preproc_bak (no git dependency)
  for _op_kernel_dir in "${_TRACE_PROCESSED_DIRS[@]}"; do
      _bak_dir="${_op_kernel_dir}.preproc_bak"
      if [[ -d "${_bak_dir}" ]]; then
          log "trace_preprocessor cleanup: restoring ${_op_kernel_dir} from backup"
          rm -rf "${_op_kernel_dir}"
          mv "${_bak_dir}" "${_op_kernel_dir}"
      fi
  done

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
