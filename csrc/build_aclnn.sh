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

# ============================================================================
# Snapshot cache hooks (opt-in).
#
# Enabled by exporting VLLM_ASCEND_SNAPSHOT_CACHE_DIR=<dir>; the build then
# transparently becomes a cached build, ccache-style:
#
#   1. restore:   newest snapshot for this target is unpacked into csrc/build
#   2. reconfigure: the restored build dir embeds paths of the machine that
#                   produced it (uv build-isolation temps) - regenerate
#   3. invalidate: per-op source hashes decide which ops to recompile
#   4. build:     the normal build below only recompiles invalidated ops
#   5. save:      new snapshot uploaded, LRU-evicted to the budget
#   6. self-check: invalidated ops must have fresh artifacts, else fail
#
# Without the env var the script behaves exactly as before.
# Env vars:
#   VLLM_ASCEND_SNAPSHOT_CACHE_DIR   cache directory (local store backend)
#   VLLM_ASCEND_CACHE_TARGET         target id (default: $SOC_VERSION)
#   VLLM_ASCEND_CACHE_BUDGET         LRU budget, default 5G
#   VLLM_ASCEND_CACHE_IMAGE_TAG      CANN image tag for the global hash
# ============================================================================

SNAPSHOT_CACHE_DIR="${VLLM_ASCEND_SNAPSHOT_CACHE_DIR:-}"
CACHE_TOOLS_DIR="${ROOT_DIR}/.github/workflows/scripts"
CACHE_MANIFEST_TOOL="${CACHE_TOOLS_DIR}/csrc_snapshot_manifest.py"
CACHE_LRU_TOOL="${CACHE_TOOLS_DIR}/csrc_cache_lru.py"
CACHE_TARGET="${VLLM_ASCEND_CACHE_TARGET:-${SOC_VERSION}}"
CACHE_BUDGET="${VLLM_ASCEND_CACHE_BUDGET:-5G}"
CACHE_IMAGE_TAG="${VLLM_ASCEND_CACHE_IMAGE_TAG:-}"
CACHE_TAG=""
CACHE_MISS_OPS=()
CACHE_BUILD_START=0

cache_enabled() {
    if [[ -z "${SNAPSHOT_CACHE_DIR}" ]]; then
        return 1
    fi
    if [[ ! -f "${CACHE_MANIFEST_TOOL}" || ! -f "${CACHE_LRU_TOOL}" ]]; then
        log "::warning::snapshot cache requested but tools missing under ${CACHE_TOOLS_DIR}; building without cache"
        return 1
    fi
    return 0
}

snapshot_csrc_hash() {
    git -C "${ROOT_DIR}" ls-files -s -- csrc setup.py CMakeLists.txt cmake \
        | sha256sum | awk '{print $1}'
}

cache_restore() {
    cache_enabled || return 0
    CACHE_TAG="$(snapshot_csrc_hash)" || return 0
    log "snapshot cache: dir=${SNAPSHOT_CACHE_DIR} target=${CACHE_TARGET} budget=${CACHE_BUDGET} tag=${CACHE_TAG:0:12}"

    if python3 "${CACHE_LRU_TOOL}" --store-dir "${SNAPSHOT_CACHE_DIR}" \
        --budget "${CACHE_BUDGET}" --restore --target "${CACHE_TARGET}" \
        --dest "${ROOT_DIR}/csrc"; then
        log "snapshot restored; reconfiguring to fix embedded tool paths"
        if ! cmake --regenerate-during-build -S "${ROOT_DIR}/csrc" \
            -B "${ROOT_DIR}/csrc/build" \
            -DHI_PYTHON="$(command -v python3)"; then
            log "::warning::reconfigure failed; falling back to full build"
            rm -rf -- "${ROOT_DIR}/csrc/build"
        fi
    else
        log "no snapshot for target ${CACHE_TARGET} (cold or evicted); full build"
        rm -rf -- "${ROOT_DIR}/csrc/build"
        return 0
    fi

    if [[ ! -d "${ROOT_DIR}/csrc/build" ]]; then
        # reconfigure discarded the snapshot above; plain full build.
        return 0
    fi
    local invalidate_log="${ROOT_DIR}/csrc/build/.cache-invalidate.log"
    python3 "${CACHE_MANIFEST_TOOL}" --invalidate --repo "${ROOT_DIR}" \
        --soc-version "${SOC_VERSION}" --image-tag "${CACHE_IMAGE_TAG}" \
        | tee "${invalidate_log}"
    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -eq 2 ]]; then
        log "invalidate says full rebuild; discarding snapshot"
        rm -rf -- "${ROOT_DIR}/csrc/build"
        return 0
    fi
    if [[ ${rc} -ge 3 ]]; then
        log "::warning::invalidate tool failed (rc=${rc}); full build"
        rm -rf -- "${ROOT_DIR}/csrc/build"
        return 0
    fi
    CACHE_MISS_OPS=($(python3 -c "
import json, sys
for line in open('${invalidate_log}'):
    if line.startswith('miss_list='):
        ops = json.loads(line[len('miss_list='):].strip())
        print(' '.join(ops))
        break
"))
    if ((${#CACHE_MISS_OPS[@]})); then
        log "snapshot cache: ${#CACHE_MISS_OPS[@]} op(s) invalidated: ${CACHE_MISS_OPS[*]}"
    else
        log "snapshot cache: all ops hit, only relink/packaging expected"
    fi
    CACHE_BUILD_START="$(date +%s)"
    return 0
}

cache_save() {
    cache_enabled || return 0
    [[ -n "${CACHE_BUILD_START}" ]] || return 0

    python3 "${CACHE_MANIFEST_TOOL}" --generate --repo "${ROOT_DIR}" \
        --soc-version "${SOC_VERSION}" --image-tag "${CACHE_IMAGE_TAG}" \
        || { log "::warning::manifest generation failed; snapshot not saved"; return 0; }

    # Freshness gate BEFORE the upload: if invalidated ops were not rebuilt,
    # persisting the snapshot first would poison the cache with a broken
    # build tree (verified live: a failed build used to overwrite the good
    # snapshot before this check ran).
    if ((${#CACHE_MISS_OPS[@]})); then
        log "snapshot cache: verifying freshness of invalidated ops"
        if ! python3 "${CACHE_MANIFEST_TOOL}" --check-freshness \
            --repo "${ROOT_DIR}" --soc-version "${SOC_VERSION}" \
            --ops "${CACHE_MISS_OPS[@]}" \
            --since-epoch "${CACHE_BUILD_START}"; then
            log "::error::invalidated ops were not rebuilt - cache would serve stale artifacts; failing the build"
            exit 1
        fi
    fi

    python3 "${CACHE_LRU_TOOL}" --store-dir "${SNAPSHOT_CACHE_DIR}" \
        --budget "${CACHE_BUDGET}" --upload --path "${ROOT_DIR}/csrc/build" \
        --target "${CACHE_TARGET}" --snapshot-id "${CACHE_TAG}" \
        || { log "::warning::snapshot upload failed; cache not updated"; return 0; }
    return 0
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
        "scatter_nd_update_sk"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "rms_norm_cast"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "lightning_indexer_quant"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "quant_lightning_indexer_v2"
        "quant_lightning_indexer_v2_metadata"
        "sparse_flash_mla"
        "sparse_flash_mla_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_situ_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "recurrent_gated_delta_rule"
        "recurrent_kda"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "chunk_kda_fwd"
        "kda_gate_cumsum"
        "kda_layout_swap12"
        "store_kv_block"
        "store_kv_block_metadata"
        "sparse_attention_score"
        "k2q_csr"
        "msa_index_score"
        "fused_sparse_attention_overlap"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_sk"
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
        "rms_norm_cast"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "lightning_indexer_quant"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "quant_lightning_indexer_v2"
        "quant_lightning_indexer_v2_metadata"
        "sparse_flash_mla"
        "sparse_flash_mla_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_situ_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "recurrent_gated_delta_rule"
        "recurrent_kda"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "chunk_kda_fwd"
        "kda_gate_cumsum"
        "kda_layout_swap12"
        "store_kv_block"
        "store_kv_block_metadata"
        "sparse_attention_score"
        "k2q_csr"
        "msa_index_score"
        "fused_sparse_attention_overlap"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend950 ]]; then
    log "matched SOC branch: ascend950"
    # ASCEND950 (A5) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "add_rms_norm_bias"
        "moe_gating_top_k_hash"
        "inplace_partial_rotary_mul"
        "kv_compress_epilog"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "quant_lightning_indexer_v2"
        "quant_lightning_indexer_v2_metadata"
        "kv_quant_sparse_attn_sharedkv"
        "kv_quant_sparse_attn_sharedkv_metadata"
        "hc_post"
        "hc_pre"
        "swiglu_group_quant"
        "situ_mx_quant"
        "indexer_compress_epilog_v2"
        "causal_conv1d"
        "recurrent_gated_delta_rule"
        "recurrent_kda"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "chunk_kda_fwd"
        "kda_gate_cumsum"
        "kda_layout_swap12"
        "store_kv_block"
        "store_kv_block_metadata"
        "k2q_csr"
        "sparse_attention_score"
        "mla_prolog_v3"
        "msa_index_score"
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

cache_restore


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
     log "VLLM_BATCH_INVARIANT is not set to 1, skipping batch_invariant_ops build"
   fi
) || build_rc=$?

# P0-1 hardening: this script has no `set -e`, so a failed build subshell
# previously fell through to cache_save, which masked the non-zero exit
# status (setup.py saw success) and could persist a snapshot of the broken
# tree. Capture the status, skip the snapshot on failure, and propagate.
if [[ "${build_rc:-0}" -ne 0 ]]; then
  log "::error::csrc build failed (rc=${build_rc}); snapshot not saved"
  exit "${build_rc}"
fi

# Only reached when the build subshell succeeded: persist the new snapshot.
cache_save

# build probe: force one cache-miss build to validate build-log artifact capture (safe to revert)
