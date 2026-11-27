#!/usr/bin/env bash
set -euo pipefail

backend=${1:-}
if [[ "${backend}" != "catccos" && "${backend}" != "native" ]]; then
    echo "Usage: $0 <catccos|native>" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
profile_tag=${PROFILE_TAG:?Set PROFILE_TAG to a unique output name}
profile_root=${PROFILE_ROOT:-/home/z00956592/profiles}
profile_dir=${profile_root}/${profile_tag}
profile_iters=${PROFILE_ITERS:-20}
sync_boundaries=${SYNC_BOUNDARIES:-0}
catccos_sync_after_launch=${CATCCOS_SYNC_AFTER_LAUNCH:-false}
catccos_max_tokens_per_rank=${CATCCOS_MAX_TOKENS_PER_RANK:-512}
catccos_min_tokens=${CATCCOS_MIN_TOKENS:-1}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
model=${MODEL:-/home/weights/Qwen3-30B-A3B-Instruct-2507}
served_model_name=${SERVED_MODEL_NAME:-qwen3-catccos}
port=${PORT:-28001}
catccos_root=${CATCCOS_ROOT:-/home/z00956592/catccos}
catccos_library=${CATCCOS_LIBRARY:-${catccos_root}/build_torch_a5/lib/libcatccos_torch.so}
catccos_store_url=${CATCCOS_STORE_URL:-tcp://127.0.0.1:27020}
vllm_bin=${VLLM_BIN:-vllm}

if [[ -e "${profile_dir}" ]]; then
    echo "Profile directory already exists: ${profile_dir}" >&2
    echo "Use a new PROFILE_TAG or remove the old directory explicitly." >&2
    exit 1
fi
mkdir -p "${profile_dir}"

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_V1=1
export OMP_NUM_THREADS=1
export MSMONITOR_USE_DAEMON=0
export VLLM_ASCEND_MOE_PROFILE_RANGES=1
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=${sync_boundaries}

profiler_config=$(cat <<EOF
{"profiler":"torch","torch_profiler_dir":"${profile_dir}","torch_profiler_with_stack":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"max_iterations":${profile_iters}}
EOF
)

if [[ "${backend}" == "catccos" ]]; then
    if [[ ! -f "${catccos_library}" ]]; then
        echo "CatCCOS library not found: ${catccos_library}" >&2
        exit 1
    fi
    catccos_library_dir=$(dirname "${catccos_library}")
    export LD_LIBRARY_PATH="${catccos_library_dir}:${catccos_root}/3rdparty/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
    additional_config=$(cat <<EOF
{"enable_fused_mc2":1,"fused_mc2_backend":"catccos","catccos_library_path":"${catccos_library}","catccos_store_url":"${catccos_store_url}","catccos_local_mem_size":1073741824,"catccos_max_tokens_per_rank":${catccos_max_tokens_per_rank},"catccos_sync_after_launch":${catccos_sync_after_launch},"catccos_min_tokens":${catccos_min_tokens},"enable_prefill_mc2":true}
EOF
    )
else
    additional_config='{"enable_fused_mc2":0,"fused_mc2_backend":"auto","enable_prefill_mc2":true}'
fi

echo "backend=${backend}"
echo "profile_dir=${profile_dir}"
echo "sync_boundaries=${sync_boundaries}"
echo "max_num_batched_tokens=${max_num_batched_tokens}"
echo "additional_config=${additional_config}"
echo "Actual selected backend and global/rank-local M are recorded in vllm_ascend.moe.*.prepare/finalize profiler ranges."

exec "${vllm_bin}" serve "${model}" \
    --served-model-name "${served_model_name}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --max-model-len 8192 \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.80 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "${port}" \
    --additional-config "${additional_config}" \
    --profiler-config "${profiler_config}"
