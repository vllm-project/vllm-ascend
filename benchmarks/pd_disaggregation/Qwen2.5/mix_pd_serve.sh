#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/python/site-packages:${LD_LIBRARY_PATH:-}"
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPT_DIR}/../../../examples/disaggregated_prefill_v1/mooncake.example.json}"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/../../../../vllm-ascend/"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../../../../vllm/"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

# Optional profiler settings:
# export VLLM_TORCH_PROFILER_DIR=/path/to/profiling/vllm
# export VLLM_TORCH_PROFILER_WITH_STACK=0

export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

NPU_NUM="${NPU_NUM:-2}"
CPU_CORES=$(nproc --all)
CORES_PER_NPU=$((CPU_CORES / NPU_NUM))
CPU_AFFINITY_CONF_TMP=1
if [ "$NPU_NUM" -gt 0 ]; then
    for ((i = 0; i < NPU_NUM; i++)); do
        start_core=$((i * CORES_PER_NPU))
        end_core=$((start_core + CORES_PER_NPU - 1))
        CPU_AFFINITY_CONF_TMP+=",npu${i}:${start_core}-${end_core}"
    done
fi
export CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF_TMP
echo "CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

vllm serve "${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}" \
  --config qwen2.5_lmcache_config.sh
